"""Single simulation loop: scenario -> physics -> range scan -> clustering -> tracking -> planner -> robot.

Clock: simulation time (data.time). Wall clock is used only for latency profiling and realtime pacing.
Ground truth actor pose is read ONLY for logging/evaluation columns (`*_eval_only`, `clearance_gt_m`).
"""
from __future__ import annotations

import copy
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

from humanoid_avoidance.config import OUTPUT_DIR, PROJECT_ROOT, _deep_update, load_scenarios
from humanoid_avoidance.evaluation.metrics import episode_metrics, robot_actor_contact
from humanoid_avoidance.logging.csv_logger import CsvLogger
from humanoid_avoidance.navigation.controller import NavController
from humanoid_avoidance.perception.clustering import cluster_scan
from humanoid_avoidance.perception.range_sensor import RangeSensor
from humanoid_avoidance.prediction.risk import predict_track
from humanoid_avoidance.sim.overlay import OverlayState, draw_overlays, hud_lines
from humanoid_avoidance.sim.robot_base import make_robot
from humanoid_avoidance.sim.scenario_controller import ScenarioController
from humanoid_avoidance.sim.scene_builder import build_scene
from humanoid_avoidance.tracking.multi_tracker import MultiTracker
from humanoid_avoidance.types import NavCommand

END_HOLD_S = 1.0          # keep simulating this long after goal reached / fall, then stop


def _upstream_sha() -> str:
    p = PROJECT_ROOT / "configs" / "upstream_version.txt"
    return p.read_text(encoding="utf-8").strip() if p.exists() else "n/a"


def _setup_camera(cam: mujoco.MjvCamera, cfg: dict) -> None:
    gx, gy = cfg["simulation"]["goal_xy"]
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [gx / 2.0, 0.0, 0.4]
    cam.distance = 7.5
    cam.elevation = -50.0
    cam.azimuth = 90.0 + 25.0


def _vis_option(cfg: dict, opt: Optional[mujoco.MjvOption] = None) -> mujoco.MjvOption:
    """MuJoCo renders geom groups 0-2 by default; walls/actor live in the perceivable group (3)."""
    opt = opt if opt is not None else mujoco.MjvOption()
    opt.geomgroup[int(cfg["world"]["perceivable_geom_group"])] = 1
    return opt


class _Recorder:
    """Offscreen MP4 recorder at fixed sim-time fps, with HUD text burned in."""

    def __init__(self, model, cfg, path: Path, fps: int = 30, width: int = 960, height: int = 540):
        import imageio.v2 as imageio
        self.opt = _vis_option(cfg)
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.cam = mujoco.MjvCamera()
        _setup_camera(self.cam, cfg)
        self.period = 1.0 / fps
        self.next_t = 0.0
        path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=7,
                                         macro_block_size=8)
        self.path = path

    def maybe_capture(self, data, overlay: OverlayState) -> None:
        if data.time + 1e-9 < self.next_t:
            return
        self.next_t += self.period
        self.renderer.update_scene(data, camera=self.cam, scene_option=self.opt)
        draw_overlays(self.renderer.scene, overlay)
        frame = self.renderer.render().copy()
        try:
            import cv2
            for i, line in enumerate(hud_lines(overlay)):
                y = 28 + 26 * i
                cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        except ImportError:
            pass
        self.writer.append_data(frame)

    def close(self) -> None:
        self.writer.close()
        self.renderer.close()


def run_episode(cfg: dict, scenario_name: str, mode: str, seed: int, robot_name: str = "amr",
                headless: bool = True, record_path: Optional[Path] = None,
                run_dir: Optional[Path] = None, realtime: bool = False) -> dict:
    """Run one episode. Returns episode metrics dict + "csv_path" (+ "video_path")."""
    cfg = copy.deepcopy(cfg)
    _deep_update(cfg, copy.deepcopy((cfg.get("robot_overrides") or {}).get(robot_name) or {}))
    scenarios = load_scenarios()
    if scenario_name not in scenarios:
        raise KeyError(f"unknown scenario {scenario_name!r}; choose from {sorted(scenarios)}")
    sc_cfg = scenarios[scenario_name]

    robot = make_robot(robot_name, cfg)
    nav_cfg = cfg["navigation"]
    nav_cfg["max_vx_mps"] = float(min(nav_cfg["max_vx_mps"], robot.max_vx_mps))
    nav_cfg["robot_radius_m"] = float(robot.radius_m)
    blackout = cfg["sensor"].get("blackout_s") or None
    handles = build_scene(robot, cfg, sc_cfg)
    model, data = handles.model, handles.data

    start_xy = np.asarray(cfg["simulation"]["start_xy"], dtype=float)
    goal_xy = np.asarray(cfg["simulation"]["goal_xy"], dtype=float)
    robot.reset(model, data, start_xy, float(cfg["simulation"]["start_yaw"]))

    rng = np.random.default_rng(seed)
    scenario = ScenarioController(sc_cfg, seed)
    sensor = RangeSensor(cfg, rng)
    tracker = MultiTracker(cfg["tracking"])
    nav = NavController(cfg, mode)
    geomgroup = np.zeros(6, dtype=np.uint8)
    geomgroup[handles.perceivable_group] = 1
    robot_bodies = robot.robot_body_ids(model)
    r_robot = robot.radius_m
    r_actor = float(cfg["world"]["actor_radius_m"])
    sigma_a = float(cfg["tracking"]["sigma_accel_mps2"])
    horizon = float(cfg["prediction"]["horizon_s"])

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(run_dir) if run_dir else OUTPUT_DIR / "runs"
    ep_dir = run_dir / f"{stamp}_{robot_name}_{scenario_name}_{mode}_s{seed}"
    csv_path = ep_dir / "log.csv"
    meta = {"robot": robot_name, "scenario": scenario_name, "mode": mode, "seed": seed,
            "upstream_sha": _upstream_sha(), "scenario_cfg": sc_cfg, "config": cfg,
            "localization": "simulator oracle (ego pose read from MuJoCo) — documented MVP assumption",
            "started": stamp}
    logger = CsvLogger(csv_path, meta)

    viewer = None
    if not headless:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
        _setup_camera(viewer.cam, cfg)
        with viewer.lock():
            _vis_option(cfg, viewer.opt)
    recorder = _Recorder(model, cfg, Path(record_path)) if record_path else None

    overlay = OverlayState(mode=mode, scenario=scenario_name, robot=robot_name, goal_xy=goal_xy)
    cmd = NavCommand(0.0, 0.0, 0.0, "GO")
    robot.set_command(cmd)
    last_scan_time: Optional[float] = None
    v_filt: Optional[np.ndarray] = None
    v_filt_t = 0.0
    v_tau = float(nav_cfg.get("velocity_filter_tau_s", 0.0))
    contact_since_log = False
    end_time: Optional[float] = None
    timeout = float(cfg["simulation"]["episode_timeout_s"])
    wall_start = time.perf_counter()
    viewer_period = 1.0 / 60.0
    next_viewer_t = 0.0

    try:
        while data.time < timeout:
            t = float(data.time)
            scenario.apply(model, data, handles)

            if sensor.due(t):
                # planner runs on the sensor cadence; during a blackout the scan message is lost,
                # so the tracker only predicts and the planner sees a stale last_scan_time.
                t0 = time.perf_counter()
                s_pos, s_rot = robot.sensor_pose(model, data)
                scan = sensor.scan(model, data, t, s_pos, s_rot, geomgroup)
                if blackout and blackout[0] <= t <= blackout[1]:
                    scan, dets = None, []
                    tracker.step(t, [])
                else:
                    dets = cluster_scan(scan, handles.wall_segments, cfg)
                    tracker.step(t, dets)
                    last_scan_time = t
                confirmed = tracker.confirmed_tracks()
                t1 = time.perf_counter()
                rstate = robot.robot_state(model, data)
                # low-pass the measured base velocity (gait sway) before it enters the risk model
                if v_filt is None:
                    v_filt = rstate.v_world_mps.copy()
                else:
                    alpha = 1.0 - np.exp(-(t - v_filt_t) / max(v_tau, 1e-6))
                    v_filt = v_filt + alpha * (rstate.v_world_mps - v_filt)
                v_filt_t = t
                rstate.v_world_mps = v_filt.copy()
                cmd = nav.step(t, rstate, goal_xy, confirmed, dets, last_scan_time)
                robot.set_command(cmd)
                t2 = time.perf_counter()

                # ---- evaluation-only ground truth (after the algorithm has decided) ----
                gt = scenario.gt_actor_xy_from_sim(data, handles)
                clearance = (float(np.linalg.norm(rstate.xy_world_m - gt)) - r_robot - r_actor
                             if gt is not None else float("nan"))
                fall = bool(robot.fallen(model, data))
                dbg = nav.debug
                main = None
                if confirmed:
                    tid = dbg.get("track_id", -1)
                    main = next((tr for tr in confirmed if tr.track_id == tid), None)
                    if main is None:
                        main = min(confirmed, key=lambda tr: np.linalg.norm(tr.state[:2] - rstate.xy_world_m))
                det0 = (min(dets, key=lambda d: np.linalg.norm(d.xy_world_m - rstate.xy_world_m))
                        if dets else None)
                logger.log({
                    "sim_time_s": t, "scenario": scenario_name, "seed": seed, "mode": mode,
                    "robot_x": rstate.xy_world_m[0], "robot_y": rstate.xy_world_m[1],
                    "robot_yaw": rstate.yaw_rad, "robot_vx": rstate.v_world_mps[0],
                    "robot_vy": rstate.v_world_mps[1],
                    "obstacle_gt_x_eval_only": gt[0] if gt is not None else float("nan"),
                    "obstacle_gt_y_eval_only": gt[1] if gt is not None else float("nan"),
                    "detection_x": det0.xy_world_m[0] if det0 else float("nan"),
                    "detection_y": det0.xy_world_m[1] if det0 else float("nan"),
                    "num_detections": len(dets),
                    "track_id": main.track_id if main else -1,
                    "track_x": main.state[0] if main else float("nan"),
                    "track_y": main.state[1] if main else float("nan"),
                    "track_vx": main.state[2] if main else float("nan"),
                    "track_vy": main.state[3] if main else float("nan"),
                    "track_P_xx": main.covariance[0, 0] if main else float("nan"),
                    "track_P_yy": main.covariance[1, 1] if main else float("nan"),
                    "tca_s": dbg.get("tca_s", float("nan")), "dca_m": dbg.get("dca_m", float("nan")),
                    "margin_m": dbg.get("margin_m", float("nan")), "risk": int(bool(dbg.get("risk", False))),
                    "risk_score": dbg.get("score", 0.0), "nav_state": cmd.mode,
                    "cmd_vx": cmd.vx_mps, "cmd_vy": cmd.vy_mps, "cmd_wz": cmd.yaw_rate_rps,
                    "contact_flag": int(contact_since_log), "fall_flag": int(fall),
                    "goal_distance_m": float(np.linalg.norm(goal_xy - rstate.xy_world_m)),
                    "clearance_gt_m": clearance,
                    "sensor_latency_ms": (t1 - t0) * 1e3, "planner_latency_ms": (t2 - t1) * 1e3,
                })
                contact_since_log = False

                # ---- overlay state for viewer / video ----
                overlay.update(time_s=t, scan=scan, detections=dets, tracks=confirmed, cmd=cmd,
                               debug=dbg, robot_xy=rstate.xy_world_m, clearance=clearance,
                               predicted=(predict_track(main, horizon, 0.25, sigma_a)[1] if main else None))

                if end_time is None and (cmd.mode == "GOAL_REACHED" or fall):
                    end_time = t + END_HOLD_S
                if end_time is not None and t >= end_time:
                    break

            if robot_actor_contact(model, data, robot_bodies, handles.actor_geom_ids):
                contact_since_log = True
                overlay.collided = True

            robot.pre_step(model, data)
            mujoco.mj_step(model, data)

            if recorder is not None:
                recorder.maybe_capture(data, overlay)
            if viewer is not None and data.time >= next_viewer_t:
                next_viewer_t += viewer_period
                if not viewer.is_running():
                    break
                with viewer.lock():
                    viewer.user_scn.ngeom = 0
                    draw_overlays(viewer.user_scn, overlay)
                    if hasattr(viewer, "set_texts"):
                        try:
                            viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150,
                                              mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                              "\n".join(hud_lines(overlay)), ""))
                        except Exception:
                            pass
                viewer.sync()
                if realtime:
                    lag = data.time - (time.perf_counter() - wall_start)
                    if lag > 0:
                        time.sleep(lag)
    finally:
        logger.close()
        if recorder is not None:
            recorder.close()
        if viewer is not None:
            viewer.close()

    wall = time.perf_counter() - wall_start
    import pandas as pd
    df = pd.read_csv(csv_path)
    metrics = episode_metrics(df, cfg)
    metrics.update({"robot": robot_name, "scenario": scenario_name, "mode": mode, "seed": seed,
                    "sim_time_s": float(data.time), "wall_time_s": wall,
                    "sim_speed_factor": float(data.time) / max(wall, 1e-9),
                    "csv_path": str(csv_path)})
    if recorder is not None:
        metrics["video_path"] = str(recorder.path)
    with open(ep_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({k: (v if not isinstance(v, (np.floating, np.integer, np.bool_)) else v.item())
                   for k, v in metrics.items()}, f, indent=2, default=str)
    return metrics
