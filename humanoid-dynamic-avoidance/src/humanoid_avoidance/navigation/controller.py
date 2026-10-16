"""Local planner: finite-state machine on top of a goal heading controller.

States
------
GO                  drive toward the goal (heading controller, vx <= max_vx_mps)
SLOW                risk detected but closest approach is far in time -> vx <= slow_vx_mps
YIELD               imminent risk -> vx = 0; station keeping: heading held toward the goal and a small
                    body-frame vy cancels lateral drift from the point where the yield began
                    (G1 keeps stepping in place at zero command and drifts ~0.3 m / 10 s)
RESUME              risk-free for safe_resume_duration_s -> ramp vx over resume_ramp_s -> GO
STOP_SENSOR_FAILURE scan older than stale_scan_timeout_s (or never received after the grace
                    period), or non-finite ego pose -> zero command
GOAL_REACHED        goal within goal_tolerance_m -> zero command (latched until reset())
MANUAL              mode "manual" only: heading controller toward goal, no obstacle logic

The two obstacle modes differ ONLY in `_risk_level()`:
- reactive:   latest detection centres in the body frame; YIELD if 0 < x < stop_distance_m and
              |y| < corridor_half_width_m. No velocity, no TCA, no SLOW level.
- predictive: `prediction.risk.assess_risk` over confirmed tracks (TCA / dCA with covariance
              margin). risk and tca <= yield_tca_s -> YIELD; risk and tca <= slow_tca_s -> SLOW.
Heading controller, speed limits, slew limits, hysteresis and resume ramp are shared.

Units: m, s, rad, m/s, rad/s. All time arithmetic uses the simulation time passed to `step`.
All thresholds come from the cfg dict (configs/default.yaml).
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

from humanoid_avoidance.types import NO_RISK, Detection, NavCommand, RiskAssessment, RobotState, Track

from humanoid_avoidance.geometry.transforms import world_to_body_vel as _world_to_body
from humanoid_avoidance.geometry.transforms import wrap_angle as _wrap_angle


GO = "GO"
SLOW = "SLOW"
YIELD = "YIELD"
RESUME = "RESUME"
STOP_SENSOR_FAILURE = "STOP_SENSOR_FAILURE"
GOAL_REACHED = "GOAL_REACHED"
MANUAL = "MANUAL"
STATES = (GO, SLOW, YIELD, RESUME, STOP_SENSOR_FAILURE, GOAL_REACHED, MANUAL)
MODES = ("reactive", "predictive", "manual")

# risk levels returned by _risk_level()
_LVL_NONE, _LVL_SLOW, _LVL_YIELD = 0, 1, 2

RiskFn = Callable[[RobotState, list, float, dict, float], RiskAssessment]


class NavController:
    """FSM local planner. See module docstring.

    Args:
        cfg: full config dict (uses cfg["navigation"], cfg["prediction"], cfg["tracking"]).
        mode: "reactive" | "predictive" | "manual".
        risk_fn: optional replacement for `prediction.risk.assess_risk` with the same signature
            `(robot, tracks, robot_radius_m, cfg_prediction, sigma_a) -> RiskAssessment`
            (used by unit tests). Default: the real assess_risk, imported lazily.

    Attributes:
        debug: dict refreshed every step with keys "state", "risk", "tca_s", "dca_m", "score",
            "margin_m", "reason", "track_id", "risk_level", "risk_free_s".
    """

    def __init__(self, cfg: dict, mode: str, risk_fn: Optional[RiskFn] = None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.cfg = cfg
        self.mode = mode
        self._risk_fn = risk_fn
        nav = cfg["navigation"]
        self.max_vx = float(nav["max_vx_mps"])
        self.slow_vx = min(float(nav["slow_vx_mps"]), self.max_vx)
        self.max_wz = float(nav["max_yaw_rate_rps"])
        self.k_yaw = float(nav["k_yaw"])
        self.heading_slowdown = float(nav["heading_slowdown_rad"])
        self.goal_tol = float(nav["goal_tolerance_m"])
        self.stale_timeout = float(nav["stale_scan_timeout_s"])
        # grace period before the first scan is required ("None after 1 s" in docs/interfaces.md)
        self.first_scan_grace = float(nav.get("first_scan_grace_s", 1.0))
        self.safe_resume = float(nav["safe_resume_duration_s"])
        self.resume_ramp = float(nav["resume_ramp_s"])
        self.accel_limit = float(nav["accel_limit_mps2"])
        self.yaw_accel_limit = float(nav["yaw_accel_limit_rps2"])
        self.robot_radius = float(nav.get("robot_radius_m", nav["nominal_robot_radius_m"]))
        self.slow_tca = float(nav["slow_tca_s"])
        self.yield_tca = float(nav["yield_tca_s"])
        self.reactive_stop = float(nav["reactive"]["stop_distance_m"])
        self.reactive_half_w = float(nav["reactive"]["corridor_half_width_m"])
        self.cfg_prediction = cfg.get("prediction", {})
        self.sigma_a = float(cfg.get("tracking", {}).get("sigma_accel_mps2", 0.7))
        self.hold_k_lat = float(nav.get("hold_k_lat", 0.0))
        self.hold_max_vy = float(nav.get("hold_max_vy_mps", 0.0))
        self.debug: dict = {}
        self.reset()

    # ------------------------------------------------------------------ public
    def reset(self) -> None:
        """Clear FSM state, timers and the slew-limiter memory (command back to zero)."""
        self.state = MANUAL if self.mode == "manual" else GO
        self._t_prev: Optional[float] = None
        self._t0: Optional[float] = None
        self._vx = 0.0                 # last commanded vx (m/s), slew limiter memory
        self._wz = 0.0                 # last commanded yaw rate (rad/s)
        self._risk_free_since: Optional[float] = None
        self._resume_t0 = 0.0
        self._resume_v0 = 0.0
        self._goal_latched = False
        self._hold_xy: Optional[np.ndarray] = None
        self.debug = self._blank_debug()

    def step(self, time_s: float, robot: RobotState, goal_xy: np.ndarray,
             tracks: list[Track], detections: list[Detection],
             last_scan_time_s: Optional[float], heading_target_xy=None) -> NavCommand:
        """Advance the FSM to sim time `time_s` and return the (slew-limited) body-frame command.

        Args:
            time_s: simulation time, s (monotonic; ValueError if it goes backwards).
            robot: ego state (world frame).
            goal_xy: (2,) goal position, world frame, m.
            tracks: tracker output (only confirmed tracks are used, predictive mode).
            detections: latest clustered detections (reactive mode only).
            last_scan_time_s: sim time of the newest processed scan, or None if none yet.
            heading_target_xy: optional scan-route waypoint; goal completion uses goal_xy.
        Returns:
            NavCommand(vx, vy=0, yaw_rate, mode=<FSM state>).
        """
        t = float(time_s)
        if self._t_prev is not None and t < self._t_prev - 1e-9:
            raise ValueError(f"time went backwards: {t} < {self._t_prev}")
        dt = 0.0 if self._t_prev is None else max(0.0, t - self._t_prev)
        if self._t0 is None:
            self._t0 = t
        self._t_prev = t

        dbg = self._blank_debug()
        goal_xy = np.asarray(goal_xy, dtype=float)
        xy = np.asarray(robot.xy_world_m, dtype=float)
        pose_ok = bool(np.all(np.isfinite(xy)) and math.isfinite(float(robot.yaw_rad)))
        vx_nom, wz_nom, goal_dist = self._heading_control(xy, float(robot.yaw_rad), goal_xy) \
            if pose_ok else (0.0, 0.0, float("nan"))
        route_blocked = False
        if pose_ok and heading_target_xy is not None:
            vx_nom, wz_nom, route_dist = self._heading_control(xy, float(robot.yaw_rad), heading_target_xy)
            direction = np.asarray(heading_target_xy) - xy
            error = float(_wrap_angle(math.atan2(direction[1], direction[0]) - float(robot.yaw_rad)))
            vx_nom *= max(0.0, 1.0 - abs(error) / 0.45)
            if route_dist < 1e-6:
                vx_nom, wz_nom = 0.0, 0.0
                route_blocked = True

        # ---- 1. goal reached (latched) ------------------------------------------------
        if pose_ok and goal_dist <= self.goal_tol:
            self._goal_latched = True
        if self._goal_latched:
            self.state = GOAL_REACHED
            dbg["reason"] = f"goal within {self.goal_tol:.2f} m"
            return self._emit(0.0, 0.0, dt, dbg)

        # ---- 2. manual: locomotion check, no obstacle / sensor logic ---------------------
        if self.mode == "manual":
            self.state = MANUAL
            dbg["reason"] = "manual: heading controller only, no obstacle logic"
            return self._emit(vx_nom, wz_nom, dt, dbg)

        # ---- 3. sensor / pose validity ------------------------------------------------
        fail_reason = None
        if not pose_ok:
            fail_reason = "non-finite ego pose"
        elif last_scan_time_s is None:
            if t - self._t0 > self.first_scan_grace:
                fail_reason = f"no scan received after {self.first_scan_grace:.2f} s"
            else:
                fail_reason = "awaiting first scan"
        elif t - float(last_scan_time_s) > self.stale_timeout:
            fail_reason = f"scan age {t - float(last_scan_time_s):.3f} s > {self.stale_timeout:.3f} s"
        if fail_reason is not None:
            self.state = STOP_SENSOR_FAILURE
            self._risk_free_since = None
            dbg["reason"] = fail_reason
            return self._emit(0.0, 0.0, dt, dbg)

        if route_blocked:
            self.state = YIELD
            self._risk_free_since = None
            dbg["reason"] = "no free route in range-scan map"
            return self._emit(0.0, 0.0, dt, dbg)

        # ---- 4. risk (the ONLY mode-specific part) -------------------------------------
        level, risk_dbg = self._risk_level(robot, tracks, detections)
        dbg.update(risk_dbg)
        if level == _LVL_NONE:
            if self._risk_free_since is None:
                self._risk_free_since = t
        else:
            self._risk_free_since = None
        risk_free_s = 0.0 if self._risk_free_since is None else t - self._risk_free_since
        dbg["risk_level"] = level
        dbg["risk_free_s"] = risk_free_s

        # ---- 5. transitions ------------------------------------------------------------
        prev = self.state
        holding = prev in (YIELD, SLOW, STOP_SENSOR_FAILURE)
        if level == _LVL_YIELD:
            self.state = YIELD
        elif level == _LVL_SLOW:
            # a SLOW-level risk does not release YIELD (exit needs a risk-free period)
            self.state = YIELD if prev == YIELD else SLOW
        else:  # risk-free now
            if holding:
                if risk_free_s >= self.safe_resume - 1e-9:
                    self.state = RESUME
                    self._resume_t0 = t
                    self._resume_v0 = self._vx
                else:  # hysteresis: keep holding the current stop/slow state
                    self.state = prev
            elif prev == RESUME and t - self._resume_t0 >= self.resume_ramp - 1e-9:
                self.state = GO
            elif prev not in (RESUME, GO):
                self.state = GO

        # ---- 6. targets ----------------------------------------------------------------
        vy_t = 0.0
        if self.state == YIELD:
            if prev != YIELD or self._hold_xy is None:
                self._hold_xy = xy.copy()
            lat = _world_to_body(xy - self._hold_xy, float(robot.yaw_rad))[1]
            vy_t = float(np.clip(-self.hold_k_lat * lat, -self.hold_max_vy, self.hold_max_vy))
            vx_t, wz_t = 0.0, wz_nom
        elif self.state == STOP_SENSOR_FAILURE:
            vx_t, wz_t = 0.0, 0.0
        elif self.state == SLOW:
            vx_t, wz_t = min(vx_nom, self.slow_vx), wz_nom
        elif self.state == RESUME:
            frac = 1.0 if self.resume_ramp <= 0 else min(1.0, (t - self._resume_t0) / self.resume_ramp)
            v0 = min(self._resume_v0, vx_nom)
            vx_t, wz_t = v0 + (vx_nom - v0) * frac, wz_nom
        else:  # GO
            vx_t, wz_t = vx_nom, wz_nom

        if not dbg["reason"]:
            dbg["reason"] = {YIELD: "risk: yield", SLOW: "risk: slow",
                             RESUME: f"risk-free {risk_free_s:.2f} s, ramping",
                             GO: "clear"}.get(self.state, "")
            if self.state in (YIELD, SLOW, STOP_SENSOR_FAILURE) and level == _LVL_NONE:
                dbg["reason"] = f"hysteresis: risk-free {risk_free_s:.2f}/{self.safe_resume:.2f} s"
        if self.state != YIELD:
            self._hold_xy = None
        return self._emit(vx_t, wz_t, dt, dbg, vy_target=vy_t)

    # ------------------------------------------------------------------ internals
    def _blank_debug(self) -> dict:
        return {"state": self.state if hasattr(self, "state") else GO, "risk": False,
                "tca_s": float("nan"), "dca_m": float("nan"), "score": 0.0,
                "margin_m": float("nan"), "reason": "", "track_id": -1,
                "risk_level": _LVL_NONE, "risk_free_s": 0.0}

    def _heading_control(self, xy: np.ndarray, yaw: float, goal_xy: np.ndarray):
        """Return (vx_nominal m/s, yaw_rate rad/s, goal distance m) — shared by all modes.

        yaw_rate = clip(k_yaw * wrap(heading_goal - yaw), +-max_yaw_rate). vx = max_vx, scaled
        linearly down to 0 between |err| = heading_slowdown_rad and |err| = pi/2.
        """
        d = goal_xy - xy
        dist = float(np.hypot(d[0], d[1]))
        err = float(_wrap_angle(math.atan2(d[1], d[0]) - yaw))
        wz = float(np.clip(self.k_yaw * err, -self.max_wz, self.max_wz))
        a = abs(err)
        if a <= self.heading_slowdown:
            scale = 1.0
        else:
            span = max(math.pi / 2.0 - self.heading_slowdown, 1e-6)
            scale = float(np.clip((math.pi / 2.0 - a) / span, 0.0, 1.0))
        return self.max_vx * scale, wz, dist

    def _risk_level(self, robot: RobotState, tracks: list[Track], detections: list[Detection]):
        if self.mode == "reactive":
            return self._reactive_risk(robot, detections)
        return self._predictive_risk(robot, tracks)

    def _reactive_risk(self, robot: RobotState, detections: list[Detection]):
        """Latest detection centres only; front-corridor box in the body frame."""
        xy = np.asarray(robot.xy_world_m, dtype=float)
        nearest = float("inf")
        for det in detections or []:
            rel = _world_to_body(np.asarray(det.xy_world_m, dtype=float) - xy, float(robot.yaw_rad))
            if 0.0 < rel[0] < self.reactive_stop and abs(rel[1]) < self.reactive_half_w:
                nearest = min(nearest, float(np.hypot(rel[0], rel[1])))
        hit = math.isfinite(nearest)
        dbg = {"risk": hit, "dca_m": nearest if hit else float("nan"), "score": 1.0 if hit else 0.0,
               "margin_m": 0.0, "tca_s": float("nan"), "track_id": -1}
        if hit:
            dbg["reason"] = f"reactive: detection {nearest:.2f} m ahead in corridor"
        return (_LVL_YIELD if hit else _LVL_NONE), dbg

    def _predictive_risk(self, robot: RobotState, tracks: list[Track]):
        """TCA / dCA risk over confirmed tracks via prediction.risk.assess_risk."""
        fn = self._risk_fn
        if fn is None:
            from humanoid_avoidance.prediction.risk import assess_risk as fn  # lazy import
            self._risk_fn = fn
        confirmed = [tr for tr in (tracks or []) if tr.confirmed]
        ra = fn(robot, confirmed, self.robot_radius, self.cfg_prediction, self.sigma_a) \
            if confirmed else NO_RISK
        dbg = {"risk": bool(ra.risk), "tca_s": float(ra.tca_s), "dca_m": float(ra.dca_m),
               "score": float(ra.score), "margin_m": float(ra.margin_m),
               "track_id": int(ra.track_id)}
        level = _LVL_NONE
        if ra.risk and math.isfinite(ra.tca_s):
            if ra.tca_s <= self.yield_tca:
                level = _LVL_YIELD
                dbg["reason"] = f"predictive: tca {ra.tca_s:.2f} s <= {self.yield_tca:.2f}, dca {ra.dca_m:.2f} m"
            elif ra.tca_s <= self.slow_tca:
                level = _LVL_SLOW
                dbg["reason"] = f"predictive: tca {ra.tca_s:.2f} s <= {self.slow_tca:.2f}, dca {ra.dca_m:.2f} m"
        return level, dbg

    def _emit(self, vx_target: float, wz_target: float, dt: float, dbg: dict,
              vy_target: float = 0.0) -> NavCommand:
        """Clip to limits, slew-limit with sim-time dt, store debug, build the NavCommand."""
        vx_target = float(np.clip(vx_target, 0.0, self.max_vx))   # no reversing (G1 policy)
        wz_target = float(np.clip(wz_target, -self.max_wz, self.max_wz))
        dv = self.accel_limit * dt
        dw = self.yaw_accel_limit * dt
        self._vx = float(np.clip(self._vx + np.clip(vx_target - self._vx, -dv, dv), 0.0, self.max_vx))
        self._wz = float(np.clip(self._wz + np.clip(wz_target - self._wz, -dw, dw),
                                 -self.max_wz, self.max_wz))
        dbg["state"] = self.state
        self.debug = dbg
        vy = float(np.clip(vy_target, -self.hold_max_vy, self.hold_max_vy))
        return NavCommand(vx_mps=self._vx, vy_mps=vy, yaw_rate_rps=self._wz, mode=self.state)
