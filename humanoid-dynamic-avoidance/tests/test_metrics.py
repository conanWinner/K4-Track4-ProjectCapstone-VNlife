"""Metrics / summary / CSV logger / plot smoke tests on synthetic data (no simulator)."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from humanoid_avoidance.config import load_config
from humanoid_avoidance.evaluation.metrics import (episode_metrics, prediction_errors,
                                                   robot_actor_contact, summarize,
                                                   summary_to_markdown)
from humanoid_avoidance.logging.csv_logger import LOG_COLUMNS, CsvLogger

DT = 0.1


@pytest.fixture()
def cfg():
    return load_config()


def make_df(n=50, states=None, goal_dist=None, contact=None, clearance=None, fall=None):
    t = np.arange(n) * DT
    df = pd.DataFrame({
        "sim_time_s": t, "scenario": "crossing_slow", "seed": 0, "mode": "predictive",
        "robot_x": 0.1 * t, "robot_y": 0.0,
        "nav_state": states if states is not None else ["GO"] * n,
        "goal_distance_m": goal_dist if goal_dist is not None else np.linspace(5.0, 0.2, n),
        "contact_flag": contact if contact is not None else np.zeros(n, int),
        "fall_flag": fall if fall is not None else np.zeros(n, int),
        "clearance_gt_m": clearance if clearance is not None else np.full(n, 2.0),
        "sensor_latency_ms": np.full(n, 1.0), "planner_latency_ms": np.linspace(0.1, 1.0, n),
    })
    return df


# ------------------------------------------------------------------ episode metrics
def test_goal_reached_and_time_to_goal(cfg):
    tol = cfg["navigation"]["goal_tolerance_m"]
    gd = np.linspace(3.0, 0.0, 31)        # 0.1 m per tick
    m = episode_metrics(make_df(31, goal_dist=gd), cfg)
    assert m["goal_reached"] and not m["collision"] and m["collision_free_success"]
    first = np.flatnonzero(gd <= tol)[0]
    assert m["time_to_goal_s"] == pytest.approx(first * DT)

    m2 = episode_metrics(make_df(31, goal_dist=np.linspace(3.0, 1.0, 31)), cfg)
    assert not m2["goal_reached"] and math.isnan(m2["time_to_goal_s"])
    assert not m2["collision_free_success"]


def test_collision_breaks_success(cfg):
    c = np.zeros(50, int)
    c[20:23] = 1
    c[30] = 1
    clr = np.full(50, 1.0)
    clr[21] = -0.03
    m = episode_metrics(make_df(50, contact=c, clearance=clr), cfg)
    assert m["goal_reached"] and m["collision"] and not m["collision_free_success"]
    assert m["collision_events"] == 2
    assert m["min_clearance_m"] == pytest.approx(-0.03)


def test_collision_min_duration(cfg):
    cfg["evaluation"]["collision_min_duration_s"] = 0.25
    c = np.zeros(50, int)
    c[10] = 1                 # 0.1 s -> ignored
    m = episode_metrics(make_df(50, contact=c), cfg)
    assert not m["collision"]
    c[30:34] = 1              # 0.4 s -> counted
    m = episode_metrics(make_df(50, contact=c), cfg)
    assert m["collision"] and m["collision_events"] == 1


def test_fall_breaks_success(cfg):
    f = np.zeros(50, int)
    f[40] = 1
    m = episode_metrics(make_df(50, fall=f), cfg)
    assert m["fall"] and not m["collision_free_success"]


def test_yield_count_and_total_time(cfg):
    s = ["GO"] * 10 + ["YIELD"] * 5 + ["RESUME"] * 3 + ["GO"] * 2 + ["YIELD"] * 2 + ["GO"] * 8
    m = episode_metrics(make_df(len(s), states=s), cfg)
    assert m["yield_count"] == 2
    assert m["total_yield_s"] == pytest.approx(7 * DT)
    assert m["slow_count"] == 0


def test_false_stop_definition(cfg):
    thr = cfg["evaluation"]["false_stop_clearance_m"]
    s = ["GO"] * 10 + ["YIELD"] * 10 + ["GO"] * 30 + ["YIELD"] * 5 + ["GO"] * 45
    n = len(s)
    clr = np.full(n, thr + 1.0)
    clr[60] = thr - 0.5        # second stop: actor comes close within the horizon window
    m = episode_metrics(make_df(n, states=s, clearance=clr), cfg)
    assert m["yield_count"] == 2
    assert m["false_stop_count"] == 1
    # no actor at all (clearance NaN) -> every stop is false
    m = episode_metrics(make_df(n, states=s, clearance=np.full(n, np.nan)), cfg)
    assert m["false_stop_count"] == 2 and math.isnan(m["min_clearance_m"])


def test_latencies_and_empty(cfg):
    m = episode_metrics(make_df(50), cfg)
    assert m["mean_sensor_latency_ms"] == pytest.approx(1.0)
    assert m["p95_planner_latency_ms"] <= 1.0
    e = episode_metrics(make_df(0), cfg)
    assert e["goal_reached"] is False and e["yield_count"] == 0


def test_prediction_error_exact_cv_is_zero():
    t = np.arange(60) * DT
    gx, gy = 2.0 + 0.0 * t, -1.5 + 0.5 * t          # actor crossing at 0.5 m/s
    df = pd.DataFrame({"sim_time_s": t, "obstacle_gt_x_eval_only": gx, "obstacle_gt_y_eval_only": gy,
                       "track_x": gx, "track_y": gy, "track_vx": 0.0, "track_vy": 0.5})
    e = prediction_errors(df, horizon_s=1.0, step_s=0.1)
    assert e["track_pos_error_m"] == pytest.approx(0.0, abs=1e-12)
    assert e["pred_ade_m"] == pytest.approx(0.0, abs=1e-9)
    assert e["pred_fde_m"] == pytest.approx(0.0, abs=1e-9)
    # stale velocity (0) -> FDE equals distance travelled in the horizon
    df["track_vy"] = 0.0
    e = prediction_errors(df, horizon_s=1.0, step_s=0.1)
    assert e["pred_fde_m"] == pytest.approx(0.5)


# ------------------------------------------------------------------ summarize
def test_summarize_groups_and_excludes_errors():
    rows = []
    for mode, succ in (("reactive", [True, False, False]), ("predictive", [True, True, False])):
        for seed, s in enumerate(succ):
            rows.append({"scenario": "crossing_fast", "mode": mode, "seed": seed, "error": "",
                         "collision_free_success": s, "collision": not s, "goal_reached": True,
                         "fall": False, "min_clearance_m": 0.1 * (seed + 1),
                         "time_to_goal_s": 20.0 + seed, "yield_count": seed, "total_yield_s": 1.0,
                         "false_stop_count": 0, "pred_ade_m": np.nan})
    rows.append({"scenario": "crossing_fast", "mode": "predictive", "seed": 9,
                 "error": "RuntimeError: boom"})
    rows.append({"scenario": "empty", "mode": "predictive", "seed": 0, "error": "",
                 "collision_free_success": True, "collision": False, "goal_reached": True,
                 "fall": False, "min_clearance_m": np.nan, "time_to_goal_s": np.nan})
    s = summarize(pd.DataFrame(rows))
    assert list(s[["scenario", "mode"]].itertuples(index=False, name=None)) == [
        ("crossing_fast", "reactive"), ("crossing_fast", "predictive"), ("empty", "predictive")]
    r = s.iloc[0]
    assert r["episodes"] == 3 and r["errors"] == 0
    assert r["collision_free_success"] == pytest.approx(1 / 3)
    assert r["collision_rate"] == pytest.approx(2 / 3)
    assert r["mean_min_clearance_m"] == pytest.approx(0.2)
    assert r["mean_time_to_goal_s"] == pytest.approx(21.0)
    p = s.iloc[1]
    assert p["episodes"] == 3 and p["errors"] == 1
    assert p["collision_free_success"] == pytest.approx(2 / 3)
    assert math.isnan(p["mean_pred_ade_m"])
    e = s.iloc[2]
    assert math.isnan(e["mean_min_clearance_m"]) and math.isnan(e["mean_time_to_goal_s"])
    md = summary_to_markdown(s, ["scenario", "mode", "episodes", "collision_free_success"])
    assert "| crossing_fast | reactive | 3 | 33% |" in md
    assert "n/a" in summary_to_markdown(s)


# ------------------------------------------------------------------ CSV logger
def test_csv_logger_round_trip(tmp_path, cfg):
    path = tmp_path / "run" / "log.csv"
    meta = {"robot": "amr", "seed": np.int64(3), "goal": np.array([5.0, 0.0])}
    with CsvLogger(path, meta) as log:
        for i in range(30):
            log.log({"sim_time_s": i * DT, "scenario": "crossing_slow", "seed": 3, "mode": "reactive",
                     "robot_x": np.float64(0.1 * i), "nav_state": "YIELD" if 10 <= i < 15 else "GO",
                     "risk": np.bool_(10 <= i < 15), "contact_flag": False,
                     "track_x": float("nan"), "dca_m": float("inf"), "detection_x": None,
                     "goal_distance_m": 5.0 - 0.17 * i, "clearance_gt_m": 1.0,
                     "not_a_column": 123})
    assert log.closed and log.rows_written == 30
    assert json.loads((path.parent / "metadata.json").read_text())["goal"] == [5.0, 0.0]
    df = pd.read_csv(path)
    assert list(df.columns) == LOG_COLUMNS
    assert len(df) == 30
    assert df["robot_x"].iloc[7] == pytest.approx(0.7)
    assert df["track_x"].isna().all() and df["detection_x"].isna().all()
    assert np.isinf(df["dca_m"]).all()
    assert df["risk"].sum() == 5 and df["contact_flag"].sum() == 0
    assert "not_a_column" not in df.columns
    m = episode_metrics(df, cfg)
    assert m["yield_count"] == 1 and m["total_yield_s"] == pytest.approx(0.5)
    assert m["goal_reached"]


def test_log_columns_match_contract():
    expected = ("sim_time_s, scenario, seed, mode, robot_x, robot_y, robot_yaw, robot_vx, robot_vy, "
                "obstacle_gt_x_eval_only, obstacle_gt_y_eval_only, detection_x, detection_y, num_detections, "
                "track_id, track_x, track_y, track_vx, track_vy, track_P_xx, track_P_yy, tca_s, dca_m, "
                "margin_m, risk, risk_score, nav_state, cmd_vx, cmd_vy, cmd_wz, contact_flag, fall_flag, "
                "goal_distance_m, clearance_gt_m, sensor_latency_ms, planner_latency_ms")
    assert LOG_COLUMNS == [c.strip() for c in expected.split(",")]


# ------------------------------------------------------------------ contact (needs mujoco)
def test_robot_actor_contact():
    mujoco = pytest.importorskip("mujoco")
    xml = """
    <mujoco><worldbody>
      <geom name="floor" type="plane" size="5 5 0.1"/>
      <body name="robot" pos="0 0 0.3"><freejoint/><geom name="rg" type="sphere" size="0.3"/></body>
      <body name="actor" mocap="true" pos="{ax} 0 0.3"><geom name="ag" type="cylinder" size="0.25 0.3"/></body>
    </worldbody></mujoco>"""
    for ax, expect in ((0.5, True), (1.5, False)):
        model = mujoco.MjModel.from_xml_string(xml.format(ax=ax))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        rb = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot")}
        ag = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ag")}
        assert data.ncon >= 1                       # at least robot-floor contact
        assert robot_actor_contact(model, data, rb, ag) is expect


# ------------------------------------------------------------------ plots (smoke)
def test_plots_smoke(tmp_path, cfg):
    from humanoid_avoidance.evaluation.plots import (plot_comparison, plot_episode_topdown,
                                                     plot_prediction_error)
    path = tmp_path / "log.csv"
    with CsvLogger(path, {}) as log:
        for i in range(60):
            t = i * DT
            log.log({"sim_time_s": t, "scenario": "crossing_slow", "seed": 0, "mode": "predictive",
                     "robot_x": 0.05 * i, "robot_y": 0.0, "nav_state": "YIELD" if 20 <= i < 30 else "GO",
                     "obstacle_gt_x_eval_only": 2.5, "obstacle_gt_y_eval_only": -1.5 + 0.05 * i,
                     "detection_x": 2.4, "detection_y": -1.5 + 0.05 * i,
                     "track_x": 2.4, "track_y": -1.5 + 0.05 * i, "track_vx": 0.0, "track_vy": 0.5,
                     "contact_flag": i == 40})
    assert plot_episode_topdown(path, tmp_path / "f" / "top.png", cfg).stat().st_size > 0
    assert plot_prediction_error(path, tmp_path / "f" / "pe.png").stat().st_size > 0
    summ = pd.DataFrame([{"scenario": "crossing_slow", "mode": m, "episodes": 2,
                          "collision_free_success": v, "mean_min_clearance_m": 0.3,
                          "mean_time_to_goal_s": np.nan if m == "reactive" else 22.0}
                         for m, v in (("reactive", 0.5), ("predictive", 1.0))])
    assert plot_comparison(summ, tmp_path / "f" / "cmp.png").stat().st_size > 0
