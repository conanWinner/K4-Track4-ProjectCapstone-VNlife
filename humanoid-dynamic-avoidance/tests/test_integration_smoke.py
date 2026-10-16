"""End-to-end smoke tests on the AMR proxy (fast, headless). G1 end-to-end is exercised by
scripts/evaluate.py --robot g1 (slower) and tests/test_g1_adapter.py."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("mujoco")

from humanoid_avoidance.config import load_config
from humanoid_avoidance.sim.runner import run_episode


@pytest.fixture(scope="module")
def cfg():
    return load_config(overrides={"simulation": {"episode_timeout_s": 40.0}})


def test_empty_scene_reaches_goal(cfg, tmp_path):
    m = run_episode(cfg, "empty", "predictive", seed=0, robot_name="amr", run_dir=tmp_path)
    assert m["goal_reached"]
    assert not m["collision"]
    df = pd.read_csv(m["csv_path"])
    assert df["sim_time_s"].is_monotonic_increasing
    assert (df["num_detections"] == 0).all(), "walls must be filtered out of detections"


def test_predictive_yields_and_resumes_on_crossing(cfg, tmp_path):
    m = run_episode(cfg, "crossing_slow", "predictive", seed=0, robot_name="amr", run_dir=tmp_path)
    df = pd.read_csv(m["csv_path"])
    assert (df["nav_state"] == "YIELD").any(), "predictive mode should yield to the crossing actor"
    assert m["goal_reached"] and not m["collision"]
    tracked = df.dropna(subset=["track_vy"])
    assert len(tracked) > 5
    # actor moves +y at ~0.5 m/s: tracked velocity should be clearly positive in y at some point
    assert tracked["track_vy"].max() > 0.25


def test_sensor_blackout_triggers_safe_stop(tmp_path):
    cfg = load_config(overrides={"simulation": {"episode_timeout_s": 8.0},
                                 "sensor": {"blackout_s": [3.0, 5.0]}})
    m = run_episode(cfg, "empty", "predictive", seed=0, robot_name="amr", run_dir=tmp_path)
    df = pd.read_csv(m["csv_path"])
    stopped = df[df["nav_state"] == "STOP_SENSOR_FAILURE"]
    assert len(stopped) > 0
    assert stopped["sim_time_s"].min() > 3.0 and stopped["sim_time_s"].max() <= 5.7
    # stale after 0.3 s, then slew-limited braking (0.3 m/s at 0.5 m/s^2 -> 0.6 s)
    assert (df[(df.sim_time_s > 4.0) & (df.sim_time_s < 5.0)]["cmd_vx"] < 1e-6).all()


def test_perception_does_not_see_gt(cfg, tmp_path):
    """Detection error vs GT is bounded by actor radius + noise (visible-surface centroid)."""
    m = run_episode(cfg, "crossing_slow", "predictive", seed=1, robot_name="amr", run_dir=tmp_path)
    df = pd.read_csv(m["csv_path"]).dropna(subset=["detection_x", "obstacle_gt_x_eval_only"])
    err = np.hypot(df["detection_x"] - df["obstacle_gt_x_eval_only"],
                   df["detection_y"] - df["obstacle_gt_y_eval_only"])
    assert len(err) > 0
    assert err.median() < cfg["world"]["actor_radius_m"] + 0.15
