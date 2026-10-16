import numpy as np
import pandas as pd

from humanoid_avoidance.config import load_config
from humanoid_avoidance.navigation.controller import NavController
from humanoid_avoidance.navigation.scan_route import ScanRoute
from humanoid_avoidance.sim.runner import run_episode
from humanoid_avoidance.types import RobotState, Scan


def test_scan_creates_detour_and_blocked_route_stops():
    cfg = load_config()
    route = ScanRoute(cfg, 0.42)
    start, goal = np.array([0., 0.]), np.array([5., 0.])
    assert max(abs(p[1]) for p in route.path(start, goal)) < 0.11
    scan = Scan(0., np.array([0.]), np.array([2.]), np.array([True]))
    route.update(scan)
    path = route.path(start, goal)
    assert path and max(abs(p[1]) for p in path) > 0.8
    assert all(np.linalg.norm(p - [2., 0.]) > route.padding for p in path)
    route.blocked[15, :] = True  # wall spanning the corridor
    assert not route.path(start, goal)
    assert np.array_equal(route.target(start, goal), start)
    nav = NavController(cfg, "predictive")
    robot = RobotState(0., start, 0., np.zeros(2), 0.)
    for t in np.arange(0, 1, 0.1):
        cmd = nav.step(t, robot, goal, [], [], t, heading_target_xy=start)
    assert cmd.vx_mps == 0 and cmd.yaw_rate_rps == 0
    assert cmd.mode != "GOAL_REACHED"
    assert cmd.mode == "YIELD"


def test_obstacle_course_reaches_goal_without_contacts(tmp_path):
    cfg = load_config()
    metrics = run_episode(cfg, "obstacle_course", "predictive", 42,
                          robot_name="amr", run_dir=tmp_path)
    assert metrics["collision_free_success"]
    assert metrics["min_clearance_m"] > 0.15
    log = pd.read_csv(metrics["csv_path"])
    assert log.robot_y.abs().max() > 0.8  # robot actually goes around the first box
    assert (log.num_detections > 0).any()
    assert not log.contact_flag.any()


def test_box_collision_is_counted(tmp_path):
    cfg = load_config(overrides={"simulation": {"episode_timeout_s": 8.0}})
    metrics = run_episode(cfg, "obstacle_course", "manual", 42,
                          robot_name="amr", run_dir=tmp_path)
    assert metrics["collision"]
    assert not metrics["collision_free_success"]
