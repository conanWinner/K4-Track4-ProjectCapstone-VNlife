"""Range sensor tests on the AMR corridor scene (MuJoCo ray casting)."""
import copy

import mujoco
import numpy as np
import pytest

from humanoid_avoidance.config import load_config
from humanoid_avoidance.perception.range_sensor import RangeSensor
from humanoid_avoidance.sim.amr_adapter import AmrAdapter
from humanoid_avoidance.sim.scene_builder import build_scene


def _noiseless_cfg():
    cfg = copy.deepcopy(load_config())
    cfg["sensor"]["noise_std_m"] = 0.0
    cfg["sensor"]["dropout_probability"] = 0.0
    return cfg


def _scene(cfg, actor_xy=None, robot_xy=(0.0, 0.0), yaw=0.0):
    robot = AmrAdapter(cfg)
    scen = {"actor": actor_xy is not None}
    if actor_xy is not None:
        scen["waypoints"] = [[0.0, actor_xy[0], actor_xy[1]]]
    h = build_scene(robot, cfg, scen)
    robot.reset(h.model, h.data, np.asarray(robot_xy), yaw)
    return robot, h


def _scan(cfg, robot, h, seed=0):
    sensor = RangeSensor(cfg, np.random.default_rng(seed))
    pos, R = robot.sensor_pose(h.model, h.data)
    return sensor.scan(h.model, h.data, h.data.time, pos, R, h.geomgroup_mask())


def test_empty_corridor_sees_only_walls():
    cfg = _noiseless_cfg()
    robot, h = _scene(cfg)
    scan = _scan(cfg, robot, h)
    hw = cfg["world"]["corridor_half_width_m"]
    # side rays (+-90 deg) hit the side walls at exactly the half width
    assert scan.valid_hit[0] and scan.valid_hit[-1]
    assert scan.ranges_m[0] == pytest.approx(hw, abs=1e-6)
    assert scan.ranges_m[-1] == pytest.approx(hw, abs=1e-6)
    # every valid hit lies on a wall line
    pts = scan.sensor_pos_world[:2] + scan.ranges_m[:, None] * np.column_stack(
        [np.cos(scan.angles_rad), np.sin(scan.angles_rad)])
    pts = pts[scan.valid_hit]
    on_side = np.isclose(np.abs(pts[:, 1]), hw, atol=1e-6)
    x_max = cfg["world"]["corridor_x_min_m"] + cfg["world"]["corridor_length_m"]
    on_end = np.isclose(pts[:, 0], x_max, atol=1e-6)
    assert np.all(on_side | on_end)
    # the forward ray (front wall at 7.4 m) is beyond max range -> no return
    mid = len(scan.angles_rad) // 2
    assert not scan.valid_hit[mid]
    assert scan.ranges_m[mid] == pytest.approx(cfg["sensor"]["max_range_m"])


def test_actor_range_is_distance_minus_radius():
    cfg = _noiseless_cfg()
    d = 2.0
    robot, h = _scene(cfg, actor_xy=(d, 0.0))
    scan = _scan(cfg, robot, h)
    mid = len(scan.angles_rad) // 2
    assert scan.angles_rad[mid] == pytest.approx(0.0)
    expected = d - cfg["sensor"]["mount_forward_m"] - cfg["world"]["actor_radius_m"]
    assert scan.valid_hit[mid]
    assert scan.ranges_m[mid] == pytest.approx(expected, abs=1e-6)


def test_actor_range_with_noise_within_bounds():
    cfg = load_config()
    cfg = copy.deepcopy(cfg)
    cfg["sensor"]["dropout_probability"] = 0.0
    robot, h = _scene(cfg, actor_xy=(2.0, 0.0))
    expected = 2.0 - cfg["sensor"]["mount_forward_m"] - cfg["world"]["actor_radius_m"]
    sensor = RangeSensor(cfg, np.random.default_rng(3))
    pos, R = robot.sensor_pose(h.model, h.data)
    mid = cfg["sensor"]["num_rays"] // 2
    vals = [sensor.scan(h.model, h.data, 0.0, pos, R, h.geomgroup_mask()).ranges_m[mid] for _ in range(200)]
    assert abs(np.mean(vals) - expected) < 0.01
    assert np.std(vals) == pytest.approx(cfg["sensor"]["noise_std_m"], rel=0.25)


def test_no_self_hit_on_robot():
    cfg = _noiseless_cfg()
    robot, h = _scene(cfg)
    scan = _scan(cfg, robot, h)
    # sensor is inside the robot hull; no range may be shorter than the distance to a wall
    assert np.all(scan.ranges_m[scan.valid_hit] >= cfg["world"]["corridor_half_width_m"] - 1e-6)
    # robot geoms are not in the perceivable group
    rb = robot.robot_body_ids(h.model)
    for g in range(h.model.ngeom):
        if h.model.geom_bodyid[g] in rb:
            assert h.model.geom_group[g] != h.perceivable_group


def test_rotated_robot_scan_frame():
    cfg = _noiseless_cfg()
    robot, h = _scene(cfg, actor_xy=(0.0, 1.5), yaw=np.pi / 2)
    scan = _scan(cfg, robot, h)
    mid = len(scan.angles_rad) // 2
    expected = 1.5 - cfg["sensor"]["mount_forward_m"] - cfg["world"]["actor_radius_m"]
    assert scan.valid_hit[mid]
    assert scan.ranges_m[mid] == pytest.approx(expected, abs=1e-6)


def test_no_hit_is_invalid_and_max_range():
    cfg = _noiseless_cfg()
    cfg["sensor"]["max_range_m"] = 1.5  # walls at 2 m are out of range
    robot, h = _scene(cfg)
    scan = _scan(cfg, robot, h)
    assert not np.any(scan.valid_hit)
    assert np.allclose(scan.ranges_m, 1.5)
    assert np.all(np.isfinite(scan.ranges_m))


def test_dropout_all():
    cfg = _noiseless_cfg()
    cfg["sensor"]["dropout_probability"] = 1.0
    robot, h = _scene(cfg)
    scan = _scan(cfg, robot, h)
    assert not np.any(scan.valid_hit)


def test_cadence_on_sim_time():
    cfg = _noiseless_cfg()
    robot, h = _scene(cfg)
    sensor = RangeSensor(cfg, np.random.default_rng(0))
    assert sensor.period_s == pytest.approx(1.0 / cfg["sensor"]["frequency_hz"])
    pos, R = robot.sensor_pose(h.model, h.data)
    times = []
    dt = h.model.opt.timestep
    t = 0.0
    for k in range(int(round(1.0 / dt))):  # 1 s of sim time with float accumulation
        if sensor.due(t):
            sensor.scan(h.model, h.data, t, pos, R, h.geomgroup_mask())
            times.append(t)
        t += dt
    assert len(times) == cfg["sensor"]["frequency_hz"]
    assert np.allclose(np.diff(times), sensor.period_s, atol=dt + 1e-9)
    assert not sensor.due(times[-1] + 0.5 * sensor.period_s)


def test_scan_does_not_move_physics():
    cfg = _noiseless_cfg()
    robot, h = _scene(cfg, actor_xy=(2.0, 0.0))
    qpos = h.data.qpos.copy()
    _scan(cfg, robot, h)
    assert np.array_equal(qpos, h.data.qpos)
    assert isinstance(h.model, mujoco.MjModel)
