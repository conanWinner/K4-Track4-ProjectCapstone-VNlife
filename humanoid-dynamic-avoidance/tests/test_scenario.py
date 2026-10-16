"""ScenarioController + scene + AMR adapter physics tests."""
import copy

import mujoco
import numpy as np
import pytest

from humanoid_avoidance.config import load_config, load_scenarios
from humanoid_avoidance.sim.amr_adapter import AmrAdapter
from humanoid_avoidance.sim.robot_base import RobotAdapter, make_robot
from humanoid_avoidance.sim.scenario_controller import ScenarioController
from humanoid_avoidance.sim.scene_builder import build_scene
from humanoid_avoidance.types import NavCommand

SIMPLE = {"actor": True, "waypoints": [[2.0, 0.0, 0.0], [4.0, 2.0, 0.0], [6.0, 2.0, 2.0]]}


# ---------------------------------------------------------------- scenario controller
def test_interpolation_and_hold():
    sc = ScenarioController(SIMPLE, seed=0)
    assert np.allclose(sc.actor_xy(0.0), [0.0, 0.0])        # hold before first
    assert np.allclose(sc.actor_xy(3.0), [1.0, 0.0])
    assert np.allclose(sc.actor_xy(5.0), [2.0, 1.0])
    assert np.allclose(sc.actor_xy(100.0), [2.0, 2.0])      # hold after last
    assert np.allclose(sc.actor_velocity(3.0), [1.0, 0.0])
    assert np.allclose(sc.actor_velocity(1.0), [0.0, 0.0])
    assert np.allclose(sc.actor_velocity(50.0), [0.0, 0.0])


def test_jitter_reproducible_by_seed():
    scen = load_scenarios()["crossing_slow"]
    a, b, c = ScenarioController(scen, 5), ScenarioController(scen, 5), ScenarioController(scen, 6)
    assert np.array_equal(a.waypoints, b.waypoints)
    assert not np.array_equal(a.waypoints, c.waypoints)
    j = scen["seed_jitter"]
    for seed in range(20):
        s = ScenarioController(scen, seed)
        assert abs(s.time_offset_s) <= j["start_time_s"]
        assert abs(s.speed_scale - 1.0) <= j["speed_scale"]
        # positions are unchanged, only timing
        assert np.allclose(s.waypoints[:, 1:], np.asarray(scen["waypoints"])[:, 1:])


def test_speed_clamp():
    scen = {"actor": True, "waypoints": [[0.0, 0.0, 0.0], [1.0, 5.0, 0.0]]}  # 5 m/s requested
    sc = ScenarioController(scen, 0, actor_max_speed_mps=2.0)
    assert np.linalg.norm(sc.actor_velocity(0.5)) == pytest.approx(2.0)
    assert sc.waypoints[-1, 0] == pytest.approx(2.5)
    for t in np.linspace(0, 3, 61):
        v = sc.actor_velocity(t)
        assert np.linalg.norm(v) <= 2.0 + 1e-9


def test_no_actor_scenario():
    cfg = load_config()
    sc = ScenarioController(load_scenarios()["empty"], 0)
    assert sc.actor_xy(1.0) is None and sc.actor_velocity(1.0) is None
    robot = AmrAdapter(cfg)
    h = build_scene(robot, cfg, load_scenarios()["empty"])
    assert h.actor_mocap_id == -1 and h.actor_geom_ids == set()
    sc.apply(h.model, h.data, h)  # no-op
    assert sc.gt_actor_xy_from_sim(h.data, h) is None


def test_apply_writes_mocap():
    cfg = load_config()
    robot = AmrAdapter(cfg)
    scen = load_scenarios()["crossing_fast"]
    h = build_scene(robot, cfg, scen)
    sc = ScenarioController(scen, 3)
    h.data.time = 9.0
    sc.apply(h.model, h.data, h)
    assert np.allclose(h.data.mocap_pos[h.actor_mocap_id, :2], sc.actor_xy(9.0))
    assert h.data.mocap_pos[h.actor_mocap_id, 2] == pytest.approx(0.5 * cfg["world"]["actor_height_m"])
    assert np.allclose(sc.gt_actor_xy_from_sim(h.data, h), sc.actor_xy(9.0))


def test_interactive_does_not_overwrite():
    cfg = load_config()
    robot = AmrAdapter(cfg)
    scen = load_scenarios()["interactive"]
    h = build_scene(robot, cfg, scen)
    sc = ScenarioController(scen, 0)
    sc.apply(h.model, h.data, h)
    mid = h.actor_mocap_id
    assert np.allclose(h.data.mocap_pos[mid, :2], scen["waypoints"][0][1:])
    h.data.mocap_pos[mid, :2] = [1.0, 0.5]       # "viewer drag"
    for _ in range(10):
        sc.apply(h.model, h.data, h)
        mujoco.mj_step(h.model, h.data)
    assert np.allclose(h.data.mocap_pos[mid, :2], [1.0, 0.5])
    assert np.allclose(sc.gt_actor_xy_from_sim(h.data, h), [1.0, 0.5])


# ---------------------------------------------------------------- scene + AMR physics
def _amr_scene(scenario=None):
    cfg = copy.deepcopy(load_config())
    robot = make_robot("amr", cfg)
    h = build_scene(robot, cfg, scenario or {"actor": False})
    robot.reset(h.model, h.data, np.zeros(2), 0.0)
    return cfg, robot, h


def _run(robot, h, cmd, seconds, ctl=None):
    robot.set_command(cmd)
    n = int(round(seconds / h.model.opt.timestep))
    for _ in range(n):
        if ctl is not None:
            ctl.apply(h.model, h.data, h)
        robot.pre_step(h.model, h.data)
        mujoco.mj_step(h.model, h.data)


def test_amr_protocol_and_timestep():
    cfg, robot, h = _amr_scene()
    assert isinstance(robot, RobotAdapter)
    assert h.model.opt.timestep == pytest.approx(0.002)
    assert robot.radius_m == cfg["navigation"]["nominal_robot_radius_m"]
    assert not robot.fallen(h.model, h.data)
    assert len(h.wall_segments.shape) == 3 and h.wall_segments.shape[1:] == (2, 2)
    assert np.all(h.model.geom_group[list(h.wall_geom_ids)] == h.perceivable_group)


def test_amr_forward_motion_via_physics():
    cfg, robot, h = _amr_scene()
    _run(robot, h, NavCommand(0.3, 0.0, 0.0, "GO"), 3.0)
    st = robot.robot_state(h.model, h.data)
    assert st.xy_world_m[0] == pytest.approx(0.9, abs=0.05)
    assert abs(st.xy_world_m[1]) < 1e-3
    assert st.v_world_mps[0] == pytest.approx(0.3, abs=0.01)
    assert st.time_s == pytest.approx(3.0, abs=1e-6)


def test_amr_yaw_command_turns_and_heading_motion():
    cfg, robot, h = _amr_scene()
    _run(robot, h, NavCommand(0.0, 0.0, 0.4, "GO"), 2.0)
    st = robot.robot_state(h.model, h.data)
    assert st.yaw_rad == pytest.approx(0.8, abs=0.03)
    assert st.yaw_rate_rps == pytest.approx(0.4, abs=0.01)
    _run(robot, h, NavCommand(0.0, 0.0, 0.0, "GO"), 0.5)
    yaw = robot.robot_state(h.model, h.data).yaw_rad
    p0 = robot.robot_state(h.model, h.data).xy_world_m.copy()
    _run(robot, h, NavCommand(0.3, 0.0, 0.0, "GO"), 2.0)
    d = robot.robot_state(h.model, h.data).xy_world_m - p0
    assert np.arctan2(d[1], d[0]) == pytest.approx(yaw, abs=0.02)
    pos, R = robot.sensor_pose(h.model, h.data)
    assert pos[2] == pytest.approx(cfg["sensor"]["mount_height_m"])
    assert np.allclose(R[:2, 0], [np.cos(yaw), np.sin(yaw)])


def test_actor_contact_registers():
    scen = {"actor": True, "waypoints": [[0.0, 0.3, 0.0]]}  # actor overlapping the robot
    cfg, robot, h = _amr_scene(scen)
    ctl = ScenarioController(scen, 0)
    _run(robot, h, NavCommand(0.0, 0.0, 0.0, "GO"), 0.02, ctl)
    rb = robot.robot_body_ids(h.model)
    hit = False
    for i in range(h.data.ncon):
        c = h.data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        b1, b2 = h.model.geom_bodyid[g1], h.model.geom_bodyid[g2]
        if (g1 in h.actor_geom_ids and b2 in rb) or (g2 in h.actor_geom_ids and b1 in rb):
            hit = True
    assert hit


def test_robot_does_not_touch_floor_and_free_corridor_no_contacts():
    cfg, robot, h = _amr_scene()
    _run(robot, h, NavCommand(0.3, 0.0, 0.0, "GO"), 0.5)
    assert h.data.ncon == 0
