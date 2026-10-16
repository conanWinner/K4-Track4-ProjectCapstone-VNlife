"""Tests for the Unitree G1 adapter (pretrained unitree_rl_gym policy in MuJoCo).

Skipped (with reason) when mujoco/torch, the upstream clone or the policy weights are missing.
Run scripts/setup.ps1 (Windows) or scripts/setup.sh first.
"""
from __future__ import annotations

import ast
import inspect
import math

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco", reason="mujoco not installed")
torch = pytest.importorskip("torch", reason="torch not installed")

from humanoid_avoidance.config import load_config  # noqa: E402
from humanoid_avoidance.sim import g1_adapter as g1mod  # noqa: E402
from humanoid_avoidance.types import NavCommand  # noqa: E402

try:
    _UP = g1mod.load_upstream_deploy_config()
    _MISSING = None if (g1mod.Path(_UP["policy_path"]).is_file()
                        and g1mod.Path(_UP["xml_path"]).is_file()) else "policy/xml file missing"
except FileNotFoundError as e:  # upstream not cloned
    _MISSING = str(e)

pytestmark = pytest.mark.skipif(_MISSING is not None,
                                reason=f"unitree_rl_gym / G1 weights unavailable: {_MISSING}")


@pytest.fixture(scope="module")
def sim():
    cfg = load_config()
    ad = g1mod.G1Adapter(cfg)
    model = ad.base_spec().compile()
    data = mujoco.MjData(model)
    ad.bind(model, data)
    ad.reset(model, data, [0.0, 0.0], 0.0)
    return cfg, ad, model, data


def _run(ad, model, data, seconds):
    fell = False
    for _ in range(int(round(seconds / model.opt.timestep))):
        ad.pre_step(model, data)
        mujoco.mj_step(model, data)
        if ad.fallen(model, data):
            fell = True
            break
    return fell


def test_rates_match_upstream(sim):
    cfg, ad, model, data = sim
    assert model.opt.timestep == pytest.approx(_UP["simulation_dt"])
    assert ad.physics_dt == pytest.approx(0.002)
    assert ad.control_decimation == 10
    assert ad.policy_dt == pytest.approx(0.02)


def test_obs_and_action_shapes(sim):
    cfg, ad, model, data = sim
    mujoco.mj_resetData(model, data)
    ad.reset(model, data, [0.0, 0.0], 0.0)
    obs = ad.build_observation(model, data)
    assert obs.shape == (47,) and obs.dtype == np.float32
    # standing upright at reset: projected gravity ~ (0, 0, -1), joint offsets 0, phase (0, 1)
    np.testing.assert_allclose(obs[3:6], [0.0, 0.0, -1.0], atol=1e-6)
    np.testing.assert_allclose(obs[9:21], 0.0, atol=1e-6)
    np.testing.assert_allclose(obs[45:47], [0.0, 1.0], atol=1e-6)
    act = ad.policy_step(obs)
    assert act.shape == (12,) and np.all(np.isfinite(act))


def test_command_clipping(sim):
    cfg, ad, model, data = sim
    ad.set_command(NavCommand(5.0, -5.0, 9.0, "GO"))
    vx, vy, wz = ad.command
    assert vx == pytest.approx(min(cfg["g1"]["max_vx_mps"], 1.0))
    assert vy == pytest.approx(-min(cfg["g1"]["max_vx_mps"], 1.0))
    assert wz == pytest.approx(1.0)
    ad.set_command(NavCommand(0.2, 0.0, -0.3, "GO"))
    np.testing.assert_allclose(ad.command, [0.2, 0.0, -0.3])
    ad.set_command(NavCommand(0.0, 0.0, 0.0, "GO"))


def test_scene_structure(sim):
    cfg, ad, model, data = sim
    bodies = ad.robot_body_ids(model)
    assert 0 not in bodies and ad.base_body_id in bodies and len(bodies) == 13
    robot_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] in bodies]
    assert all(model.geom_group[g] != cfg["world"]["perceivable_geom_group"] for g in robot_geoms)
    assert model.nmeshvert > 0  # meshes loaded through absolute meshdir


def test_qpos_written_only_in_reset():
    """Structural check: `<x>.qpos[...] = ...` assignments appear only inside G1Adapter.reset."""
    tree = ast.parse(inspect.getsource(g1mod))
    offenders = []
    for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
        for fn in (n for n in cls.body if isinstance(n, ast.FunctionDef)):
            for node in ast.walk(fn):
                targets = []
                if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Attribute) \
                            and t.value.attr == "qpos" and fn.name != "reset":
                        offenders.append(fn.name)
    assert offenders == []


def test_matches_upstream_deploy_loop(sim):
    """1 s of the verbatim deploy_mujoco.py loop vs the adapter from the same initial state."""
    cfg, ad, model, data = sim
    up = _UP
    m = mujoco.MjModel.from_xml_path(up["xml_path"])
    d = mujoco.MjData(m)
    m.opt.timestep = up["simulation_dt"]
    kps = np.array(up["kps"], dtype=np.float32)
    kds = np.array(up["kds"], dtype=np.float32)
    da = np.array(up["default_angles"], dtype=np.float32)
    cs = np.array(up["cmd_scale"], dtype=np.float32)
    cmd = np.array([0.3, 0.0, 0.0], dtype=np.float32)
    d.qpos[7:] = da
    mujoco.mj_forward(m, d)
    pol = torch.jit.load(up["policy_path"])
    action = np.zeros(12, dtype=np.float32)
    target = da.copy()
    obs = np.zeros(47, dtype=np.float32)
    counter = 0
    for _ in range(500):
        d.ctrl[:] = (target - d.qpos[7:]) * kps + (0 - d.qvel[6:]) * kds
        mujoco.mj_step(m, d)
        counter += 1
        if counter % up["control_decimation"] == 0:
            ph = counter * up["simulation_dt"] % 0.8 / 0.8
            obs[:3] = d.qvel[3:6] * up["ang_vel_scale"]
            obs[3:6] = g1mod.gravity_orientation(d.qpos[3:7])
            obs[6:9] = cmd * cs
            obs[9:21] = (d.qpos[7:] - da) * up["dof_pos_scale"]
            obs[21:33] = d.qvel[6:] * up["dof_vel_scale"]
            obs[33:45] = action
            obs[45:47] = [math.sin(2 * math.pi * ph), math.cos(2 * math.pi * ph)]
            with torch.no_grad():
                action = pol(torch.from_numpy(obs).unsqueeze(0)).numpy().squeeze()
            target = action * up["action_scale"] + da

    mujoco.mj_resetData(model, data)
    ad.reset(model, data, [0.0, 0.0], 0.0)
    ad.set_command(NavCommand(0.3, 0.0, 0.0, "MANUAL"))
    for _ in range(500):
        ad.pre_step(model, data)
        mujoco.mj_step(model, data)
    np.testing.assert_allclose(data.qpos, d.qpos, atol=1e-4)
    ad.set_command(NavCommand(0.0, 0.0, 0.0, "MANUAL"))


def test_walks_forward_without_falling(sim):
    cfg, ad, model, data = sim
    mujoco.mj_resetData(model, data)
    ad.reset(model, data, [0.0, 0.0], 0.0)
    ad.set_command(NavCommand(0.3, 0.0, 0.0, "MANUAL"))
    s0 = ad.robot_state(model, data)
    fell = _run(ad, model, data, 10.0)
    s1 = ad.robot_state(model, data)
    assert not fell, f"G1 fell at t={data.time:.2f}s"
    assert s1.xy_world_m[0] - s0.xy_world_m[0] > 1.0
    assert s1.base_height_m > cfg["g1"]["fall_height_m"]
    assert ad.policy_calls == pytest.approx(10.0 / ad.policy_dt, abs=1)
    pos, R = ad.sensor_pose(model, data)
    assert 0.9 < pos[2] < 1.25
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)


def test_stop_and_resume(sim):
    cfg, ad, model, data = sim
    mujoco.mj_resetData(model, data)
    ad.reset(model, data, [0.0, 0.0], 0.0)
    for vx, dur in ((0.3, 3.0), (0.0, 3.0), (0.3, 3.0)):
        ad.set_command(NavCommand(vx, 0.0, 0.0, "MANUAL"))
        x0 = ad.robot_state(model, data).xy_world_m[0]
        assert not _run(ad, model, data, dur)
        dx = ad.robot_state(model, data).xy_world_m[0] - x0
        if vx == 0.0:
            assert abs(dx) < 0.3, f"drifted {dx:.3f} m while commanded to stop"
        else:
            assert dx > 0.5
