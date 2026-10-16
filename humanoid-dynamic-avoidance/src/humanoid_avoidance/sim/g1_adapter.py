"""Unitree G1 (12-DoF legs) adapter driven by the upstream pretrained walking policy.

Upstream: https://github.com/unitreerobotics/unitree_rl_gym (SHA pinned in
configs/upstream_version.txt). This module reproduces
``deploy/deploy_mujoco/deploy_mujoco.py`` + ``deploy/deploy_mujoco/configs/g1.yaml`` exactly,
except that it runs headless and the velocity command comes from the navigation stack:

* physics dt = ``simulation_dt`` (0.002 s); policy every ``control_decimation`` (10) physics
  steps -> 50 Hz (0.02 s), same as training (sim.dt 0.005 x decimation 4).
* every physics step: ``tau = kp * (target_q - q) + kd * (0 - dq)`` written to ``data.ctrl``.
* every policy step: 47-dim observation (see ``build_observation``), TorchScript policy
  (an LSTM; its hidden state lives inside the scripted module and is reset in ``reset``),
  ``target_q = action * action_scale + default_angles``.

The robot is moved ONLY through these joint torques. ``qpos`` is written only in ``reset``.
All values are read from the upstream YAML at construction time (no copies of gains here).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from humanoid_avoidance.config import THIRD_PARTY_DIR
from humanoid_avoidance.types import NavCommand, RobotState

UPSTREAM_DIR = THIRD_PARTY_DIR / "unitree_rl_gym"
UPSTREAM_DEPLOY_CFG = UPSTREAM_DIR / "deploy" / "deploy_mujoco" / "configs" / "g1.yaml"

# Gait clock period used by upstream (g1_env.py `_post_physics_step_callback` and
# deploy_mujoco.py: `period = 0.8`). Hard-coded upstream, not in the YAML.
UPSTREAM_GAIT_PERIOD_S = 0.8

# Training command ranges: legged_gym/envs/base/legged_robot_config.py `commands.ranges`
# (G1RoughCfg does not override them). heading_command=True during training, so the yaw-rate
# command was 0.5*heading_error clipped to [-1, 1]. Training also zeroed |v_xy| < 0.2
# (legged_robot.py `_resample_commands`): small non-zero linear commands were never trained.
UPSTREAM_CMD_RANGE = {"vx": (-1.0, 1.0), "vy": (-1.0, 1.0), "yaw_rate": (-1.0, 1.0)}
UPSTREAM_SMALL_CMD_ZEROED_BELOW_MPS = 0.2

# Policy joint order == actuator order == qpos[7:19] order in g1_12dof.xml.
G1_JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
)
BASE_BODY = "pelvis"
FREE_JOINT = "floating_base_joint"
# Range-sensor mount relative to the mount body origin (pelvis frame), metres. The 12-DoF
# model has no torso_link body (torso geoms are welded to the pelvis). Pelvis origin stands at
# ~0.78 m, so +0.30 m puts the sensor at ~1.08 m (chest height). Overridable via
# cfg["g1"]["sensor_offset_m"].
DEFAULT_SENSOR_OFFSET_M = (0.10, 0.0, 0.30)


def load_upstream_deploy_config(path: Path = UPSTREAM_DEPLOY_CFG) -> dict:
    """Read upstream g1.yaml and resolve {LEGGED_GYM_ROOT_DIR} to absolute paths."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Upstream G1 deploy config not found at {path}. Run scripts/setup.ps1 (or setup.sh) "
            "to clone unitree_rl_gym into third_party/.")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = str(UPSTREAM_DIR.resolve()).replace("\\", "/")
    for key in ("policy_path", "xml_path"):
        cfg[key] = str(Path(cfg[key].replace("{LEGGED_GYM_ROOT_DIR}", root)).resolve())
    return cfg


def gravity_orientation(quat_wxyz: np.ndarray) -> np.ndarray:
    """Projected gravity in the base frame, (3,). Verbatim port of upstream
    `get_gravity_orientation` (deploy_mujoco.py). quat in MuJoCo order (w, x, y, z)."""
    qw, qx, qy, qz = (float(v) for v in quat_wxyz)
    return np.array([
        2.0 * (-qz * qx + qw * qy),
        -2.0 * (qz * qy + qw * qx),
        1.0 - 2.0 * (qw * qw + qz * qz),
    ])


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


class G1Adapter:
    """RobotAdapter implementation for the Unitree G1 12-DoF + pretrained policy.

    Attributes:
        name: "g1".
        radius_m: planning radius, cfg navigation.nominal_robot_radius_m [m].
        max_vx_mps: min(cfg g1.max_vx_mps, training range max) [m/s].
        physics_dt: upstream simulation_dt [s] (0.002).
        policy_dt: physics_dt * control_decimation [s] (0.02).
    """

    name = "g1"

    def __init__(self, cfg: dict, device: str = "cpu"):
        import torch  # local import: torch only needed for the G1 path

        self._torch = torch
        self.cfg = cfg
        self.up = load_upstream_deploy_config()
        up = self.up
        self.policy_path = Path(up["policy_path"])
        self.xml_path = Path(up["xml_path"])
        if not self.policy_path.is_file():
            raise FileNotFoundError(
                f"G1 pretrained policy missing: {self.policy_path}. It ships with unitree_rl_gym "
                "(deploy/pre_train/g1/motion.pt); re-run scripts/setup.ps1. No fake weights are used.")
        if not self.xml_path.is_file():
            raise FileNotFoundError(f"G1 scene xml missing: {self.xml_path}")

        self.physics_dt = float(up["simulation_dt"])
        self.control_decimation = int(up["control_decimation"])
        self.policy_dt = self.physics_dt * self.control_decimation
        self.kps = np.asarray(up["kps"], dtype=np.float64)
        self.kds = np.asarray(up["kds"], dtype=np.float64)
        self.default_angles = np.asarray(up["default_angles"], dtype=np.float64)
        self.ang_vel_scale = float(up["ang_vel_scale"])
        self.dof_pos_scale = float(up["dof_pos_scale"])
        self.dof_vel_scale = float(up["dof_vel_scale"])
        self.action_scale = float(up["action_scale"])
        self.cmd_scale = np.asarray(up["cmd_scale"], dtype=np.float64)
        self.num_actions = int(up["num_actions"])
        self.num_obs = int(up["num_obs"])
        self.gait_period_s = UPSTREAM_GAIT_PERIOD_S
        assert self.num_actions == len(G1_JOINT_NAMES) == 12, "unexpected upstream action dim"
        assert self.num_obs == 9 + 3 * self.num_actions + 2 == 47, "unexpected upstream obs dim"

        g1 = cfg["g1"]
        self.radius_m = float(cfg["navigation"]["nominal_robot_radius_m"])
        self.max_vx_mps = float(min(g1["max_vx_mps"], UPSTREAM_CMD_RANGE["vx"][1]))
        self.fall_height_m = float(g1["fall_height_m"])
        self.fall_tilt_rad = float(g1["fall_tilt_rad"])
        self.perceivable_group = int(cfg["world"]["perceivable_geom_group"])
        self.sensor_offset_m = np.asarray(g1.get("sensor_offset_m", DEFAULT_SENSOR_OFFSET_M),
                                          dtype=np.float64)

        self.device = device
        self.policy = torch.jit.load(str(self.policy_path), map_location=device)
        self.policy.eval()

        self._cmd = np.zeros(3)                  # body-frame [vx, vy, yaw_rate] after clipping
        self._bound = False
        self._reset_internal_state()

    # ------------------------------------------------------------------ model construction
    def base_spec(self):
        """Upstream scene (G1 12-DoF + floor + light) as an MjSpec.

        Changes vs upstream: option.timestep forced to upstream simulation_dt (deploy_mujoco.py
        sets m.opt.timestep the same way after loading); any robot geom in the perceivable group is
        moved to group 1 (none are, upstream uses groups 0 and 1). Actuators, joints, gains,
        collision filtering untouched.
        """
        import mujoco

        spec = mujoco.MjSpec.from_file(str(self.xml_path))
        # absolute meshdir so the spec still compiles after it was edited/attached elsewhere
        spec.meshdir = str((self.xml_path.parent / (spec.meshdir or "")).resolve())
        spec.option.timestep = self.physics_dt
        base = spec.body(BASE_BODY)
        if base is None:
            raise RuntimeError(f"body '{BASE_BODY}' not found in {self.xml_path}")
        stack = [base]
        while stack:
            b = stack.pop()
            for g in b.geoms:
                if g.group == self.perceivable_group:
                    g.group = 1
            stack.extend(b.bodies)
        return spec

    def bind(self, model, data) -> None:
        """Resolve joint/actuator/body ids after compilation and check the timestep."""
        import mujoco

        if abs(model.opt.timestep - self.physics_dt) > 1e-12:
            raise AssertionError(
                f"model.opt.timestep={model.opt.timestep} != upstream simulation_dt={self.physics_dt}")
        self.base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BASE_BODY)
        fj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, FREE_JOINT)
        if self.base_body_id < 0 or fj < 0:
            raise RuntimeError("G1 pelvis / floating base joint not found in compiled model")
        self.free_qpos_adr = int(model.jnt_qposadr[fj])
        self.free_qvel_adr = int(model.jnt_dofadr[fj])
        self.qpos_adr = np.zeros(12, dtype=int)
        self.qvel_adr = np.zeros(12, dtype=int)
        self.act_ids = np.zeros(12, dtype=int)
        for i, jn in enumerate(G1_JOINT_NAMES):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid < 0:
                raise RuntimeError(f"joint {jn} missing")
            self.qpos_adr[i] = model.jnt_qposadr[jid]
            self.qvel_adr[i] = model.jnt_dofadr[jid]
            acts = [a for a in range(model.nu)
                    if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
                    and model.actuator_trnid[a, 0] == jid]
            if len(acts) != 1:
                raise RuntimeError(f"expected exactly one actuator on {jn}, got {acts}")
            self.act_ids[i] = acts[0]
        # initial base height from the upstream MJCF (pelvis pos z = 0.793)
        self.init_base_height_m = float(model.qpos0[self.free_qpos_adr + 2])
        self._bound = True

    # ------------------------------------------------------------------ episode control
    def _reset_internal_state(self) -> None:
        self._action = np.zeros(self.num_actions, dtype=np.float32)
        self._target_q = self.default_angles.copy()
        self._obs = np.zeros(self.num_obs, dtype=np.float32)
        self._t0 = 0.0
        self._last_policy_step = 0
        self.policy_calls = 0
        self.last_policy_wall_s = float("nan")
        if hasattr(self, "policy"):
            self.policy.reset_memory()   # LSTM hidden/cell state -> 0 (upstream fresh start)

    def reset(self, model, data, start_xy, start_yaw: float) -> None:
        """Place the robot. The ONLY method that writes qpos.

        start_xy: (2,) world [m]; start_yaw: [rad]. Base height = upstream initial height,
        joints = default_angles, all velocities 0, policy memory/action cleared, mj_forward.
        """
        import mujoco

        assert self._bound, "call bind() first"
        a = self.free_qpos_adr
        data.qpos[a + 0] = float(start_xy[0])
        data.qpos[a + 1] = float(start_xy[1])
        data.qpos[a + 2] = self.init_base_height_m
        data.qpos[a + 3:a + 7] = [math.cos(start_yaw / 2), 0.0, 0.0, math.sin(start_yaw / 2)]
        data.qpos[self.qpos_adr] = self.default_angles
        v = self.free_qvel_adr
        data.qvel[v:v + 6] = 0.0
        data.qvel[self.qvel_adr] = 0.0
        data.ctrl[self.act_ids] = 0.0
        mujoco.mj_forward(model, data)
        self._reset_internal_state()
        self._t0 = float(data.time)

    def set_command(self, cmd: NavCommand) -> None:
        """Store body-frame [vx, vy, yaw_rate] clipped to the training range and g1.max_vx_mps."""
        vx = float(np.clip(cmd.vx_mps, max(UPSTREAM_CMD_RANGE["vx"][0], -self.max_vx_mps),
                           self.max_vx_mps))
        vy = float(np.clip(cmd.vy_mps, max(UPSTREAM_CMD_RANGE["vy"][0], -self.max_vx_mps),
                           min(UPSTREAM_CMD_RANGE["vy"][1], self.max_vx_mps)))
        wz = float(np.clip(cmd.yaw_rate_rps, *UPSTREAM_CMD_RANGE["yaw_rate"]))
        self._cmd = np.array([vx, vy, wz])

    @property
    def command(self) -> np.ndarray:
        """Currently applied (clipped) body-frame command [vx, vy, yaw_rate], (3,)."""
        return self._cmd.copy()

    # ------------------------------------------------------------------ control loop
    def build_observation(self, model, data) -> np.ndarray:
        """47-dim policy observation, float32, exactly as deploy_mujoco.py:

        [0:3]   base angular velocity (body frame, free-joint qvel[3:6]) * ang_vel_scale
        [3:6]   projected gravity from base quaternion (free-joint qpos[3:7])
        [6:9]   command [vx, vy, wz] * cmd_scale
        [9:21]  (q - default_angles) * dof_pos_scale
        [21:33] dq * dof_vel_scale
        [33:45] previous action (raw policy output)
        [45:47] sin, cos of 2*pi*phase; phase = (t % 0.8) / 0.8, t = sim time since reset
        """
        a, v = self.free_qpos_adr, self.free_qvel_adr
        quat = data.qpos[a + 3:a + 7]
        omega = data.qvel[v + 3:v + 6]
        qj = data.qpos[self.qpos_adr]
        dqj = data.qvel[self.qvel_adr]
        n = self.num_actions
        # upstream: count = counter * simulation_dt; counter = physics steps since start
        t = data.time - self._t0
        phase = (t % self.gait_period_s) / self.gait_period_s
        obs = self._obs
        obs[0:3] = omega * self.ang_vel_scale
        obs[3:6] = gravity_orientation(quat)
        obs[6:9] = self._cmd * self.cmd_scale
        obs[9:9 + n] = (qj - self.default_angles) * self.dof_pos_scale
        obs[9 + n:9 + 2 * n] = dqj * self.dof_vel_scale
        obs[9 + 2 * n:9 + 3 * n] = self._action
        obs[9 + 3 * n:9 + 3 * n + 2] = (math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase))
        return obs.copy()

    def policy_step(self, obs: np.ndarray) -> np.ndarray:
        """Run the TorchScript policy on a (47,) obs; returns the (12,) raw action."""
        import time

        torch = self._torch
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.policy(torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0))
        self.last_policy_wall_s = time.perf_counter() - t0
        self.policy_calls += 1
        return out.detach().cpu().numpy().reshape(-1).astype(np.float32)

    def pre_step(self, model, data) -> None:
        """Call before EVERY mj_step.

        Physics step index k = round((data.time - t_reset) / dt). Upstream increments its counter
        after mj_step and updates targets when counter % decimation == 0, so the targets for
        step k (k > 0, k % decimation == 0) come from the state after k steps -> computed here
        before the PD torque of step k. Then tau = kp*(target - q) - kd*dq -> data.ctrl.
        """
        k = int(round((data.time - self._t0) / self.physics_dt))
        if k > 0 and k % self.control_decimation == 0 and k != self._last_policy_step:
            obs = self.build_observation(model, data)
            self._action = self.policy_step(obs)
            self._target_q = self._action.astype(np.float64) * self.action_scale + self.default_angles
            self._last_policy_step = k
        q = data.qpos[self.qpos_adr]
        dq = data.qvel[self.qvel_adr]
        data.ctrl[self.act_ids] = self.kps * (self._target_q - q) + self.kds * (0.0 - dq)

    # ------------------------------------------------------------------ state queries
    def robot_state(self, model, data) -> RobotState:
        """Localization oracle (pelvis pose) + MEASURED base velocity.

        MuJoCo free joint: qvel[0:3] = linear velocity of the body origin in the WORLD frame,
        qvel[3:6] = angular velocity in the BODY frame. yaw_rate = world-z of R @ omega_body.
        Note: raw pelvis velocity contains the gait sway (lateral oscillation each step).
        """
        a, v = self.free_qpos_adr, self.free_qvel_adr
        quat = np.array(data.qpos[a + 3:a + 7])
        R = _quat_to_rotmat(quat)
        yaw = math.atan2(R[1, 0], R[0, 0])
        omega_world = R @ np.array(data.qvel[v + 3:v + 6])
        return RobotState(
            time_s=float(data.time),
            xy_world_m=np.array(data.qpos[a:a + 2], dtype=float),
            yaw_rad=yaw,
            v_world_mps=np.array(data.qvel[v:v + 2], dtype=float),
            yaw_rate_rps=float(omega_world[2]),
            base_height_m=float(data.qpos[a + 2]),
        )

    def sensor_pose(self, model, data) -> tuple[np.ndarray, np.ndarray]:
        """Range-sensor mount: pelvis frame + sensor_offset_m. Returns (pos (3,), R (3,3)) in world.
        Body-fixed (follows pelvis roll/pitch sway, typically a few degrees while walking)."""
        R = np.array(data.xmat[self.base_body_id]).reshape(3, 3)
        pos = np.array(data.xpos[self.base_body_id]) + R @ self.sensor_offset_m
        return pos, R

    def robot_body_ids(self, model) -> set[int]:
        """All bodies in the G1 subtree (pelvis and descendants)."""
        root = self.base_body_id
        return {b for b in range(model.nbody) if model.body_rootid[b] == root}

    def tilt_rad(self, data) -> float:
        """Angle between base z-axis and world z [rad]."""
        R = np.array(data.xmat[self.base_body_id]).reshape(3, 3)
        return float(math.acos(max(-1.0, min(1.0, R[2, 2]))))

    def fallen(self, model, data) -> bool:
        """Base height < g1.fall_height_m or tilt > g1.fall_tilt_rad."""
        h = float(data.qpos[self.free_qpos_adr + 2])
        return h < self.fall_height_m or self.tilt_rad(data) > self.fall_tilt_rad
