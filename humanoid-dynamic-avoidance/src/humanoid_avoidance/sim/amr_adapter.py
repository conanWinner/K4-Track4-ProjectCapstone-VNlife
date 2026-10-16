"""Cylinder AMR proxy robot (fallback for the Unitree G1).

The robot is a single rigid cylinder with three degrees of freedom:
slide-x and slide-y (world-aligned, because they precede the hinge) and a yaw hinge.
Each DoF is driven by a MuJoCo velocity actuator (force = kv * (ctrl - qvel)), so motion
comes from physics; qpos is written only in `reset`.

Optional cfg keys (section `amr:`, defaults used if absent):
    height_m (1.2), mass_kg (30), kv_lin (600 N/(m/s)), kv_yaw (60 N m/(rad/s)),
    max_vx_mps (1.0), max_speed_ctrl_mps (1.5), max_yaw_ctrl_rps (2.0), floor_clearance_m (0.02)
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from humanoid_avoidance.types import NavCommand, RobotState

_AMR_DEFAULTS = {
    "height_m": 1.2,
    "mass_kg": 30.0,
    "kv_lin": 600.0,
    "kv_yaw": 60.0,
    "max_vx_mps": 1.0,
    "max_speed_ctrl_mps": 1.5,
    "max_yaw_ctrl_rps": 2.0,
    "floor_clearance_m": 0.02,
}

# Robot geoms collide through bit 2 only: they touch walls/actor (contype|conaffinity = 3)
# but not the floor (1), so the base glides on its actuated slide joints without friction.
ROBOT_CONTYPE = 2
ROBOT_CONAFFINITY = 2


class AmrAdapter:
    """Cylinder AMR implementing `RobotAdapter`.

    Args:
        cfg: full config dict. Uses navigation.nominal_robot_radius_m, sensor.mount_height_m
            (height of the sensor above the floor, m), sensor.mount_forward_m, world.perceivable_geom_group
            and the optional `amr:` section.
    """

    name = "amr"
    BODY_NAME = "amr_base"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        p = dict(_AMR_DEFAULTS)
        p.update(cfg.get("amr", {}) or {})
        self.params = p
        self.radius_m = float(cfg["navigation"]["nominal_robot_radius_m"])
        self.height_m = float(p["height_m"])
        self.max_vx_mps = float(p["max_vx_mps"])
        self.mount_height_m = float(cfg["sensor"]["mount_height_m"])
        self.mount_forward_m = float(cfg["sensor"]["mount_forward_m"])
        perceivable = int(cfg["world"]["perceivable_geom_group"])
        # visual-only group that is never the perceivable group
        self._body_group = 0 if perceivable != 0 else 1
        self._marker_group = 1 if perceivable != 1 else 2
        self._cmd = np.zeros(3)  # body-frame [vx, vy, wz]
        self._body_id = -1
        self._jnt_qpos = np.zeros(3, dtype=int)
        self._jnt_qvel = np.zeros(3, dtype=int)
        self._act_ids = np.zeros(3, dtype=int)

    # ------------------------------------------------------------------ spec
    def base_spec(self) -> mujoco.MjSpec:
        """MjSpec with floor (checker, group 0), light and the AMR. Timestep 0.002 s."""
        p = self.params
        spec = mujoco.MjSpec()
        spec.modelname = "amr_corridor"
        spec.option.timestep = 0.002
        spec.visual.global_.offwidth = 1280
        spec.visual.global_.offheight = 720

        tex = spec.add_texture()
        tex.name = "floor_checker"
        tex.type = mujoco.mjtTexture.mjTEXTURE_2D
        tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
        tex.width = 512
        tex.height = 512
        tex.rgb1 = [0.32, 0.34, 0.37]
        tex.rgb2 = [0.24, 0.26, 0.28]
        tex.mark = mujoco.mjtMark.mjMARK_EDGE
        tex.markrgb = [0.55, 0.55, 0.5]
        mat = spec.add_material()
        mat.name = "floor_mat"
        mat.texrepeat = [1.0, 1.0]
        mat.texuniform = True
        mat.reflectance = 0.05
        textures = list(mat.textures)
        textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "floor_checker"
        mat.textures = textures

        wb = spec.worldbody
        floor = wb.add_geom()
        floor.name = "floor"
        floor.type = mujoco.mjtGeom.mjGEOM_PLANE
        floor.size = [15.0, 8.0, 0.05]
        floor.pos = [3.0, 0.0, 0.0]
        floor.material = "floor_mat"
        floor.group = 0
        floor.contype = 1
        floor.conaffinity = 1

        light = wb.add_light()
        light.name = "sun"
        light.pos = [3.0, 0.0, 6.0]
        light.dir = [0.0, 0.0, -1.0]
        light.diffuse = [0.8, 0.8, 0.8]
        light.castshadow = True

        body = wb.add_body()
        body.name = self.BODY_NAME
        body.pos = [0.0, 0.0, 0.0]
        for jname, jtype, axis in (
            ("amr_slide_x", mujoco.mjtJoint.mjJNT_SLIDE, [1, 0, 0]),
            ("amr_slide_y", mujoco.mjtJoint.mjJNT_SLIDE, [0, 1, 0]),
            ("amr_yaw", mujoco.mjtJoint.mjJNT_HINGE, [0, 0, 1]),
        ):
            j = body.add_joint()
            j.name = jname
            j.type = jtype
            j.axis = axis

        half_h = 0.5 * self.height_m
        clearance = float(p["floor_clearance_m"])
        cyl = body.add_geom()
        cyl.name = "amr_hull"
        cyl.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        cyl.size = [self.radius_m, half_h, 0.0]
        cyl.pos = [0.0, 0.0, clearance + half_h]
        cyl.rgba = [0.2, 0.45, 0.85, 1.0]
        cyl.group = self._body_group
        cyl.contype = ROBOT_CONTYPE
        cyl.conaffinity = ROBOT_CONAFFINITY
        cyl.mass = float(p["mass_kg"])

        marker = body.add_geom()
        marker.name = "amr_heading"
        marker.type = mujoco.mjtGeom.mjGEOM_BOX
        marker.size = [0.5 * self.radius_m, 0.04, 0.02]
        marker.pos = [0.5 * self.radius_m, 0.0, clearance + self.height_m + 0.02]
        marker.rgba = [1.0, 0.95, 0.2, 1.0]
        marker.group = self._marker_group
        marker.contype = 0
        marker.conaffinity = 0
        marker.mass = 0.1

        sensor_vis = body.add_geom()
        sensor_vis.name = "amr_sensor_vis"
        sensor_vis.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        sensor_vis.size = [0.05, 0.03, 0.0]
        sensor_vis.pos = [self.mount_forward_m, 0.0, self.mount_height_m]
        sensor_vis.rgba = [0.1, 0.1, 0.1, 1.0]
        sensor_vis.group = self._marker_group
        sensor_vis.contype = 0
        sensor_vis.conaffinity = 0
        sensor_vis.mass = 0.1

        vmax = float(p["max_speed_ctrl_mps"])
        wmax = float(p["max_yaw_ctrl_rps"])
        for aname, jname, kv, lim in (
            ("amr_vx", "amr_slide_x", float(p["kv_lin"]), vmax),
            ("amr_vy", "amr_slide_y", float(p["kv_lin"]), vmax),
            ("amr_wz", "amr_yaw", float(p["kv_yaw"]), wmax),
        ):
            a = spec.add_actuator()
            a.name = aname
            a.target = jname
            a.trntype = mujoco.mjtTrn.mjTRN_JOINT
            a.set_to_velocity(kv=kv)
            a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
            a.ctrlrange = [-lim, lim]
        return spec

    # ------------------------------------------------------------------ runtime
    def bind(self, model, data) -> None:
        self._body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.BODY_NAME)
        if self._body_id < 0:
            raise RuntimeError("AMR body not found in model")
        for i, jn in enumerate(("amr_slide_x", "amr_slide_y", "amr_yaw")):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            self._jnt_qpos[i] = model.jnt_qposadr[jid]
            self._jnt_qvel[i] = model.jnt_dofadr[jid]
        for i, an in enumerate(("amr_vx", "amr_vy", "amr_wz")):
            self._act_ids[i] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, an)
        self._cmd[:] = 0.0

    def reset(self, model, data, start_xy, start_yaw) -> None:
        """Teleport to start pose (reset only), zero velocities and controls, mj_forward."""
        data.qpos[self._jnt_qpos[0]] = float(start_xy[0])
        data.qpos[self._jnt_qpos[1]] = float(start_xy[1])
        data.qpos[self._jnt_qpos[2]] = float(start_yaw)
        data.qvel[self._jnt_qvel] = 0.0
        data.ctrl[self._act_ids] = 0.0
        self._cmd[:] = 0.0
        mujoco.mj_forward(model, data)

    def set_command(self, cmd: NavCommand) -> None:
        self._cmd[:] = (float(cmd.vx_mps), float(cmd.vy_mps), float(cmd.yaw_rate_rps))

    def _yaw(self, data) -> float:
        return float(data.qpos[self._jnt_qpos[2]])

    def pre_step(self, model, data) -> None:
        """Rotate body-frame (vx, vy) to world with the current yaw and write actuator ctrl."""
        yaw = self._yaw(data)
        c, s = math.cos(yaw), math.sin(yaw)
        vx, vy, wz = self._cmd
        data.ctrl[self._act_ids[0]] = c * vx - s * vy
        data.ctrl[self._act_ids[1]] = s * vx + c * vy
        data.ctrl[self._act_ids[2]] = wz

    def robot_state(self, model, data) -> RobotState:
        """Pose from joint positions (localization oracle), velocity from qvel (measured)."""
        qx, qy, qyaw = (float(data.qpos[i]) for i in self._jnt_qpos)
        vx, vy, wz = (float(data.qvel[i]) for i in self._jnt_qvel)
        yaw = math.atan2(math.sin(qyaw), math.cos(qyaw))
        return RobotState(
            time_s=float(data.time),
            xy_world_m=np.array([qx, qy]),
            yaw_rad=yaw,
            v_world_mps=np.array([vx, vy]),
            yaw_rate_rps=wz,
            base_height_m=float(self.params["floor_clearance_m"]) + 0.5 * self.height_m,
        )

    def sensor_pose(self, model, data) -> tuple[np.ndarray, np.ndarray]:
        """Horizontal sensor: pos = base + R_yaw [mount_forward, 0, 0], z = mount_height_m."""
        qx, qy = float(data.qpos[self._jnt_qpos[0]]), float(data.qpos[self._jnt_qpos[1]])
        yaw = self._yaw(data)
        c, s = math.cos(yaw), math.sin(yaw)
        pos = np.array([qx + c * self.mount_forward_m, qy + s * self.mount_forward_m, self.mount_height_m])
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return pos, R

    def robot_body_ids(self, model) -> set[int]:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.BODY_NAME)
        return {int(bid)} if bid >= 0 else set()

    def fallen(self, model, data) -> bool:
        return False
