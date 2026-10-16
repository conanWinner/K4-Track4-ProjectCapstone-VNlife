"""Factory-corridor scene: robot base spec + walls + optional occluder + mocap actor + goal marker.

Collision bits: walls / occluder / actor use contype = conaffinity = 3 (bits 1 and 2), so they
collide with the AMR hull (bit 2) and with G1 geoms (MuJoCo default bit 1). Perceivable geoms
(walls, occluder, actor body) are in group cfg world.perceivable_geom_group; the range sensor
casts rays against that group only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

SCENE_CONTYPE = 3
SCENE_CONAFFINITY = 3
ACTOR_BODY_NAME = "actor"
ACTOR_PARK_XY = (0.0, -50.0)  # parking spot if a scenario has no waypoints (far outside corridor)


@dataclass
class SceneHandles:
    """Handles to the compiled scene.

    Attributes:
        model, data: compiled MjModel / MjData.
        actor_mocap_id: index into data.mocap_pos for the actor, -1 if no actor.
        actor_geom_ids: geom ids of the actor (collision + perceivable).
        wall_geom_ids: geom ids of the known corridor walls (not the occluder).
        perceivable_group: geom group seen by the range sensor.
        wall_segments: (M, 2, 2) known-map wall inner-face lines [[x0, y0], [x1, y1]], m.
        occluder_geom_ids: geom ids of the static occluder (unknown to the map), may be empty.
        actor_body_id: body id of the actor, -1 if none.
        actor_height_m: actor cylinder height, m (mocap z = height / 2).
    """

    model: mujoco.MjModel
    data: mujoco.MjData
    actor_mocap_id: int
    actor_geom_ids: set[int]
    wall_geom_ids: set[int]
    perceivable_group: int
    wall_segments: np.ndarray
    occluder_geom_ids: set[int] = field(default_factory=set)
    actor_body_id: int = -1
    actor_height_m: float = 0.0

    def geomgroup_mask(self) -> np.ndarray:
        """uint8 (6,) mask selecting only the perceivable group (for mj_ray / RangeSensor.scan)."""
        return perceivable_geomgroup_mask(self.perceivable_group)


def perceivable_geomgroup_mask(group: int) -> np.ndarray:
    """uint8 (6,) geomgroup mask with only `group` enabled."""
    mask = np.zeros(6, dtype=np.uint8)
    mask[int(group)] = 1
    return mask


def _add_box(parent, name, pos, half_size, rgba, group, contype, conaffinity):
    g = parent.add_geom()
    g.name = name
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.pos = [float(v) for v in pos]
    g.size = [float(v) for v in half_size]
    g.rgba = rgba
    g.group = group
    g.contype = contype
    g.conaffinity = conaffinity
    return g


def corridor_wall_segments(cfg_world: dict, end_walls: bool = True) -> np.ndarray:
    """Known map: inner-face lines of the corridor walls, (M, 2, 2) in m."""
    x0 = float(cfg_world["corridor_x_min_m"])
    x1 = x0 + float(cfg_world["corridor_length_m"])
    hw = float(cfg_world["corridor_half_width_m"])
    segs = [[[x0, hw], [x1, hw]], [[x0, -hw], [x1, -hw]]]
    if end_walls:
        segs += [[[x0, -hw], [x0, hw]], [[x1, -hw], [x1, hw]]]
    return np.asarray(segs, dtype=float)


def build_scene(robot, cfg: dict, scenario_cfg: dict) -> SceneHandles:
    """Build and compile the corridor scene around `robot.base_spec()`.

    Args:
        robot: RobotAdapter (AMR or G1). Its base_spec() supplies robot, floor and light.
        cfg: full config dict (uses world.*, simulation.goal_xy, navigation.goal_tolerance_m).
            Optional world.end_walls (bool, default True).
        scenario_cfg: one entry of configs/scenarios.yaml (keys actor, waypoints, occluder).

    Returns:
        SceneHandles; robot.bind(model, data) has been called. The robot is NOT reset here.
    """
    w = cfg["world"]
    group = int(w["perceivable_geom_group"])
    x0 = float(w["corridor_x_min_m"])
    length = float(w["corridor_length_m"])
    x1 = x0 + length
    hw = float(w["corridor_half_width_m"])
    t = float(w["wall_thickness_m"])
    h = float(w["wall_height_m"])
    end_walls = bool(w.get("end_walls", True))
    wall_rgba = [0.78, 0.78, 0.74, 1.0]

    spec = robot.base_spec()
    wb = spec.worldbody

    wall_names = []
    xc = 0.5 * (x0 + x1)
    for name, y in (("wall_left", hw + 0.5 * t), ("wall_right", -hw - 0.5 * t)):
        _add_box(wb, name, [xc, y, 0.5 * h], [0.5 * length + t, 0.5 * t, 0.5 * h],
                 wall_rgba, group, SCENE_CONTYPE, SCENE_CONAFFINITY)
        wall_names.append(name)
    if end_walls:
        for name, x in (("wall_back", x0 - 0.5 * t), ("wall_front", x1 + 0.5 * t)):
            _add_box(wb, name, [x, 0.0, 0.5 * h], [0.5 * t, hw + t, 0.5 * h],
                     wall_rgba, group, SCENE_CONTYPE, SCENE_CONAFFINITY)
            wall_names.append(name)

    occluder_names = []
    occ = scenario_cfg.get("occluder")
    if occ:
        size = [float(v) for v in occ["size"]]  # MuJoCo box half-sizes, m
        oxy = [float(v) for v in occ["xy"]]
        _add_box(wb, "occluder", [oxy[0], oxy[1], size[2]], size, [0.45, 0.4, 0.35, 1.0],
                 group, SCENE_CONTYPE, SCENE_CONAFFINITY)
        occluder_names.append("occluder")

    # goal marker: visual only (no collision, not perceivable)
    goal_xy = cfg.get("simulation", {}).get("goal_xy")
    if goal_xy is not None:
        tol = float(cfg.get("navigation", {}).get("goal_tolerance_m", 0.3))
        vis_group = 2 if group != 2 else 1
        g = wb.add_geom()
        g.name = "goal_marker"
        g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        g.size = [tol, 0.004, 0.0]
        g.pos = [float(goal_xy[0]), float(goal_xy[1]), 0.005]
        g.rgba = [0.1, 0.85, 0.2, 0.45]
        g.group = vis_group
        g.contype = 0
        g.conaffinity = 0

    has_actor = bool(scenario_cfg.get("actor", False))
    ar = float(w["actor_radius_m"])
    ah = float(w["actor_height_m"])
    if has_actor:
        wps = scenario_cfg.get("waypoints") or []
        start = wps[0][1:3] if wps else ACTOR_PARK_XY
        body = wb.add_body()
        body.name = ACTOR_BODY_NAME
        body.mocap = True
        body.pos = [float(start[0]), float(start[1]), 0.5 * ah]
        g = body.add_geom()
        g.name = "actor_body"
        g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        g.size = [ar, 0.5 * ah, 0.0]
        g.rgba = [1.0, 0.5, 0.1, 1.0]
        g.group = group
        g.contype = SCENE_CONTYPE
        g.conaffinity = SCENE_CONAFFINITY
        head = body.add_geom()  # cosmetic only: no collision, not perceivable
        head.name = "actor_head"
        head.type = mujoco.mjtGeom.mjGEOM_SPHERE
        head.size = [0.6 * ar, 0.0, 0.0]
        head.pos = [0.0, 0.0, 0.5 * ah + 0.5 * ar]
        head.rgba = [1.0, 0.75, 0.55, 1.0]
        head.group = 2 if group != 2 else 1
        head.contype = 0
        head.conaffinity = 0

    model = spec.compile()
    data = mujoco.MjData(model)

    def gid(n):
        return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n))

    actor_mocap_id, actor_body_id, actor_geom_ids = -1, -1, set()
    if has_actor:
        actor_body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ACTOR_BODY_NAME))
        actor_mocap_id = int(model.body_mocapid[actor_body_id])
        actor_geom_ids = {gid("actor_body")}

    mujoco.mj_forward(model, data)
    robot.bind(model, data)
    return SceneHandles(
        model=model,
        data=data,
        actor_mocap_id=actor_mocap_id,
        actor_geom_ids=actor_geom_ids,
        wall_geom_ids={gid(n) for n in wall_names},
        perceivable_group=group,
        wall_segments=corridor_wall_segments(w, end_walls),
        occluder_geom_ids={gid(n) for n in occluder_names},
        actor_body_id=actor_body_id,
        actor_height_m=ah,
    )
