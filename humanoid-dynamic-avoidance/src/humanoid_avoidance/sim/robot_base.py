"""Robot adapter protocol shared by the AMR proxy and the Unitree G1 adapter.

The rest of the stack (runner, perception, navigation) talks to a robot only through
this protocol, so the cylinder AMR and the G1 humanoid are interchangeable.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from humanoid_avoidance.types import NavCommand, RobotState


@runtime_checkable
class RobotAdapter(Protocol):
    """Contract for a simulated robot.

    Attributes:
        name: short identifier ("amr" / "g1").
        radius_m: footprint radius used by planning, m.
        max_vx_mps: maximum forward speed the robot can track, m/s.
    """

    name: str
    radius_m: float
    max_vx_mps: float

    def base_spec(self):  # -> mujoco.MjSpec
        """Return an MjSpec with robot + floor + light only. Robot geoms must NOT be in
        the perceivable geom group (cfg world.perceivable_geom_group)."""
        ...

    def bind(self, model, data) -> None:
        """Resolve joint / actuator / body ids after the full scene has been compiled."""
        ...

    def reset(self, model, data, start_xy, start_yaw) -> None:
        """Place the robot at start_xy (2,) m, start_yaw rad. Only place where qpos is written."""
        ...

    def set_command(self, cmd: NavCommand) -> None:
        """Store the latest body-frame velocity command (applied in pre_step)."""
        ...

    def pre_step(self, model, data) -> None:
        """Called before EVERY mj_step; writes data.ctrl (G1: policy at its own decimation)."""
        ...

    def robot_state(self, model, data) -> RobotState:
        """Localization oracle pose + MEASURED base velocity (world frame)."""
        ...

    def sensor_pose(self, model, data) -> tuple[np.ndarray, np.ndarray]:
        """(pos (3,) m, R (3,3) sensor->world) of the range sensor mount."""
        ...

    def robot_body_ids(self, model) -> set[int]:
        """Ids of all bodies belonging to the robot (for contact checks)."""
        ...

    def fallen(self, model, data) -> bool:
        """True if the robot has fallen (always False for the AMR)."""
        ...


def make_robot(name: str, cfg: dict) -> RobotAdapter:
    """Factory: "amr" -> AmrAdapter, "g1" -> G1Adapter (imported lazily so the AMR path
    works without the Unitree assets).

    Args:
        name: "amr" or "g1".
        cfg: full config dict (configs/default.yaml).
    """
    key = name.lower().strip()
    if key == "amr":
        from humanoid_avoidance.sim.amr_adapter import AmrAdapter

        return AmrAdapter(cfg)
    if key == "g1":
        from humanoid_avoidance.sim.g1_adapter import G1Adapter

        return G1Adapter(cfg)
    raise ValueError(f"unknown robot '{name}', expected 'amr' or 'g1'")
