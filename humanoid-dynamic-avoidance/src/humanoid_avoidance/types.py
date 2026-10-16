"""Shared data contracts. Units: m, s, rad, m/s, rad/s. World frame: XY ground plane, yaw CCW+.

Ground-truth actor state NEVER appears in these types; it lives only in
ScenarioController / evaluation code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Scan:
    time_s: float
    angles_rad: np.ndarray      # (N,), relative to sensor +x, CCW+
    ranges_m: np.ndarray        # (N,), finite; equals max_range where no hit
    valid_hit: np.ndarray       # (N,), bool; False = no return (max range / dropout)
    sensor_pos_world: np.ndarray = field(default_factory=lambda: np.zeros(3))   # (3,) at scan time
    sensor_rot_world: np.ndarray = field(default_factory=lambda: np.eye(3))     # (3,3) sensor->world rotation


@dataclass
class Detection:
    time_s: float
    xy_world_m: np.ndarray      # (2,) visible-surface centroid (NOT ground-truth center)
    radius_m: float             # extent / 2 of the cluster (+ inflation)
    covariance_xy: np.ndarray   # (2,2) measurement covariance R, m^2
    num_points: int = 0


@dataclass
class Track:
    track_id: int
    time_s: float
    state: np.ndarray           # (4,) [x, y, vx, vy] world frame
    covariance: np.ndarray      # (4,4)
    radius_m: float
    confirmed: bool
    missed_updates: int
    hits: int = 0
    last_nis: float = float("nan")


@dataclass
class RobotState:
    """Ego state. MVP: from simulator (localization oracle), documented as such."""
    time_s: float
    xy_world_m: np.ndarray      # (2,)
    yaw_rad: float
    v_world_mps: np.ndarray     # (2,) MEASURED base velocity (not the command)
    yaw_rate_rps: float
    base_height_m: float = float("nan")


@dataclass
class RiskAssessment:
    track_id: int               # -1 if no track
    tca_s: float                # time of closest approach, clipped to [0, horizon]
    dca_m: float                # distance (center-to-center) at closest approach
    required_clearance_m: float # r_robot + r_obs + margin
    margin_m: float
    approaching: bool
    risk: bool
    score: float                # 0..1, 1 = most dangerous (for HUD / logs)


@dataclass
class NavCommand:
    vx_mps: float               # body frame
    vy_mps: float
    yaw_rate_rps: float
    mode: str                   # GO / SLOW / YIELD / RESUME / STOP_SENSOR_FAILURE / GOAL_REACHED / MANUAL


NO_RISK = RiskAssessment(track_id=-1, tca_s=float("nan"), dca_m=float("inf"),
                         required_clearance_m=0.0, margin_m=0.0, approaching=False,
                         risk=False, score=0.0)
