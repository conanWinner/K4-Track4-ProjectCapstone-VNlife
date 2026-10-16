"""Simulated planar 2D range sensor (LiDAR-like) using MuJoCo ray casting.

The sensor never reads the actor pose: it only casts rays against geoms of the perceivable
group (walls, occluder, actor body) via mujoco.mj_multiRay.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from humanoid_avoidance.types import Scan

_TIME_EPS = 1e-6


class RangeSensor:
    """Horizontal fan of `num_rays` rays over `fov_deg`, centred on sensor +x, CCW+.

    Args:
        cfg: full config dict (uses the `sensor` section: frequency_hz, fov_deg, num_rays,
            min_range_m, max_range_m, noise_std_m, dropout_probability) or the sensor section itself.
        rng: numpy Generator for noise / dropout (seeded by the caller for reproducibility).
    """

    def __init__(self, cfg: dict, rng: np.random.Generator):
        s = cfg["sensor"] if "sensor" in cfg else cfg
        self.frequency_hz = float(s["frequency_hz"])
        self.period_s = 1.0 / self.frequency_hz
        self.fov_rad = math.radians(float(s["fov_deg"]))
        self.num_rays = int(s["num_rays"])
        self.min_range_m = float(s["min_range_m"])
        self.max_range_m = float(s["max_range_m"])
        self.noise_std_m = float(s["noise_std_m"])
        self.dropout_probability = float(s["dropout_probability"])
        self.rng = rng
        if self.num_rays == 1:
            self.angles_rad = np.zeros(1)
        else:
            self.angles_rad = np.linspace(-0.5 * self.fov_rad, 0.5 * self.fov_rad, self.num_rays)
        self._dirs_sensor = np.column_stack(
            [np.cos(self.angles_rad), np.sin(self.angles_rad), np.zeros(self.num_rays)])
        self._geomid = np.zeros(self.num_rays, dtype=np.int32)
        self._dist = np.zeros(self.num_rays, dtype=np.float64)
        self._next_time_s = 0.0
        self.last_scan_time_s: float | None = None

    def reset(self) -> None:
        """Restart the cadence (next scan due at t = 0)."""
        self._next_time_s = 0.0
        self.last_scan_time_s = None

    def due(self, time_s: float) -> bool:
        """True if a scan is due at sim time `time_s` (s). Scans occur on the fixed grid
        k * period_s; the grid advances when `scan` is called."""
        return float(time_s) + _TIME_EPS >= self._next_time_s

    def scan(self, model, data, time_s: float, sensor_pos, sensor_R, geomgroup_mask) -> Scan:
        """Cast the ray fan and return a Scan.

        Args:
            model, data: MuJoCo model/data (kinematics must be current, e.g. after mj_step/mj_forward).
            time_s: sim time of the scan, s.
            sensor_pos: (3,) sensor origin in world, m.
            sensor_R: (3,3) sensor->world rotation; rays lie in the sensor x-y plane.
            geomgroup_mask: uint8 (6,) — only enabled groups are hit (use the perceivable group only).

        Returns:
            Scan with ranges (N,) in m (max_range where no return) and valid_hit (N,) bool.
            Noise: additive gaussian noise_std_m on valid hits, clipped to [min, max] range.
            Dropout: each valid ray is lost with dropout_probability.
        """
        pos = np.asarray(sensor_pos, dtype=np.float64).reshape(3)
        R = np.asarray(sensor_R, dtype=np.float64).reshape(3, 3)
        dirs_world = self._dirs_sensor @ R.T  # (N,3) unit vectors
        mask = np.asarray(geomgroup_mask, dtype=np.uint8).reshape(6)
        self._geomid[:] = -1
        self._dist[:] = -1.0
        mujoco.mj_multiRay(model, data, pos, np.ascontiguousarray(dirs_world.reshape(-1)), mask,
                           1, -1, self._geomid, self._dist, None, self.num_rays, self.max_range_m)
        dist = self._dist.copy()
        valid = (self._geomid >= 0) & (dist >= self.min_range_m) & (dist <= self.max_range_m)
        if self.noise_std_m > 0.0:
            dist = dist + self.rng.normal(0.0, self.noise_std_m, size=self.num_rays)
        if self.dropout_probability > 0.0:
            valid &= self.rng.random(self.num_rays) >= self.dropout_probability
        ranges = np.where(valid, np.clip(dist, self.min_range_m, self.max_range_m), self.max_range_m)

        t = float(time_s)
        self.last_scan_time_s = t
        self._next_time_s = (math.floor(t / self.period_s + _TIME_EPS) + 1) * self.period_s
        return Scan(
            time_s=t,
            angles_rad=self.angles_rad.copy(),
            ranges_m=ranges,
            valid_hit=valid,
            sensor_pos_world=pos.copy(),
            sensor_rot_world=R.copy(),
        )
