"""Scripted actor motion (ground truth). ONLY the runner (to drive the mocap body) and the
evaluator may call this. Perception / tracking / planning never see it.

Trajectory: piecewise-linear through waypoints [t_s, x, y]; the actor holds the first waypoint
before its time and the last waypoint after its time.

Per-seed jitter (np.random.default_rng(seed)), from scenario_cfg["seed_jitter"]:
    dt    ~ U(-start_time_s, +start_time_s)   added to all waypoint times
    scale ~ U(1 - speed_scale, 1 + speed_scale) speed multiplier (segment durations / scale)
Then every segment whose speed exceeds actor_max_speed_mps is stretched to exactly that speed
(later waypoints shift in time accordingly).
"""
from __future__ import annotations

import numpy as np


class ScenarioController:
    """Ground-truth actor trajectory for one scenario + seed.

    Args:
        scenario_cfg: one entry of configs/scenarios.yaml.
        seed: episode seed (jitter is reproducible per seed).
        actor_max_speed_mps: speed clamp per segment, m/s.
    """

    def __init__(self, scenario_cfg: dict, seed: int, actor_max_speed_mps: float = 2.0):
        self.cfg = scenario_cfg
        self.seed = int(seed)
        self.has_actor = bool(scenario_cfg.get("actor", False))
        self.interactive = bool(scenario_cfg.get("interactive", False))
        self.actor_max_speed_mps = float(actor_max_speed_mps)
        self._initialized = False
        self.time_offset_s = 0.0
        self.speed_scale = 1.0

        wps = np.asarray(scenario_cfg.get("waypoints") or [], dtype=float).reshape(-1, 3)
        if self.has_actor and len(wps) == 0:
            raise ValueError("scenario has actor=true but no waypoints")
        self.raw_waypoints = wps.copy()
        if len(wps) == 0:
            self.waypoints = wps
            return
        if np.any(np.diff(wps[:, 0]) < 0):
            raise ValueError("waypoint times must be non-decreasing")

        rng = np.random.default_rng(self.seed)
        jit = scenario_cfg.get("seed_jitter") or {}
        jt = float(jit.get("start_time_s", 0.0))
        js = float(jit.get("speed_scale", 0.0))
        # draw both always (fixed draw order -> reproducible)
        u_t, u_s = rng.uniform(-1.0, 1.0, size=2)
        if not self.interactive:
            self.time_offset_s = jt * u_t
            self.speed_scale = 1.0 + js * u_s

        t = wps[:, 0]
        new_t = np.empty_like(t)
        new_t[0] = t[0] + self.time_offset_s
        for i in range(1, len(wps)):
            dur = (t[i] - t[i - 1]) / self.speed_scale
            dist = float(np.linalg.norm(wps[i, 1:] - wps[i - 1, 1:]))
            if dist > 0.0 and self.actor_max_speed_mps > 0.0:
                dur = max(dur, dist / self.actor_max_speed_mps)
            new_t[i] = new_t[i - 1] + dur
        self.waypoints = np.column_stack([new_t, wps[:, 1:]])

    # ------------------------------------------------------------------ GT queries (eval only)
    def _scripted(self, time_s: float) -> bool:
        return self.has_actor and not self.interactive and len(self.waypoints) > 0

    def actor_xy(self, time_s: float) -> np.ndarray | None:
        """Scripted GT actor position (2,) m at sim time `time_s`; None if no actor or the
        scenario is interactive (then use gt_actor_xy_from_sim). EVAL ONLY."""
        if not self._scripted(time_s):
            return None
        wp = self.waypoints
        t = float(time_s)
        if t <= wp[0, 0]:
            return wp[0, 1:].copy()
        if t >= wp[-1, 0]:
            return wp[-1, 1:].copy()
        i = int(np.searchsorted(wp[:, 0], t, side="right")) - 1
        t0, t1 = wp[i, 0], wp[i + 1, 0]
        a = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        return (1.0 - a) * wp[i, 1:] + a * wp[i + 1, 1:]

    def actor_velocity(self, time_s: float) -> np.ndarray | None:
        """Scripted GT actor velocity (2,) m/s (zero while holding). EVAL ONLY."""
        if not self._scripted(time_s):
            return None
        wp = self.waypoints
        t = float(time_s)
        if t < wp[0, 0] or t >= wp[-1, 0] or len(wp) < 2:
            return np.zeros(2)
        i = int(np.searchsorted(wp[:, 0], t, side="right")) - 1
        dt = wp[i + 1, 0] - wp[i, 0]
        if dt <= 0.0:
            return np.zeros(2)
        return (wp[i + 1, 1:] - wp[i, 1:]) / dt

    # ------------------------------------------------------------------ sim
    def apply(self, model, data, handles) -> None:
        """Write the actor mocap position for the current data.time (call before each mj_step).
        Interactive scenario: sets the initial position once, then never overwrites it so
        the viewer can drag the actor."""
        mid = handles.actor_mocap_id
        if not self.has_actor or mid < 0:
            return
        z = 0.5 * float(handles.actor_height_m)
        if self.interactive:
            if not self._initialized:
                data.mocap_pos[mid] = [self.waypoints[0, 1], self.waypoints[0, 2], z]
                self._initialized = True
            return
        xy = self.actor_xy(float(data.time))
        data.mocap_pos[mid] = [xy[0], xy[1], z]
        self._initialized = True

    def gt_actor_xy_from_sim(self, data, handles) -> np.ndarray | None:
        """GT actor position (2,) m read from data.mocap_pos (works for interactive). EVAL ONLY."""
        mid = handles.actor_mocap_id
        if not self.has_actor or mid < 0:
            return None
        return np.array(data.mocap_pos[mid, :2], dtype=float)
