"""Multi-object tracker: CV Kalman filters + Mahalanobis gating + Hungarian assignment.

Lifecycle (all thresholds from ``cfg["tracking"]``):

- **tentative**: spawned from an unassociated detection.
- **confirmed**: once it has >= ``min_confirmations`` hits within its last ``confirmation_window``
  tracker updates (the spawning detection counts as a hit). A tentative track that can no longer
  reach that count (window full, too few hits) is deleted.
- **deleted**: confirmed tracks are removed after ``max_misses`` consecutive missed updates.

Time handling: every ``step`` predicts each track to ``time_s`` using the *actual* elapsed sim
time since its last state (variable dt, no fixed-rate assumption). Time must be non-decreasing.
Repeating a timestamp never counts a second miss and never re-predicts (dt = 0 is a no-op).

Used cfg keys: ``sigma_accel_mps2``, ``init_velocity_sigma_mps``, ``mahalanobis_gate_sq``,
``min_confirmations``, ``confirmation_window``, ``max_misses``; optional ``radius_ema_alpha``
(weight of the newest detection radius in the track radius EMA, default 0.3).
"""
from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from humanoid_avoidance.tracking.kalman_cv import init_state, kf_predict, kf_update, mahalanobis_sq
from humanoid_avoidance.types import Detection, Track

#: Default EMA weight for the track radius if ``radius_ema_alpha`` is absent from cfg.
DEFAULT_RADIUS_EMA_ALPHA = 0.3


@dataclass
class _TrackEntry:
    """Internal bookkeeping wrapped around a public ``Track``."""
    track: Track
    history: deque                    # bools, True = hit, newest last; maxlen = confirmation_window
    last_event_time_s: float          # sim time of the last hit/miss bookkeeping
    last_event_hit: bool = field(default=True)


class MultiTracker:
    """Track multiple point-like obstacles in the world XY plane from clustered detections."""

    def __init__(self, cfg_tracking: dict):
        """Args: cfg_tracking: the ``tracking`` section of the config (see module docstring)."""
        self.sigma_a = float(cfg_tracking["sigma_accel_mps2"])
        self.init_velocity_sigma = float(cfg_tracking["init_velocity_sigma_mps"])
        self.gate_sq = float(cfg_tracking["mahalanobis_gate_sq"])
        self.min_confirmations = int(cfg_tracking["min_confirmations"])
        self.confirmation_window = int(cfg_tracking["confirmation_window"])
        self.max_misses = int(cfg_tracking["max_misses"])
        self.radius_alpha = float(cfg_tracking.get("radius_ema_alpha", DEFAULT_RADIUS_EMA_ALPHA))
        if self.confirmation_window < self.min_confirmations:
            raise ValueError("confirmation_window must be >= min_confirmations")
        # Cost used for gated-out pairs in the assignment matrix: large but finite so the
        # solver always succeeds; such pairs are rejected after solving.
        self._big_cost = 1e6 * max(self.gate_sq, 1.0)
        self.reset()

    # ------------------------------------------------------------------ public API
    def reset(self) -> None:
        """Drop all tracks, restart ids at 1 and forget the last timestamp."""
        self._entries: list[_TrackEntry] = []
        self._next_id = 1
        self._last_time_s: float | None = None

    def step(self, time_s: float, detections: list[Detection]) -> list[Track]:
        """Advance the tracker to sim time ``time_s`` [s] and fuse ``detections``.

        Predict all tracks to time_s (sim time, monotonic; ValueError if time goes back), gate with
        squared Mahalanobis distance (cfg ``mahalanobis_gate_sq``), assign with
        ``linear_sum_assignment`` on the gated cost matrix (pairs outside the gate are never
        assigned), update matched tracks, spawn tentative tracks for unmatched detections and run the
        lifecycle. Call with ``[]`` for a predict-only (missed) update.

        All detections are treated as measured at ``time_s`` (the scan timestamp).
        Returns copies of ALL live tracks (tentative and confirmed), sorted by id.
        """
        time_s = float(time_s)
        if self._last_time_s is not None and time_s < self._last_time_s:
            raise ValueError(f"MultiTracker.step: time went backwards ({time_s} < {self._last_time_s})")
        same_time = self._last_time_s is not None and time_s == self._last_time_s
        if same_time and not detections:
            return self._copies(self._entries)
        self._last_time_s = time_s

        # 1) predict every track to time_s with its own real elapsed dt
        for e in self._entries:
            t = e.track
            dt = time_s - t.time_s
            t.state, t.covariance = kf_predict(t.state, t.covariance, max(dt, 0.0), self.sigma_a)
            t.time_s = max(t.time_s, time_s)

        # 2) gated assignment
        matches, unmatched_dets = self._associate(detections)
        matched_idx = set()
        for ti, di in matches:
            matched_idx.add(ti)
            self._apply_hit(self._entries[ti], detections[di], time_s)
        for ti, e in enumerate(self._entries):
            if ti not in matched_idx:
                self._apply_miss(e, time_s)

        # 3) lifecycle: delete, then spawn
        self._entries = [e for e in self._entries if not self._should_delete(e)]
        for di in unmatched_dets:
            self._spawn(detections[di], time_s)
        return self._copies(self._entries)

    def tracks(self) -> list[Track]:
        """Copies of all live tracks."""
        return self._copies(self._entries)

    def confirmed_tracks(self) -> list[Track]:
        """Copies of confirmed live tracks only."""
        return self._copies([e for e in self._entries if e.track.confirmed])

    # ------------------------------------------------------------------ internals
    def _associate(self, detections: list[Detection]) -> tuple[list[tuple[int, int]], list[int]]:
        n_t, n_d = len(self._entries), len(detections)
        if n_t == 0 or n_d == 0:
            return [], list(range(n_d))
        d2 = np.empty((n_t, n_d))
        for i, e in enumerate(self._entries):
            for j, det in enumerate(detections):
                d2[i, j] = mahalanobis_sq(e.track.state, e.track.covariance,
                                          det.xy_world_m, det.covariance_xy)
        in_gate = d2 <= self.gate_sq
        cost = np.where(in_gate, d2, self._big_cost)
        rows, cols = linear_sum_assignment(cost)
        matches = [(int(r), int(c)) for r, c in zip(rows, cols) if in_gate[r, c]]
        assigned = {c for _, c in matches}
        return matches, [j for j in range(n_d) if j not in assigned]

    def _apply_hit(self, e: _TrackEntry, det: Detection, time_s: float) -> None:
        t = e.track
        t.state, t.covariance, t.last_nis = kf_update(t.state, t.covariance,
                                                      det.xy_world_m, det.covariance_xy)
        t.radius_m = (1.0 - self.radius_alpha) * t.radius_m + self.radius_alpha * float(det.radius_m)
        if e.last_event_time_s == time_s:
            # second measurement at an already-booked timestamp: turn a booked miss into a hit
            if not e.last_event_hit:
                e.history[-1] = True
                e.last_event_hit = True
                t.missed_updates = 0
                t.hits += 1
        else:
            e.history.append(True)
            e.last_event_time_s = time_s
            e.last_event_hit = True
            t.missed_updates = 0
            t.hits += 1
        if not t.confirmed and sum(e.history) >= self.min_confirmations:
            t.confirmed = True

    def _apply_miss(self, e: _TrackEntry, time_s: float) -> None:
        if e.last_event_time_s == time_s:
            return  # already booked at this timestamp
        e.history.append(False)
        e.last_event_time_s = time_s
        e.last_event_hit = False
        e.track.missed_updates += 1

    def _should_delete(self, e: _TrackEntry) -> bool:
        t = e.track
        if t.missed_updates >= self.max_misses:
            return True
        if not t.confirmed and len(e.history) == self.confirmation_window \
                and sum(e.history) < self.min_confirmations:
            return True
        return False

    def _spawn(self, det: Detection, time_s: float) -> None:
        x, P = init_state(det.xy_world_m, det.covariance_xy, self.init_velocity_sigma)
        track = Track(track_id=self._next_id, time_s=time_s, state=x, covariance=P,
                      radius_m=float(det.radius_m), confirmed=self.min_confirmations <= 1,
                      missed_updates=0, hits=1)
        self._next_id += 1
        hist = deque([True], maxlen=self.confirmation_window)
        self._entries.append(_TrackEntry(track=track, history=hist, last_event_time_s=time_s))

    @staticmethod
    def _copies(entries: list[_TrackEntry]) -> list[Track]:
        return [copy.deepcopy(e.track) for e in sorted(entries, key=lambda e: e.track.track_id)]
