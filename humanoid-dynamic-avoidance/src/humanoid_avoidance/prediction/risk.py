"""Constant-velocity motion prediction and TCA-based collision risk (2D discs).

Model and assumptions
---------------------
- Robot and obstacle are discs in the XY plane; gait / foot sweep is not modelled.
- Obstacle motion: constant velocity from its track; position covariance at look-ahead tau is
  ``[F(tau) P F(tau)^T + Q(tau)]_xy`` (default ``covariance_model="single_interval"``, as specified:
  one acceleration sample held constant over the whole look-ahead). Optional
  ``covariance_model="stepwise"`` instead chains ``ceil(tau/step_s)`` predictions of at most
  ``step_s`` each, i.e. the same piecewise-constant-per-interval assumption the tracker uses; it
  grows much more slowly with tau (Q(tau) position variance ~ sigma_a^2 tau^4 / 4).
- Robot motion: constant *measured* base velocity ``RobotState.v_world_mps`` (never the command).
- TCA = **time of closest approach** (not "time to collision"):
  with q = p_obs - p_robot and w = v_obs - v_robot,
  ``tau* = clip(-(q.w)/(w.w+eps), 0, horizon)``, ``dca = |q + w tau*|``.
- Uncertainty-aware margin (heuristic, capped, NOT a probabilistic guarantee):
  ``margin = m0 + min(k * sqrt(lambda_max(P_xy(tau*))), max_uncertainty_margin_m)``.
  The cap exists because the predicted covariance grows quickly with look-ahead (Q(tau) alone
  gives a ~2 m position sigma at tau = 2.5 s with sigma_a = 0.7). Without the cap almost every
  approaching or passing obstacle would be flagged, including ones parked well beside the path.

Used ``cfg_prediction`` keys: ``horizon_s``, ``step_s``, ``nominal_margin_m``, ``uncertainty_k``,
``approach_speed_eps_mps``; optional ``max_uncertainty_margin_m`` (cap on the uncertainty term,
default inf = uncapped) and ``covariance_model`` ("single_interval" | "stepwise").
"""
from __future__ import annotations

import dataclasses

import numpy as np

from humanoid_avoidance.tracking.kalman_cv import F_cv, Q_cv, kf_predict
from humanoid_avoidance.types import NO_RISK, RiskAssessment, RobotState, Track


def time_of_closest_approach(q: np.ndarray, w: np.ndarray, horizon_s: float,
                             eps: float = 1e-6) -> tuple[float, float]:
    """Time of closest approach (TCA) of two constant-velocity points.

    Args:
        q: (2,) relative position p_obs - p_robot [m].
        w: (2,) relative velocity v_obs - v_robot [m/s].
        horizon_s: look-ahead horizon [s]; tau is clipped to [0, horizon_s].
        eps: regulariser [m^2/s^2]; if w.w <= eps the relative motion is treated as zero (tau = 0).
    Returns:
        (tau_s, dca_m): TCA [s] and centre-to-centre distance at TCA [m].
    """
    q = np.asarray(q, dtype=float).reshape(2)
    w = np.asarray(w, dtype=float).reshape(2)
    ww = float(w @ w)
    if ww <= eps:
        tau = 0.0
    else:
        tau = float(np.clip(-(q @ w) / (ww + eps), 0.0, horizon_s))
    dca = float(np.linalg.norm(q + w * tau))
    return tau, dca


COVARIANCE_MODELS = ("single_interval", "stepwise")


def _position_covariance(P: np.ndarray, tau: float, sigma_a: float,
                         covariance_model: str = "single_interval",
                         step_s: float | None = None) -> np.ndarray:
    """XY block (2,2) of the predicted covariance at look-ahead ``tau`` [s].

    single_interval: F(tau) P F(tau)^T + Q(tau).
    stepwise: P propagated through ceil(tau/step_s) CV predictions of length <= step_s.
    """
    P = np.asarray(P, dtype=float)
    if covariance_model == "single_interval":
        F = F_cv(tau)
        Pt = F @ P @ F.T + Q_cv(tau, sigma_a)
    elif covariance_model == "stepwise":
        if step_s is None or step_s <= 0:
            raise ValueError("stepwise covariance model needs step_s > 0")
        Pt = P
        remaining = float(tau)
        while remaining > 1e-12:
            dt = min(step_s, remaining)
            F = F_cv(dt)
            Pt = F @ Pt @ F.T + Q_cv(dt, sigma_a)
            remaining -= dt
    else:
        raise ValueError(f"unknown covariance_model {covariance_model!r}; use one of {COVARIANCE_MODELS}")
    Pxy = Pt[:2, :2]
    return 0.5 * (Pxy + Pxy.T)


def predict_track(track: Track, horizon_s: float, step_s: float, sigma_a: float,
                  covariance_model: str = "single_interval") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Open-loop CV prediction of a track from its own timestamp.

    Args:
        track: track with state (4,) [x, y, vx, vy] and covariance (4,4).
        horizon_s: look-ahead [s].  step_s: sample spacing [s] (> 0).
        sigma_a: process-noise acceleration std [m/s^2] (same as the tracker's).
        covariance_model: "single_interval" (spec, default) or "stepwise" (see module docstring).
    Returns:
        times (K,) look-ahead offsets [s] from 0 to horizon inclusive,
        means (K,2) predicted positions [m], covs (K,2,2) position covariances [m^2].
    """
    if step_s <= 0:
        raise ValueError("step_s must be > 0")
    K = int(np.floor(horizon_s / step_s + 1e-9)) + 1
    times = np.arange(K) * step_s
    x = np.asarray(track.state, dtype=float)
    P = np.asarray(track.covariance, dtype=float)
    means = x[:2][None, :] + times[:, None] * x[2:][None, :]
    covs = np.stack([_position_covariance(P, float(t), sigma_a, covariance_model, step_s) for t in times])
    return times, means, covs


def _track_at_time(track: Track, time_s: float, sigma_a: float) -> tuple[np.ndarray, np.ndarray]:
    """Track state/covariance extrapolated to ``time_s`` if it is newer than the track (else as is)."""
    dt = float(time_s) - float(track.time_s)
    if np.isfinite(dt) and dt > 0.0:
        return kf_predict(track.state, track.covariance, dt, sigma_a)
    return np.asarray(track.state, dtype=float), np.asarray(track.covariance, dtype=float)


def assess_track_risk(robot: RobotState, track: Track, robot_radius_m: float,
                      cfg_prediction: dict, sigma_a: float) -> RiskAssessment:
    """Collision risk of one track w.r.t. the robot over the prediction horizon.

    The track is first extrapolated to ``robot.time_s`` if the robot state is newer.

    - ``approaching``: radial closing speed exceeds ``approach_speed_eps_mps``
      (q.w < -eps*|q|), or the pair is currently overlapping.
    - ``currently overlapping``: |q| < r_robot + r_obs + nominal_margin_m (inside the nominal safety
      bubble right now, independent of motion).
    - ``risk``: (approaching and dca < required_clearance) or currently overlapping, with
      required_clearance = r_robot + r_obs + margin and
      margin = m0 + min(k*sqrt(lambda_max(P_xy(tau*))), max_uncertainty_margin_m) (heuristic, capped).
    - ``score`` in [0, 1]: 1 if overlapping; if approaching
      clip(required_clearance / max(dca, 1e-3), 0, 1) * (1 - tau/horizon); else 0.

    Args:
        robot: ego state (uses xy_world_m, v_world_mps = measured velocity, time_s).
        track: obstacle track.  robot_radius_m: robot disc radius [m].
        cfg_prediction: the ``prediction`` config section.  sigma_a: tracker accel std [m/s^2].
    """
    horizon = float(cfg_prediction["horizon_s"])
    m0 = float(cfg_prediction["nominal_margin_m"])
    margin_cap = float(cfg_prediction.get("max_uncertainty_margin_m", np.inf))
    k = float(cfg_prediction["uncertainty_k"])
    eps_v = float(cfg_prediction["approach_speed_eps_mps"])
    cov_model = str(cfg_prediction.get("covariance_model", "single_interval"))
    step_s = float(cfg_prediction["step_s"])

    x, P = _track_at_time(track, robot.time_s, sigma_a)
    q = x[:2] - np.asarray(robot.xy_world_m, dtype=float).reshape(2)
    w = x[2:] - np.asarray(robot.v_world_mps, dtype=float).reshape(2)
    tau, dca = time_of_closest_approach(q, w, horizon)

    Pxy = _position_covariance(P, tau, sigma_a, cov_model, step_s)
    lam_max = float(max(np.linalg.eigvalsh(Pxy)[-1], 0.0))
    margin = m0 + min(k * np.sqrt(lam_max), margin_cap)
    r_sum = float(robot_radius_m) + float(track.radius_m)
    required = r_sum + margin

    dist_now = float(np.linalg.norm(q))
    overlapping = dist_now < r_sum + m0
    closing = float(q @ w) < -eps_v * dist_now
    approaching = bool(closing or overlapping)
    risk = bool((approaching and dca < required) or overlapping)

    if overlapping:
        score = 1.0
    elif approaching:
        score = float(np.clip(required / max(dca, 1e-3), 0.0, 1.0) * (1.0 - tau / horizon))
    else:
        score = 0.0

    return RiskAssessment(track_id=int(track.track_id), tca_s=float(tau), dca_m=float(dca),
                          required_clearance_m=float(required), margin_m=float(margin),
                          approaching=approaching, risk=risk, score=float(np.clip(score, 0.0, 1.0)))


def assess_risk(robot: RobotState, tracks: list[Track], robot_radius_m: float,
                cfg_prediction: dict, sigma_a: float) -> RiskAssessment:
    """Worst risk over CONFIRMED tracks (tentative tracks are ignored).

    Ordering: risky tracks first, among them the smallest TCA (ties: higher score); if none is
    risky, the highest score (ties: smallest dca). Returns a copy of ``types.NO_RISK`` if there is
    no confirmed track.
    """
    results = [assess_track_risk(robot, t, robot_radius_m, cfg_prediction, sigma_a)
               for t in tracks if t.confirmed]
    if not results:
        return dataclasses.replace(NO_RISK)
    risky = [r for r in results if r.risk]
    if risky:
        return min(risky, key=lambda r: (r.tca_s, -r.score))
    return min(results, key=lambda r: (-r.score, r.dca_m))
