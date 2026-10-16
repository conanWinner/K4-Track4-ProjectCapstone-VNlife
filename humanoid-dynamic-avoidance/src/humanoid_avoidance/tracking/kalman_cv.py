"""Functional constant-velocity (CV) Kalman filter in the 2D world plane.

State ``x = [p_x, p_y, v_x, v_y]`` (m, m, m/s, m/s), measurement ``z = [p_x, p_y]`` (m).

Process-noise model (stated assumption)
---------------------------------------
``Q_cv`` uses the **piecewise-constant white acceleration** model (a.k.a. discrete white-noise
acceleration): the acceleration is assumed constant over each interval ``dt`` and independent
between intervals, with variance ``sigma_a^2`` per axis. Per axis this gives

    Q_axis = sigma_a^2 * [[dt^4/4, dt^3/2],
                          [dt^3/2, dt^2  ]]

This is *not* the continuous white-noise acceleration model (which has
``q * [[dt^3/3, dt^2/2], [dt^2/2, dt]]``); ``sigma_a`` is a tuning parameter, not a calibrated value.

All functions are pure: inputs are never modified, new arrays are returned.
"""
from __future__ import annotations

import numpy as np

#: Measurement matrix (2,4): observes position only.
H_CV = np.array([[1.0, 0.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0, 0.0]])
H_CV.setflags(write=False)

_I4 = np.eye(4)


def _symmetrize(P: np.ndarray) -> np.ndarray:
    return 0.5 * (P + P.T)


def F_cv(dt: float) -> np.ndarray:
    """CV state-transition matrix (4,4) for a time step ``dt`` [s]."""
    F = np.eye(4)
    F[0, 2] = dt
    F[1, 3] = dt
    return F


def Q_cv(dt: float, sigma_a: float) -> np.ndarray:
    """Process noise (4,4) for piecewise-constant white acceleration.

    Args:
        dt: time step [s] (>= 0).
        sigma_a: acceleration standard deviation per axis [m/s^2].
    Returns:
        (4,4) covariance in state order [px, py, vx, vy]:
        sigma_a^2 * [[dt^4/4, dt^3/2], [dt^3/2, dt^2]] per axis, no cross-axis terms.
    """
    s2 = float(sigma_a) ** 2
    q_pp = s2 * dt ** 4 / 4.0
    q_pv = s2 * dt ** 3 / 2.0
    q_vv = s2 * dt ** 2
    return np.array([[q_pp, 0.0, q_pv, 0.0],
                     [0.0, q_pp, 0.0, q_pv],
                     [q_pv, 0.0, q_vv, 0.0],
                     [0.0, q_pv, 0.0, q_vv]])


def kf_predict(x: np.ndarray, P: np.ndarray, dt: float, sigma_a: float) -> tuple[np.ndarray, np.ndarray]:
    """Propagate state (4,) and covariance (4,4) forward by ``dt`` [s].

    ``dt == 0`` is a no-op (copies returned); ``dt < 0`` raises ValueError (no retrodiction).
    Returns (x_pred (4,), P_pred (4,4)) with P symmetrised.
    """
    dt = float(dt)
    if dt < 0.0:
        raise ValueError(f"kf_predict: negative dt={dt}")
    x = np.asarray(x, dtype=float)
    P = np.asarray(P, dtype=float)
    if dt == 0.0:
        return x.copy(), P.copy()
    F = F_cv(dt)
    x_pred = F @ x
    P_pred = _symmetrize(F @ P @ F.T + Q_cv(dt, sigma_a))
    return x_pred, P_pred


def innovation(x: np.ndarray, P: np.ndarray, z: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Innovation y = z - Hx (2,) and its covariance S = H P H^T + R (2,2)."""
    y = np.asarray(z, dtype=float).reshape(2) - H_CV @ np.asarray(x, dtype=float)
    S = H_CV @ np.asarray(P, dtype=float) @ H_CV.T + np.asarray(R, dtype=float)
    return y, _symmetrize(S)


def mahalanobis_sq(x: np.ndarray, P: np.ndarray, z: np.ndarray, R: np.ndarray) -> float:
    """Squared Mahalanobis distance (z-Hx)^T S^-1 (z-Hx) of measurement z (2,) [unitless, chi^2 2 DoF]."""
    y, S = innovation(x, P, z, R)
    return float(y @ np.linalg.solve(S, y))


def kf_update(x: np.ndarray, P: np.ndarray, z: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Kalman measurement update with a position measurement.

    Args:
        x: (4,) predicted state.  P: (4,4) predicted covariance.
        z: (2,) measured position [m].  R: (2,2) measurement covariance [m^2].
    Returns:
        (x_post (4,), P_post (4,4), nis) where nis = y^T S^-1 y is the normalised innovation squared
        (expected value 2 for a consistent filter). P_post uses the Joseph form
        (I-KH) P (I-KH)^T + K R K^T and is symmetrised.
    """
    x = np.asarray(x, dtype=float)
    P = np.asarray(P, dtype=float)
    R = np.asarray(R, dtype=float)
    y, S = innovation(x, P, z, R)
    PHt = P @ H_CV.T                                  # (4,2)
    K = np.linalg.solve(S, PHt.T).T                   # (4,2) = P H^T S^-1 (S symmetric)
    x_post = x + K @ y
    IKH = _I4 - K @ H_CV
    P_post = _symmetrize(IKH @ P @ IKH.T + K @ R @ K.T)
    nis = float(y @ np.linalg.solve(S, y))
    return x_post, P_post, nis


def init_state(z: np.ndarray, R: np.ndarray, init_velocity_sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Initialise a track from a single position measurement.

    Velocity is set to 0 with a large standard deviation ``init_velocity_sigma`` [m/s] per axis
    (speed is never inferred from a single frame). Position covariance = R.
    Returns (x (4,), P (4,4)).
    """
    x = np.zeros(4)
    x[:2] = np.asarray(z, dtype=float).reshape(2)
    P = np.zeros((4, 4))
    P[:2, :2] = np.asarray(R, dtype=float)
    P[2, 2] = P[3, 3] = float(init_velocity_sigma) ** 2
    return x, _symmetrize(P)
