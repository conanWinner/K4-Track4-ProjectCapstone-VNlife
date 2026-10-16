"""Rigid-body transform helpers (SE(2) / SE(3)) and frame conversions.

Conventions
-----------
- Units: metres, radians, m/s.
- World frame: XY ground plane, Z up, yaw counter-clockwise positive.
- A homogeneous transform ``T_parent_local`` maps points expressed in the *local* frame into the
  *parent* frame: ``p_parent = T @ [p_local, 1]``.
- Quaternions follow the MuJoCo convention ``(w, x, y, z)``.
"""
from __future__ import annotations

import numpy as np

_TWO_PI = 2.0 * np.pi


def wrap_angle(a: float | np.ndarray) -> float | np.ndarray:
    """Wrap angle(s) to the half-open interval [-pi, pi).

    Args:
        a: angle in rad, scalar or array of any shape.
    Returns:
        Wrapped angle(s), same shape/type as the input (python float for scalar input).
    """
    wrapped = np.mod(np.asarray(a, dtype=float) + np.pi, _TWO_PI) - np.pi
    if np.ndim(wrapped) == 0:
        return float(wrapped)
    return wrapped


def rot2d(yaw: float) -> np.ndarray:
    """2D rotation matrix (2,2) for a CCW rotation of ``yaw`` rad."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=float)


def se2(x: float, y: float, yaw: float) -> np.ndarray:
    """Homogeneous SE(2) transform (3,3) of a frame at (x, y) [m] with heading yaw [rad].

    Maps local-frame points into the parent frame.
    """
    T = np.eye(3)
    T[:2, :2] = rot2d(yaw)
    T[:2, 2] = (x, y)
    return T


def se2_inverse(T: np.ndarray) -> np.ndarray:
    """Inverse of a homogeneous SE(2) transform (3,3) -> (3,3), using R^T (no general inverse)."""
    T = np.asarray(T, dtype=float)
    R_t = T[:2, :2].T
    Ti = np.eye(3)
    Ti[:2, :2] = R_t
    Ti[:2, 2] = -R_t @ T[:2, 2]
    return Ti


def transform_points_2d(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply SE(2) transform T (3,3) to points (N,2) [m] -> (N,2). A single (2,) point is accepted."""
    T = np.asarray(T, dtype=float)
    p = np.asarray(pts, dtype=float)
    single = p.ndim == 1
    p = np.atleast_2d(p)
    out = p @ T[:2, :2].T + T[:2, 2]
    return out[0] if single else out


def pose3(pos: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Homogeneous SE(3) transform (4,4) from position (3,) [m] and rotation matrix (3,3)."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=float)
    T[:3, 3] = np.asarray(pos, dtype=float).reshape(3)
    return T


def transform_points_3d(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply SE(3) transform T (4,4) to points (N,3) [m] -> (N,3). A single (3,) point is accepted."""
    T = np.asarray(T, dtype=float)
    p = np.asarray(pts, dtype=float)
    single = p.ndim == 1
    p = np.atleast_2d(p)
    out = p @ T[:3, :3].T + T[:3, 3]
    return out[0] if single else out


def quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Unit quaternion (4,) in MuJoCo order (w, x, y, z) -> rotation matrix (3,3).

    The quaternion is normalised first; a zero quaternion raises ValueError.
    """
    q = np.asarray(q, dtype=float).reshape(4)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("zero-norm quaternion")
    w, x, y, z = q / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def yaw_from_rotmat(R: np.ndarray) -> float:
    """Heading [rad, in [-pi, pi)] of a body whose orientation is R (3,3): angle of its x-axis in world XY."""
    R = np.asarray(R, dtype=float)
    return float(wrap_angle(np.arctan2(R[1, 0], R[0, 0])))


def quat_wxyz_to_yaw(q: np.ndarray) -> float:
    """Heading [rad, in [-pi, pi)] from a MuJoCo (w, x, y, z) quaternion (4,)."""
    return yaw_from_rotmat(quat_wxyz_to_rotmat(q))


def body_to_world_vel(v_body: np.ndarray, yaw: float) -> np.ndarray:
    """Planar velocity (2,) [m/s] in the body frame -> world frame, given body yaw [rad]."""
    return rot2d(yaw) @ np.asarray(v_body, dtype=float).reshape(2)


def world_to_body_vel(v_world: np.ndarray, yaw: float) -> np.ndarray:
    """Planar velocity (2,) [m/s] in the world frame -> body frame, given body yaw [rad]."""
    return rot2d(yaw).T @ np.asarray(v_world, dtype=float).reshape(2)


def polar_to_points(ranges: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Range/bearing pairs -> sensor-frame XY points.

    Args:
        ranges: (N,) ranges [m].
        angles: (N,) bearings [rad] relative to sensor +x, CCW positive.
    Returns:
        (N,2) points [m] in the sensor frame.
    """
    r = np.asarray(ranges, dtype=float).reshape(-1)
    a = np.asarray(angles, dtype=float).reshape(-1)
    if r.shape != a.shape:
        raise ValueError(f"ranges {r.shape} and angles {a.shape} must match")
    return np.column_stack((r * np.cos(a), r * np.sin(a)))
