"""Scan -> obstacle detections: world projection, known-map wall filter, distance-gap clustering.

A Detection centroid is the centroid of the VISIBLE SURFACE points of a cluster. It is NOT the
object center: for a cylinder of radius r seen from one side it is biased toward the sensor by
roughly 0.6-0.8 r. Downstream code must treat it as a noisy measurement (R reflects this).
"""
from __future__ import annotations

import numpy as np

from humanoid_avoidance.geometry.transforms import polar_to_points, pose3, transform_points_3d
from humanoid_avoidance.types import Detection, Scan


def scan_to_world_points(scan: Scan) -> np.ndarray:
    """Valid hits of `scan` projected to world XY.

    Returns:
        (K, 2) world points, m, in increasing ray-angle order (K = number of valid hits).
    """
    valid = np.asarray(scan.valid_hit, dtype=bool)
    if not np.any(valid):
        return np.zeros((0, 2))
    pts2 = polar_to_points(np.asarray(scan.ranges_m)[valid], np.asarray(scan.angles_rad)[valid])
    pts3 = np.column_stack([pts2, np.zeros(len(pts2))])
    T = pose3(np.asarray(scan.sensor_pos_world, float), np.asarray(scan.sensor_rot_world, float))
    return transform_points_3d(T, pts3)[:, :2]


def point_segment_distances(points: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """Distance from every point to every segment.

    Args:
        points: (K, 2) m.  segments: (M, 2, 2) m.
    Returns:
        (K, M) Euclidean distances, m.
    """
    P = np.asarray(points, float).reshape(-1, 1, 2)
    S = np.asarray(segments, float).reshape(-1, 2, 2)
    a = S[None, :, 0, :]
    d = S[None, :, 1, :] - a
    dd = np.sum(d * d, axis=-1)
    t = np.where(dd > 0.0, np.sum((P - a) * d, axis=-1) / np.where(dd > 0.0, dd, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    closest = a + t[..., None] * d
    return np.linalg.norm(P - closest, axis=-1)


def filter_map_walls(points: np.ndarray, wall_segments: np.ndarray, margin: float) -> np.ndarray:
    """Mask of points that are NOT explained by the known map.

    Args:
        points: (K, 2) world points, m.
        wall_segments: (M, 2, 2) known wall lines, m.
        margin: points closer than this to any wall segment are removed, m.
    Returns:
        (K,) bool, True = keep (not a known wall).
    """
    points = np.asarray(points, float).reshape(-1, 2)
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    segs = np.asarray(wall_segments, float).reshape(-1, 2, 2)
    if len(segs) == 0:
        return np.ones(len(points), dtype=bool)
    return np.min(point_segment_distances(points, segs), axis=1) >= float(margin)


def _extent(pts: np.ndarray) -> float:
    """Largest pairwise distance between points (K, 2), m."""
    if len(pts) < 2:
        return 0.0
    diff = pts[:, None, :] - pts[None, :, :]
    return float(np.sqrt(np.max(np.sum(diff * diff, axis=-1))))


def cluster_scan(scan: Scan, wall_segments: np.ndarray, cfg: dict) -> list[Detection]:
    """Detect non-map obstacles in a scan.

    Steps: project valid hits to world (angular order) -> drop points within
    perception.wall_filter_margin_m of known walls -> split consecutive remaining points where
    the gap exceeds perception.cluster_gap_m -> drop clusters with fewer than
    min_cluster_points or extent > max_obstacle_extent_m.

    Each Detection: xy_world_m = visible-surface centroid (NOT the object center),
    radius_m = extent / 2 + radius_inflation_m,
    covariance_xy = diag(max(sigma^2, var_x), max(sigma^2, var_y)), sigma = measurement_sigma_m.

    Args:
        scan: Scan from RangeSensor.
        wall_segments: (M, 2, 2) known map walls, m.
        cfg: full config dict (uses `perception`) or the perception section itself.
    """
    p = cfg["perception"] if "perception" in cfg else cfg
    gap = float(p["cluster_gap_m"])
    min_pts = int(p["min_cluster_points"])
    max_extent = float(p["max_obstacle_extent_m"])
    margin = float(p["wall_filter_margin_m"])
    inflation = float(p["radius_inflation_m"])
    sigma2 = float(p["measurement_sigma_m"]) ** 2

    pts = scan_to_world_points(scan)
    if len(pts) == 0:
        return []
    pts = pts[filter_map_walls(pts, wall_segments, margin)]
    if len(pts) == 0:
        return []

    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    breaks = np.flatnonzero(steps > gap) + 1
    detections: list[Detection] = []
    for cl in np.split(pts, breaks):
        if len(cl) < min_pts:
            continue
        ext = _extent(cl)
        if ext > max_extent:
            continue
        var = np.var(cl, axis=0)
        detections.append(Detection(
            time_s=float(scan.time_s),
            xy_world_m=cl.mean(axis=0),
            radius_m=0.5 * ext + inflation,
            covariance_xy=np.diag([max(sigma2, float(var[0])), max(sigma2, float(var[1]))]),
            num_points=int(len(cl)),
        ))
    return detections
