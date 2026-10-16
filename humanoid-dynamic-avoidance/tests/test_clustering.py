"""Clustering / map-filter tests, plus a top-down scan check figure (outputs/figures/scan_check.png)."""
import copy
from pathlib import Path

import numpy as np
import pytest

from humanoid_avoidance.config import OUTPUT_DIR, load_config
from humanoid_avoidance.perception.clustering import (cluster_scan, filter_map_walls,
                                                       point_segment_distances, scan_to_world_points)
from humanoid_avoidance.perception.range_sensor import RangeSensor
from humanoid_avoidance.sim.amr_adapter import AmrAdapter
from humanoid_avoidance.sim.scene_builder import build_scene
from humanoid_avoidance.types import Scan


def _setup(actor_xy, robot_xy=(0.0, 0.0), yaw=0.0, noise=True, seed=0, scenario_extra=None):
    cfg = copy.deepcopy(load_config())
    if not noise:
        cfg["sensor"]["noise_std_m"] = 0.0
        cfg["sensor"]["dropout_probability"] = 0.0
    robot = AmrAdapter(cfg)
    scen = {"actor": actor_xy is not None}
    if actor_xy is not None:
        scen["waypoints"] = [[0.0, actor_xy[0], actor_xy[1]]]
    if scenario_extra:
        scen.update(scenario_extra)
    h = build_scene(robot, cfg, scen)
    robot.reset(h.model, h.data, np.asarray(robot_xy), yaw)
    sensor = RangeSensor(cfg, np.random.default_rng(seed))
    pos, R = robot.sensor_pose(h.model, h.data)
    scan = sensor.scan(h.model, h.data, 0.0, pos, R, h.geomgroup_mask())
    return cfg, h, scan


def test_point_segment_distance():
    segs = np.array([[[0.0, 0.0], [2.0, 0.0]]])
    pts = np.array([[1.0, 1.0], [-1.0, 0.0], [3.0, 4.0], [0.5, 0.0]])
    d = point_segment_distances(pts, segs)[:, 0]
    assert np.allclose(d, [1.0, 1.0, np.hypot(1.0, 4.0), 0.0])
    keep = filter_map_walls(pts, segs, 0.15)
    assert keep.tolist() == [True, True, True, False]


def test_scan_to_world_points_transform():
    yaw = np.pi / 2
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    scan = Scan(time_s=0.0, angles_rad=np.array([0.0, 0.5]), ranges_m=np.array([1.0, 2.0]),
                valid_hit=np.array([True, False]), sensor_pos_world=np.array([1.0, 2.0, 1.0]),
                sensor_rot_world=R)
    pts = scan_to_world_points(scan)
    assert pts.shape == (1, 2)
    assert np.allclose(pts[0], [1.0, 3.0])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_single_actor_detection_in_empty_corridor(seed):
    actor = np.array([2.0, 0.3])
    cfg, h, scan = _setup(actor, seed=seed)
    dets = cluster_scan(scan, h.wall_segments, cfg)
    assert len(dets) == 1
    det = dets[0]
    r = cfg["world"]["actor_radius_m"]
    dist_to_center = np.linalg.norm(det.xy_world_m - actor)
    # visible-surface centroid: inside the actor disc, biased toward the sensor
    assert 0.3 * r < dist_to_center < r + 0.05
    assert det.xy_world_m[0] < actor[0]
    assert det.num_points >= cfg["perception"]["min_cluster_points"]
    assert 0.5 * r < det.radius_m < r + 0.2
    sig2 = cfg["perception"]["measurement_sigma_m"] ** 2
    assert det.covariance_xy[0, 0] >= sig2 and det.covariance_xy[1, 1] >= sig2
    assert det.covariance_xy[0, 1] == 0.0


def test_walls_filtered_out_empty_corridor():
    for seed in range(5):
        cfg, h, scan = _setup(None, seed=seed)
        assert scan.valid_hit.sum() > 10
        assert cluster_scan(scan, h.wall_segments, cfg) == []


def test_walls_dropped_by_extent_without_map():
    cfg, h, scan = _setup(None, noise=False)
    dets = cluster_scan(scan, np.zeros((0, 2, 2)), cfg)
    # without the map, long wall clusters are dropped by extent; at most short fragments remain
    for d in dets:
        assert 2 * (d.radius_m - cfg["perception"]["radius_inflation_m"]) <= cfg["perception"]["max_obstacle_extent_m"]


def test_actor_near_wall_not_merged():
    hw = load_config()["world"]["corridor_half_width_m"]
    r = load_config()["world"]["actor_radius_m"]
    actor = np.array([2.0, hw - r - 0.3])  # 0.3 m gap to the wall
    cfg, h, scan = _setup(actor, noise=False)
    dets = cluster_scan(scan, h.wall_segments, cfg)
    assert len(dets) == 1
    assert np.linalg.norm(dets[0].xy_world_m - actor) < r + 0.05


def test_occluder_is_detected_as_unknown_obstacle():
    cfg, h, scan = _setup(None, noise=False,
                          scenario_extra={"occluder": {"xy": [1.6, -0.9], "size": [0.25, 0.25, 0.8]}})
    dets = cluster_scan(scan, h.wall_segments, cfg)
    assert len(dets) == 1
    assert np.linalg.norm(dets[0].xy_world_m - np.array([1.6, -0.9])) < 0.4


def render_scan_check(out_png: Path) -> Path:
    """Top-down figure of one scan: walls, occluder, hits, detections and GT actor (eval only)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    actor = np.array([2.2, 0.6])
    occ = {"xy": [1.6, -0.9], "size": [0.25, 0.25, 0.8]}
    cfg, h, scan = _setup(actor, seed=7, scenario_extra={"occluder": occ})
    pts = scan_to_world_points(scan)
    keep = filter_map_walls(pts, h.wall_segments, cfg["perception"]["wall_filter_margin_m"])
    dets = cluster_scan(scan, h.wall_segments, cfg)

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=120)
    for seg in h.wall_segments:
        ax.plot(seg[:, 0], seg[:, 1], color="0.3", lw=3, label="known walls" if seg is h.wall_segments[0] else None)
    ax.add_patch(Rectangle((occ["xy"][0] - occ["size"][0], occ["xy"][1] - occ["size"][1]),
                           2 * occ["size"][0], 2 * occ["size"][1], fc="0.75", ec="0.4", label="occluder (unmapped)"))
    ax.add_patch(Circle(actor, cfg["world"]["actor_radius_m"], fc="none", ec="tab:orange", ls="--", lw=1.5,
                        label="actor GT (eval only)"))
    sp = scan.sensor_pos_world[:2]
    ax.add_patch(Circle((0, 0), cfg["navigation"]["nominal_robot_radius_m"], fc="tab:blue", alpha=0.3, label="robot"))
    end = sp + scan.ranges_m[:, None] * np.column_stack([np.cos(scan.angles_rad), np.sin(scan.angles_rad)])
    for e, v in zip(end, scan.valid_hit):
        ax.plot([sp[0], e[0]], [sp[1], e[1]], color="tab:green" if v else "0.85", lw=0.4, zorder=0)
    ax.scatter(pts[~keep, 0], pts[~keep, 1], s=10, c="0.5", label="hits on map walls")
    ax.scatter(pts[keep, 0], pts[keep, 1], s=14, c="tab:red", label="non-map hits")
    for i, d in enumerate(dets):
        ax.add_patch(Circle(d.xy_world_m, d.radius_m, fc="none", ec="tab:cyan", lw=2,
                            label="detection (radius)" if i == 0 else None))
        ax.plot(*d.xy_world_m, "x", color="tab:cyan", ms=8)
    ax.set_aspect("equal")
    ax.set_xlim(-1.8, 7.8)
    ax.set_ylim(-2.4, 2.4)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Scan check: {int(scan.valid_hit.sum())} hits, {len(dets)} detections (t={scan.time_s:.1f}s)")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


def test_render_scan_check():
    out = render_scan_check(OUTPUT_DIR / "figures" / "scan_check.png")
    assert out.exists() and out.stat().st_size > 10_000


if __name__ == "__main__":
    print(render_scan_check(OUTPUT_DIR / "figures" / "scan_check.png"))
