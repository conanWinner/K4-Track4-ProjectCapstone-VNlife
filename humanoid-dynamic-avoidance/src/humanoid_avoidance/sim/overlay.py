"""Debug overlays drawn into an MjvScene (passive viewer user_scn or offscreen Renderer scene).

Colours (spec section 13): hit points cyan, Kalman track yellow, predicted trajectory orange,
goal green. Ground truth is never drawn by this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import mujoco
import numpy as np

CYAN = np.array([0.0, 0.9, 1.0, 0.9], dtype=np.float32)
YELLOW = np.array([1.0, 0.9, 0.0, 0.9], dtype=np.float32)
ORANGE = np.array([1.0, 0.5, 0.0, 0.8], dtype=np.float32)
RED = np.array([1.0, 0.1, 0.1, 0.35], dtype=np.float32)
GREEN = np.array([0.1, 0.9, 0.2, 0.35], dtype=np.float32)
_EYE = np.eye(3).flatten()


@dataclass
class OverlayState:
    mode: str
    scenario: str
    robot: str
    goal_xy: np.ndarray
    time_s: float = 0.0
    hits_world: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    detections: list = field(default_factory=list)
    tracks: list = field(default_factory=list)
    predicted: Optional[np.ndarray] = None   # (K,2) mean predicted positions of the main track
    nav_state: str = "GO"
    cmd_vx: float = 0.0
    tca_s: float = float("nan")
    dca_m: float = float("nan")
    risk: bool = False
    robot_xy: np.ndarray = field(default_factory=lambda: np.zeros(2))
    clearance: float = float("nan")
    collided: bool = False

    def update(self, time_s, scan, detections, tracks, cmd, debug, robot_xy, clearance, predicted):
        self.time_s = time_s
        if scan is not None and np.any(scan.valid_hit):
            a = scan.angles_rad[scan.valid_hit]
            r = scan.ranges_m[scan.valid_hit]
            local = np.stack([r * np.cos(a), r * np.sin(a), np.zeros_like(r)], axis=1)
            self.hits_world = local @ np.asarray(scan.sensor_rot_world).T + np.asarray(scan.sensor_pos_world)
        else:
            self.hits_world = np.zeros((0, 3))
        self.detections = list(detections)
        self.tracks = list(tracks)
        self.predicted = predicted
        self.nav_state = cmd.mode
        self.cmd_vx = cmd.vx_mps
        self.tca_s = float(debug.get("tca_s", float("nan")))
        self.dca_m = float(debug.get("dca_m", float("nan")))
        self.risk = bool(debug.get("risk", False))
        self.robot_xy = np.asarray(robot_xy, dtype=float)
        self.clearance = clearance


def _add(scn: mujoco.MjvScene, gtype, size, pos, rgba, label: str = "") -> None:
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.asarray(size, dtype=np.float64), np.asarray(pos, dtype=np.float64),
                        _EYE, rgba)
    if label:
        g.label = label[:99]
    scn.ngeom += 1


def _add_line(scn: mujoco.MjvScene, p0, p1, width, rgba) -> None:
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), _EYE, rgba)
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
                         np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64))
    scn.ngeom += 1


def draw_overlays(scn: mujoco.MjvScene, st: OverlayState) -> None:
    sphere = mujoco.mjtGeom.mjGEOM_SPHERE
    cyl = mujoco.mjtGeom.mjGEOM_CYLINDER
    for p in st.hits_world:
        _add(scn, sphere, [0.03, 0, 0], p, CYAN)
    for tr in st.tracks:
        x, y = tr.state[:2]
        _add(scn, cyl, [max(tr.radius_m, 0.1), 0.01, 0], [x, y, 0.02], YELLOW,
             label=f"T{tr.track_id} v=({tr.state[2]:+.2f},{tr.state[3]:+.2f})")
        # uncertainty disc ~ 2 sigma of position
        sig = float(np.sqrt(max(np.linalg.eigvalsh(tr.covariance[:2, :2]).max(), 0.0)))
        _add(scn, cyl, [max(tr.radius_m, 0.1) + 2 * sig, 0.005, 0], [x, y, 0.01],
             np.array([1.0, 0.9, 0.0, 0.25], dtype=np.float32))
    if st.predicted is not None and len(st.predicted) > 1:
        pts = np.asarray(st.predicted)
        for i in range(0, len(pts) - 1, 2):    # dashed
            _add_line(scn, [*pts[i], 0.05], [*pts[i + 1], 0.05], 0.025, ORANGE)
    # robot status ring
    ring = RED if st.risk else GREEN
    _add(scn, cyl, [0.45, 0.004, 0], [st.robot_xy[0], st.robot_xy[1], 0.006], ring,
         label=f"{st.nav_state}")


def hud_lines(st: OverlayState) -> list[str]:
    tca = "-" if not np.isfinite(st.tca_s) else f"{st.tca_s:.2f}s"
    dca = "-" if not np.isfinite(st.dca_m) else f"{st.dca_m:.2f}m"
    clr = "-" if not np.isfinite(st.clearance) else f"{st.clearance:.2f}m"
    return [
        f"{st.robot.upper()} | mode={st.mode} | scenario={st.scenario} | t={st.time_s:5.1f}s",
        f"state={st.nav_state}  cmd_vx={st.cmd_vx:+.2f} m/s  risk={'YES' if st.risk else 'no'}",
        f"TCA={tca}  dCA={dca}  tracks={len(st.tracks)}  dets={len(st.detections)}",
        f"GT clearance (eval only)={clr}  collision={'YES' if st.collided else 'no'}",
    ]
