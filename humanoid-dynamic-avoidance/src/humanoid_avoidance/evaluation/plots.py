"""Evaluation figures from per-tick CSV logs (matplotlib Agg backend, PNG at 120 dpi).

Ground-truth actor positions are drawn ONLY here (labelled "GT (eval only)").
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DPI = 120
STATE_COLORS = {
    "GO": "#2ca02c", "SLOW": "#ff7f0e", "YIELD": "#d62728", "RESUME": "#1f77b4",
    "STOP_SENSOR_FAILURE": "#7f7f7f", "GOAL_REACHED": "#9467bd", "MANUAL": "#17becf",
}
MODE_COLORS = {"reactive": "#e377c2", "predictive": "#1f77b4", "manual": "#7f7f7f"}


def _col(df: pd.DataFrame, c: str) -> np.ndarray:
    if c not in df.columns:
        return np.full(len(df), np.nan)
    return pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)


def _save(fig, out_png) -> Path:
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_episode_topdown(csv_path, out_png, cfg: dict) -> Path:
    """Top-down XY view of one episode.

    Walls at y = +-world.corridor_half_width_m, robot path coloured by nav_state, GT actor path
    (dashed grey, eval only), nearest detection per tick (cyan), main track (yellow), goal (green),
    contacts (red x). Returns the PNG path.
    """
    df = pd.read_csv(csv_path)
    w = cfg["world"]
    hw = float(w["corridor_half_width_m"])
    x0 = float(w.get("corridor_x_min_m", -1.5))
    x1 = x0 + float(w.get("corridor_length_m", 9.0))
    goal = np.asarray(cfg["simulation"]["goal_xy"], dtype=float)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    for y in (-hw, hw):
        ax.plot([x0, x1], [y, y], color="black", lw=3, solid_capstyle="butt")

    gx, gy = _col(df, "obstacle_gt_x_eval_only"), _col(df, "obstacle_gt_y_eval_only")
    if np.isfinite(gx).any():
        ax.plot(gx, gy, ls="--", color="grey", lw=1.5, label="actor GT (eval only)")
    dx, dy = _col(df, "detection_x"), _col(df, "detection_y")
    if np.isfinite(dx).any():
        ax.scatter(dx, dy, s=8, color="cyan", edgecolors="none", label="detections (latest, nearest)")
    tx, ty = _col(df, "track_x"), _col(df, "track_y")
    if np.isfinite(tx).any():
        ax.scatter(tx, ty, s=10, color="yellow", edgecolors="#8a7a00", linewidths=0.4,
                   label="Kalman track", zorder=3)

    rx, ry = _col(df, "robot_x"), _col(df, "robot_y")
    states = df["nav_state"].astype(str).to_numpy() if "nav_state" in df.columns else np.array([])
    ax.plot(rx, ry, color="#444444", lw=0.6, zorder=2)
    for s in [s for s in STATE_COLORS if s in set(states)]:
        m = states == s
        ax.scatter(rx[m], ry[m], s=12, color=STATE_COLORS[s], label=f"robot: {s}", zorder=4)
    cf = _col(df, "contact_flag") > 0.5
    if cf.any():
        ax.scatter(rx[cf], ry[cf], marker="x", s=60, color="red", label="contact", zorder=5)
    ax.scatter([goal[0]], [goal[1]], marker="*", s=250, color="green", edgecolors="black",
               label="goal", zorder=5)
    if len(rx):
        ax.scatter([rx[0]], [ry[0]], marker="o", s=60, facecolors="none", edgecolors="black",
                   label="start", zorder=5)

    title = Path(csv_path).parent.name
    if len(df) and {"scenario", "mode", "seed"} <= set(df.columns):
        title = f"{df['scenario'].iloc[0]} | {df['mode'].iloc[0]} | seed {df['seed'].iloc[0]}"
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xlim(x0, x1)
    ax.set_ylim(-hw - 0.5, hw + 0.5)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    return _save(fig, out_png)


def plot_comparison(summary_df: pd.DataFrame, out_png) -> Path:
    """Grouped bars (scenario on x, one bar per mode) for collision-free success, mean min
    clearance and mean time-to-goal. Missing values are annotated 'n/a'; n = episodes."""
    metrics = [("collision_free_success", "collision-free success [%]", 100.0),
               ("mean_min_clearance_m", "mean min clearance [m]\n(circle approx., GT)", 1.0),
               ("mean_time_to_goal_s", "mean time-to-goal [s]\n(reached episodes)", 1.0)]
    scen = list(dict.fromkeys(summary_df["scenario"])) if len(summary_df) else []
    modes = list(dict.fromkeys(summary_df["mode"])) if len(summary_df) else []
    fig, axes = plt.subplots(1, 3, figsize=(4.2 * 3, 4.2))
    width = 0.8 / max(len(modes), 1)
    x = np.arange(len(scen))
    for ax, (col, label, scale) in zip(axes, metrics):
        for j, mode in enumerate(modes):
            vals, ns = [], []
            for s in scen:
                r = summary_df[(summary_df["scenario"] == s) & (summary_df["mode"] == mode)]
                v = float(r[col].iloc[0]) if len(r) and col in r else np.nan
                vals.append(v * scale if np.isfinite(v) else np.nan)
                ns.append(int(r["episodes"].iloc[0]) if len(r) else 0)
            pos = x - 0.4 + width * (j + 0.5)
            ax.bar(pos, np.nan_to_num(vals, nan=0.0), width, color=MODE_COLORS.get(mode, None),
                   label=mode)
            for p, v, n in zip(pos, vals, ns):
                txt = "n/a" if not np.isfinite(v) else (f"{v:.0f}" if scale == 100.0 else f"{v:.2f}")
                ax.text(p, (v if np.isfinite(v) else 0.0), f"{txt}\nn={n}", ha="center",
                        va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(scen, rotation=20, ha="right")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.3)
        ax.margins(y=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Reactive vs predictive (simulation, measured)")
    fig.tight_layout()
    return _save(fig, out_png)


def plot_prediction_error(csv_path, out_png, horizon_s: float | None = None) -> Path:
    """Track position error vs GT over time (eval only), plus the constant-velocity prediction
    error at `horizon_s` (default: 1.0 s) attributed to the prediction start time. YIELD
    intervals are shaded red, SLOW orange."""
    df = pd.read_csv(csv_path)
    t = _col(df, "sim_time_s")
    gx, gy = _col(df, "obstacle_gt_x_eval_only"), _col(df, "obstacle_gt_y_eval_only")
    tx, ty = _col(df, "track_x"), _col(df, "track_y")
    vx, vy = _col(df, "track_vx"), _col(df, "track_vy")
    h = 1.0 if horizon_s is None else float(horizon_s)

    fig, ax = plt.subplots(figsize=(9, 3.8))
    err_now = np.hypot(tx - gx, ty - gy)
    ax.plot(t, err_now, color="#1f77b4", lw=1.2, label="|track - GT| now")
    gt_ok = np.isfinite(t) & np.isfinite(gx) & np.isfinite(gy)
    if gt_ok.sum() >= 2:
        tg = t[gt_ok]
        th = t + h
        ok = np.isfinite(tx) & np.isfinite(vx) & (th <= tg.max()) & (th >= tg.min())
        e_h = np.full(len(t), np.nan)
        e_h[ok] = np.hypot(tx[ok] + vx[ok] * h - np.interp(th[ok], tg, gx[gt_ok]),
                           ty[ok] + vy[ok] * h - np.interp(th[ok], tg, gy[gt_ok]))
        ax.plot(t, e_h, color="#ff7f0e", lw=1.2, label=f"CV prediction error @ {h:.1f} s")
    if "nav_state" in df.columns and len(t) > 1:
        st = df["nav_state"].astype(str).to_numpy()
        for name, colr in (("YIELD", "#d62728"), ("SLOW", "#ff7f0e")):
            m = st == name
            first = True
            for i in np.flatnonzero(m):
                t_end = t[i + 1] if i + 1 < len(t) else t[i]
                ax.axvspan(t[i], t_end, color=colr, alpha=0.12, lw=0,
                           label=name if first else None)
                first = False
    ax.set_xlabel("sim time [s]")
    ax.set_ylabel("error [m]")
    title = "prediction error (GT eval only)"
    if len(df) and {"scenario", "mode"} <= set(df.columns):
        title = f"{df['scenario'].iloc[0]} | {df['mode'].iloc[0]} — {title}"
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, out_png)
