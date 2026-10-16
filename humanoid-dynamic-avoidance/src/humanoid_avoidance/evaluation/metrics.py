"""Episode metrics and summary table (README section 8). Evaluation code: may read ground truth.

All metrics are computed from the per-tick CSV log written by `logging.csv_logger.CsvLogger`
(one row per planner tick). Nothing here feeds back into perception / tracking / planning.

Definitions
-----------
goal_reached           final-row goal_distance_m <= navigation.goal_tolerance_m.
collision              at least one contact event: a run of consecutive rows with contact_flag=1
                       lasting >= evaluation.collision_min_duration_s (0 -> any contact).
                       contact_flag covers the actor, static boxes and occluders,
                       so floor / foot contacts are never counted.
fall                   any row with fall_flag=1 (fall detection itself is done by the robot adapter).
collision_free_success goal_reached and not collision and not fall.
min_clearance_m        min over the episode of clearance_gt_m = |p_robot - p_actor_GT| - r_robot -
                       r_actor, i.e. a CIRCLE APPROXIMATION of both bodies (surface distance,
                       negative = overlap). Static-box clearance uses XY box distance minus
                       robot radius; the logged value is the minimum across obstacles.
                       NaN when the scenario has no obstacles.
time_to_goal_s         first sim time with goal_distance_m <= tolerance minus first logged sim
                       time; NaN if the goal was never reached.
yield_count            number of entries into nav_state == YIELD.
total_yield_s          time spent in YIELD; each row lasts until the next row's sim_time_s.
false_stop_count       number of YIELD entries (stop events) for which the ground-truth clearance
                       stays > evaluation.false_stop_clearance_m during the whole window
                       [t_entry, t_entry + prediction.horizon_s] (rows with no actor count as
                       infinite clearance). I.e. "the robot stopped although, within the planner's
                       own prediction horizon, the actor never came closer than the threshold".
                       Caveat: the window is observed WITH the robot stopped, so a stop that
                       actually prevented a conflict can still be counted; it is a simple,
                       documented proxy, not a counterfactual simulation.
track_pos_error_m      mean |track_xy - actor_GT_xy| at the same tick (rows with a track and GT).
                       Includes the visible-surface bias (detections are surface centroids).
pred_ade_m / pred_fde_m constant-velocity prediction error of the logged track: prediction
                       p + v*tau for tau = step_s..horizon_s vs GT interpolated at t + tau;
                       ADE = mean over tau and ticks, FDE = mean at tau = horizon_s. Single-actor
                       scenarios only (GT column holds one actor).
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- contact
def robot_actor_contact(model, data, robot_body_ids: set[int], actor_geom_ids: set[int]) -> bool:
    """True if any active MuJoCo contact pairs an actor geom with a geom of a robot body.

    Args:
        model: mujoco.MjModel (uses model.geom_bodyid).
        data: mujoco.MjData (iterates data.contact[:data.ncon]).
        robot_body_ids: body ids belonging to the robot.
        actor_geom_ids: geom ids of the dynamic actor.
    Only contacts with penetration distance <= 0 count (contacts inside a positive margin but not
    touching are ignored). Floor/wall contacts are never counted.
    """
    if not actor_geom_ids or not robot_body_ids:
        return False
    geom_body = model.geom_bodyid
    for i in range(int(data.ncon)):
        c = data.contact[i]
        if float(c.dist) > 0.0:
            continue
        g1, g2 = int(c.geom1), int(c.geom2)
        if g1 in actor_geom_ids and int(geom_body[g2]) in robot_body_ids:
            return True
        if g2 in actor_geom_ids and int(geom_body[g1]) in robot_body_ids:
            return True
    return False


# --------------------------------------------------------------------------- helpers
def _num(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), np.nan)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs of consecutive True runs."""
    runs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _row_durations(t: np.ndarray) -> np.ndarray:
    """Duration of each row = time until next row (last row: 0)."""
    if len(t) == 0:
        return t
    d = np.diff(t)
    return np.append(np.maximum(d, 0.0), 0.0)


def _nan_stat(x: np.ndarray, fn) -> float:
    x = x[np.isfinite(x)]
    return float(fn(x)) if len(x) else float("nan")


def prediction_errors(df: pd.DataFrame, horizon_s: float, step_s: float) -> dict:
    """Track-vs-GT errors (eval only). Returns track_pos_error_m, pred_ade_m, pred_fde_m (NaN if
    there is no tick with both a track and GT)."""
    out = {"track_pos_error_m": float("nan"), "pred_ade_m": float("nan"), "pred_fde_m": float("nan")}
    t = _num(df, "sim_time_s")
    gx, gy = _num(df, "obstacle_gt_x_eval_only"), _num(df, "obstacle_gt_y_eval_only")
    tx, ty = _num(df, "track_x"), _num(df, "track_y")
    tvx, tvy = _num(df, "track_vx"), _num(df, "track_vy")
    gt_ok = np.isfinite(t) & np.isfinite(gx) & np.isfinite(gy)
    if gt_ok.sum() < 2:
        return out
    now_ok = gt_ok & np.isfinite(tx) & np.isfinite(ty)
    if now_ok.any():
        out["track_pos_error_m"] = float(np.mean(np.hypot(tx[now_ok] - gx[now_ok], ty[now_ok] - gy[now_ok])))

    tg, xg, yg = t[gt_ok], gx[gt_ok], gy[gt_ok]
    t_max = tg.max()
    trk = np.isfinite(t) & np.isfinite(tx) & np.isfinite(ty) & np.isfinite(tvx) & np.isfinite(tvy)
    if not trk.any() or step_s <= 0 or horizon_s <= 0:
        return out
    taus = np.arange(1, int(round(horizon_s / step_s)) + 1) * step_s     # (K,)
    t0 = t[trk][:, None]                                                  # (M,1)
    tt = t0 + taus[None, :]                                               # (M,K)
    px = tx[trk][:, None] + tvx[trk][:, None] * taus[None, :]
    py = ty[trk][:, None] + tvy[trk][:, None] * taus[None, :]
    ex = px - np.interp(tt, tg, xg)
    ey = py - np.interp(tt, tg, yg)
    err = np.hypot(ex, ey)
    valid = (tt <= t_max + 1e-9) & (tt >= tg.min())
    if valid.any():
        out["pred_ade_m"] = float(err[valid].mean())
    if valid[:, -1].any():
        out["pred_fde_m"] = float(err[valid[:, -1], -1].mean())
    return out


# --------------------------------------------------------------------------- per episode
def episode_metrics(df: pd.DataFrame, cfg: dict) -> dict:
    """Compute the metrics of one episode from its per-tick log (see module docstring).

    Args:
        df: DataFrame with (a subset of) LOG_COLUMNS, one row per tick, sorted by sim_time_s.
        cfg: full config (navigation.goal_tolerance_m, evaluation.*, prediction.horizon_s/step_s).
    Returns:
        dict of floats / bools / ints (NaN where not applicable).
    """
    nav, ev, pred = cfg["navigation"], cfg.get("evaluation", {}), cfg.get("prediction", {})
    tol = float(nav["goal_tolerance_m"])
    horizon = float(pred.get("horizon_s", 2.5))
    step = float(pred.get("step_s", 0.1))
    min_contact_s = float(ev.get("collision_min_duration_s", 0.0))
    fs_clear = float(ev.get("false_stop_clearance_m", 1.5))

    n = len(df)
    m: dict = {"num_ticks": n}
    if n == 0:
        m.update({"goal_reached": False, "collision": False, "fall": False,
                  "collision_free_success": False, "collision_events": 0,
                  "min_clearance_m": float("nan"), "time_to_goal_s": float("nan"),
                  "final_goal_distance_m": float("nan"), "yield_count": 0, "total_yield_s": 0.0,
                  "slow_count": 0, "total_slow_s": 0.0, "false_stop_count": 0,
                  "sensor_failure_ticks": 0, "episode_duration_s": 0.0,
                  "mean_sensor_latency_ms": float("nan"), "p95_sensor_latency_ms": float("nan"),
                  "mean_planner_latency_ms": float("nan"), "p95_planner_latency_ms": float("nan"),
                  "track_pos_error_m": float("nan"), "pred_ade_m": float("nan"),
                  "pred_fde_m": float("nan")})
        return m

    t = _num(df, "sim_time_s")
    dur = _row_durations(t)
    gdist = _num(df, "goal_distance_m")
    state = df["nav_state"].astype(str).to_numpy() if "nav_state" in df.columns else np.array([""] * n)

    # goal
    final_d = float(gdist[-1]) if np.isfinite(gdist[-1]) else float("nan")
    goal_reached = bool(np.isfinite(final_d) and final_d <= tol)
    reached_idx = np.flatnonzero(np.isfinite(gdist) & (gdist <= tol))
    time_to_goal = float(t[reached_idx[0]] - t[0]) if (goal_reached and len(reached_idx)) else float("nan")

    # collision events
    contact = np.nan_to_num(_num(df, "contact_flag"), nan=0.0) > 0.5
    events = 0
    for a, b in _runs(contact):
        if (t[b] - t[a]) + dur[b] >= min_contact_s - 1e-12 or min_contact_s <= 0.0:
            events += 1
    collision = events > 0
    fall = bool((np.nan_to_num(_num(df, "fall_flag"), nan=0.0) > 0.5).any())

    # clearance (circle approximation, GT eval only)
    clr = _num(df, "clearance_gt_m")
    min_clear = _nan_stat(clr, np.min)

    # yield / slow
    is_yield = state == "YIELD"
    is_slow = state == "SLOW"
    yield_runs = _runs(is_yield)
    clr_inf = np.where(np.isfinite(clr), clr, np.inf)
    false_stops = 0
    for a, _b in yield_runs:
        win = (t >= t[a] - 1e-9) & (t <= t[a] + horizon + 1e-9)
        if np.min(clr_inf[win]) > fs_clear:
            false_stops += 1

    sl, pl = _num(df, "sensor_latency_ms"), _num(df, "planner_latency_ms")
    m.update({
        "goal_reached": goal_reached,
        "collision": collision,
        "fall": fall,
        "collision_free_success": bool(goal_reached and not collision and not fall),
        "collision_events": events,
        "min_clearance_m": min_clear,
        "time_to_goal_s": time_to_goal,
        "final_goal_distance_m": final_d,
        "yield_count": len(yield_runs),
        "total_yield_s": float(dur[is_yield].sum()),
        "slow_count": len(_runs(is_slow)),
        "total_slow_s": float(dur[is_slow].sum()),
        "false_stop_count": false_stops,
        "sensor_failure_ticks": int((state == "STOP_SENSOR_FAILURE").sum()),
        "episode_duration_s": float(t[-1] - t[0]) if np.isfinite(t[[0, -1]]).all() else float("nan"),
        "mean_sensor_latency_ms": _nan_stat(sl, np.mean),
        "p95_sensor_latency_ms": _nan_stat(sl, lambda x: np.percentile(x, 95)),
        "mean_planner_latency_ms": _nan_stat(pl, np.mean),
        "p95_planner_latency_ms": _nan_stat(pl, lambda x: np.percentile(x, 95)),
    })
    m.update(prediction_errors(df, horizon, step))
    return m


# --------------------------------------------------------------------------- summary
SUMMARY_COLUMNS = [
    "scenario", "mode", "episodes", "collision_free_success", "collision_rate",
    "mean_min_clearance_m", "mean_time_to_goal_s",
    "goal_rate", "fall_rate", "mean_yield_count", "mean_total_yield_s", "mean_false_stop_count",
    "mean_track_pos_error_m", "mean_pred_ade_m", "mean_pred_fde_m", "errors",
]


def _is_error_row(df: pd.DataFrame) -> pd.Series:
    if "error" not in df.columns:
        return pd.Series(False, index=df.index)
    e = df["error"]
    return e.notna() & (e.astype(str).str.strip() != "") & (e.astype(str) != "nan")


def _rate(s: pd.Series) -> float:
    s = s.dropna()
    if s.dtype == object:
        s = s.astype(str).str.lower().map({"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0})
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.mean()) if len(s) else float("nan")


def _mean(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return float("nan")
    s = pd.to_numeric(df[col], errors="coerce")
    s = s[np.isfinite(s)]
    return float(s.mean()) if len(s) else float("nan")


def summarize(episodes: pd.DataFrame) -> pd.DataFrame:
    """Group episode rows by (scenario, mode) into the README section-8 table (+ extras).

    `episodes` columns: scenario, mode, and the keys of `episode_metrics`; an optional `error`
    column marks episodes whose simulation raised — those are excluded from every statistic and
    only counted in `errors`. Rates are fractions in [0, 1]. mean_time_to_goal_s averages over
    episodes that reached the goal only. Group order follows first appearance.
    """
    rows = []
    if episodes is None or len(episodes) == 0:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    err = _is_error_row(episodes)
    for (sc, mode), g in episodes.groupby(["scenario", "mode"], sort=False):
        ok = g[~err.loc[g.index]]
        rows.append({
            "scenario": sc, "mode": mode, "episodes": int(len(ok)),
            "collision_free_success": _rate(ok["collision_free_success"]) if "collision_free_success" in ok else float("nan"),
            "collision_rate": _rate(ok["collision"]) if "collision" in ok else float("nan"),
            "mean_min_clearance_m": _mean(ok, "min_clearance_m"),
            "mean_time_to_goal_s": _mean(ok, "time_to_goal_s"),
            "goal_rate": _rate(ok["goal_reached"]) if "goal_reached" in ok else float("nan"),
            "fall_rate": _rate(ok["fall"]) if "fall" in ok else float("nan"),
            "mean_yield_count": _mean(ok, "yield_count"),
            "mean_total_yield_s": _mean(ok, "total_yield_s"),
            "mean_false_stop_count": _mean(ok, "false_stop_count"),
            "mean_track_pos_error_m": _mean(ok, "track_pos_error_m"),
            "mean_pred_ade_m": _mean(ok, "pred_ade_m"),
            "mean_pred_fde_m": _mean(ok, "pred_fde_m"),
            "errors": int(err.loc[g.index].sum()),
        })
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def summary_to_markdown(summary: pd.DataFrame, columns: Iterable[str] | None = None) -> str:
    """Render a summary DataFrame as a GitHub markdown table (NaN -> 'n/a', rates as %)."""
    cols = list(columns) if columns is not None else list(summary.columns)
    pct = {"collision_free_success", "collision_rate", "goal_rate", "fall_rate"}

    def cell(c, v):
        if isinstance(v, (float, np.floating)):
            if not math.isfinite(v):
                return "n/a"
            return f"{100 * v:.0f}%" if c in pct else f"{v:.2f}"
        return str(v)

    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" if c in ("scenario", "mode") else "---:" for c in cols) + "|"]
    for _, r in summary.iterrows():
        lines.append("| " + " | ".join(cell(c, r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"
