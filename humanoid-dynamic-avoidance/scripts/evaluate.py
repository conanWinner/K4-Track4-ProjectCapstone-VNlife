"""Batch evaluation: every scenario x mode x seed through sim.runner.run_episode.

Outputs (under --out, default outputs/):
  episodes.csv      one row per episode (metrics from the simulation logs; failed runs keep an
                    `error` message and no metrics — nothing is filled in or invented)
  comparison.csv    summarize(episodes) grouped by scenario, mode
  comparison.md     markdown version of the README section-8 table + extras
  figures/          comparison.png, per-episode top-down and prediction-error plots
  runs/             per-episode CSV logs (written by run_episode)

Example:
  .venv\\Scripts\\python.exe scripts/evaluate.py --scenarios crossing_slow crossing_fast sudden_turn \\
      --modes reactive predictive --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from humanoid_avoidance.config import PROJECT_ROOT, load_config  # noqa: E402
from humanoid_avoidance.evaluation.metrics import (episode_metrics, summarize,  # noqa: E402
                                                   summary_to_markdown)

README_COLUMNS = ["scenario", "mode", "episodes", "collision_free_success", "collision_rate",
                  "mean_min_clearance_m", "mean_time_to_goal_s"]
EXTRA_COLUMNS = ["scenario", "mode", "goal_rate", "fall_rate", "mean_yield_count",
                 "mean_total_yield_s", "mean_false_stop_count", "mean_track_pos_error_m",
                 "mean_pred_ade_m", "mean_pred_fde_m", "errors"]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", choices=["amr", "g1"], default=None,
                   help="robot adapter (default: simulation.robot from the config)")
    p.add_argument("--scenarios", nargs="+", default=["crossing_slow", "crossing_fast", "sudden_turn"])
    p.add_argument("--modes", nargs="+", default=["reactive", "predictive"],
                   choices=["reactive", "predictive", "manual"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs")
    p.add_argument("--config", type=Path, default=None, help="config yaml (default configs/default.yaml)")
    p.add_argument("--plot-all", action="store_true",
                   help="per-episode figures for every seed (default: first seed only)")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


def _metrics_from_result(result: dict, cfg: dict) -> dict:
    """run_episode already returns metrics; recompute from the CSV only if they are missing."""
    if "collision_free_success" in result or not result.get("csv_path"):
        return dict(result)
    df = pd.read_csv(result["csv_path"])
    out = episode_metrics(df, cfg)
    out.update(result)
    return out


def _f(v, nd: int = 2) -> str:
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    robot = args.robot or cfg["simulation"].get("robot", "amr")
    out = Path(args.out)
    if not out.is_absolute():
        out = Path.cwd() / out
    fig_dir, run_dir = out / "figures", out / "runs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    from humanoid_avoidance.sim.runner import run_episode  # heavy import (mujoco) only here

    combos = [(s, m, seed) for s in args.scenarios for m in args.modes for seed in args.seeds]
    print(f"[evaluate] robot={robot} episodes={len(combos)} out={out}")
    rows = []
    t_start = time.perf_counter()
    for i, (scenario, mode, seed) in enumerate(combos, 1):
        tag = f"[{i}/{len(combos)}] {scenario:<16} {mode:<10} seed={seed}"
        t0 = time.perf_counter()
        try:
            res = run_episode(cfg, scenario, mode, seed, robot_name=robot, headless=True,
                              run_dir=run_dir)
            row = _metrics_from_result(res, cfg)
            row.update({"scenario": scenario, "mode": mode, "seed": seed, "robot": robot, "error": ""})
        except Exception as exc:  # record and continue — never fabricate a result
            traceback.print_exc()
            row = {"scenario": scenario, "mode": mode, "seed": seed, "robot": robot,
                   "error": f"{type(exc).__name__}: {exc}"}
            print(f"{tag}  ERROR {row['error']}")
        else:
            print(f"{tag}  success={row.get('collision_free_success')} collision={row.get('collision')} "
                  f"min_clear={_f(row.get('min_clearance_m'))} m ttg={_f(row.get('time_to_goal_s'))} s "
                  f"yields={row.get('yield_count')} ({time.perf_counter() - t0:.1f} s wall)")
        rows.append(row)

    episodes = pd.DataFrame(rows)
    lead = ["scenario", "mode", "seed", "robot", "error"]
    episodes = episodes[lead + [c for c in episodes.columns if c not in lead]]
    episodes.to_csv(out / "episodes.csv", index=False)
    summary = summarize(episodes)
    summary.to_csv(out / "comparison.csv", index=False)

    n_err = int((episodes["error"].astype(str) != "").sum())
    md = [f"# Evaluation results ({robot})", "",
          f"Generated {datetime.now():%Y-%m-%d %H:%M} by `scripts/evaluate.py` from simulation logs "
          f"(config: `{args.config or 'configs/default.yaml'}`, seeds: {args.seeds}). "
          f"Failed episodes: {n_err} (excluded from statistics, see `episodes.csv`).", "",
          "Rates are over completed episodes. Min clearance uses a circle approximation of robot and "
          "actor with ground truth (evaluation only). Time-to-goal averages reached episodes only.", "",
          summary_to_markdown(summary, README_COLUMNS), "",
          "## Additional metrics", "",
          "false stop = YIELD entry after which GT clearance stays > "
          f"{cfg.get('evaluation', {}).get('false_stop_clearance_m')} m for the next "
          f"{cfg['prediction']['horizon_s']} s (see evaluation/metrics.py).", "",
          summary_to_markdown(summary, EXTRA_COLUMNS)]
    (out / "comparison.md").write_text("\n".join(md), encoding="utf-8")

    if not args.no_plots and len(summary):
        from humanoid_avoidance.evaluation.plots import (plot_comparison, plot_episode_topdown,
                                                         plot_prediction_error)
        try:
            plot_comparison(summary, fig_dir / "comparison.png")
        except Exception:
            traceback.print_exc()
        first_seed = args.seeds[0]
        for row in rows:
            csv_path = row.get("csv_path")
            if row.get("error") or not csv_path or not Path(csv_path).exists():
                continue
            if not args.plot_all and row["seed"] != first_seed:
                continue
            stem = f"{row['scenario']}_{row['mode']}_s{row['seed']}"
            try:
                plot_episode_topdown(csv_path, fig_dir / f"topdown_{stem}.png", cfg)
                if row["mode"] == "predictive":
                    plot_prediction_error(csv_path, fig_dir / f"prediction_error_{stem}.png")
            except Exception:
                traceback.print_exc()

    print()
    print(summary_to_markdown(summary, README_COLUMNS))
    print(f"[evaluate] wrote {out / 'episodes.csv'}, {out / 'comparison.csv'}, {out / 'comparison.md'}, "
          f"{fig_dir} ({time.perf_counter() - t_start:.1f} s total, {n_err} failed)")
    return 1 if n_err == len(rows) else 0


if __name__ == "__main__":
    sys.exit(main())
