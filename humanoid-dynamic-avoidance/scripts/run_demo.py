"""Run one demo episode.

Examples:
  python scripts/run_demo.py --scenario crossing_slow --mode predictive
  python scripts/run_demo.py --robot g1 --scenario crossing_slow --mode reactive --record
  python scripts/run_demo.py --scenario interactive --mode predictive     # drag the actor in the viewer
  python scripts/run_demo.py --scenario empty --mode manual --headless
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humanoid_avoidance.config import OUTPUT_DIR, load_config, load_scenarios  # noqa: E402
from humanoid_avoidance.sim.runner import run_episode  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", default=None, choices=["amr", "g1"], help="default: simulation.robot in config")
    ap.add_argument("--scenario", default="crossing_slow", choices=sorted(load_scenarios()))
    ap.add_argument("--mode", default="predictive", choices=["manual", "reactive", "predictive"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default=None)
    ap.add_argument("--headless", action="store_true", help="no viewer window (runs faster than realtime)")
    ap.add_argument("--no-realtime", action="store_true", help="with viewer: do not pace to wall clock")
    ap.add_argument("--record", nargs="?", const="auto", default=None,
                    help="write MP4 (offscreen renderer). Optional path; default outputs/videos/<...>.mp4")
    ap.add_argument("--timeout", type=float, default=None, help="override episode_timeout_s")
    args = ap.parse_args()

    overrides = {}
    if args.timeout:
        overrides["simulation"] = {"episode_timeout_s": args.timeout}
    if args.scenario == "interactive":
        overrides.setdefault("simulation", {})["episode_timeout_s"] = args.timeout or 600.0
    cfg = load_config(args.config, overrides or None)
    robot = args.robot or cfg["simulation"]["robot"]

    record = None
    if args.record:
        record = (Path(args.record) if args.record != "auto"
                  else OUTPUT_DIR / "videos" / f"{robot}_{args.scenario}_{args.mode}_s{args.seed}.mp4")

    m = run_episode(cfg, args.scenario, args.mode, args.seed, robot_name=robot,
                    headless=args.headless, record_path=record, realtime=not args.no_realtime)

    keys = ["robot", "scenario", "mode", "seed", "goal_reached", "collision", "fall",
            "collision_free_success", "min_clearance_m", "time_to_goal_s", "yield_count",
            "total_yield_s", "false_stop_count", "sim_time_s", "sim_speed_factor", "csv_path", "video_path"]
    print("\n=== episode result ===")
    for k in keys:
        if k in m:
            v = m[k]
            print(f"{k:24s} {v:.3f}" if isinstance(v, float) else f"{k:24s} {v}")

    try:
        from humanoid_avoidance.evaluation.plots import plot_episode_topdown
        png = Path(m["csv_path"]).with_name("topdown.png")
        plot_episode_topdown(m["csv_path"], png, cfg)
        print(f"{'topdown_png':24s} {png}")
    except Exception as e:  # plotting must never hide the episode result
        print(f"(plot skipped: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
