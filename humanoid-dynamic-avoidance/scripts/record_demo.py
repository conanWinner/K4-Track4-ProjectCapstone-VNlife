"""Record the backup demo videos (offscreen MuJoCo renderer, real simulation — no fake animation).

Produces outputs/videos/*.mp4 for: reactive vs predictive on the same crossing scenario + seed,
and the sudden_turn failure case; then concatenates them into outputs/demo_backup.mp4.

  python scripts/record_demo.py --robot g1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humanoid_avoidance.config import OUTPUT_DIR, load_config  # noqa: E402
from humanoid_avoidance.sim.runner import run_episode  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default=None, choices=["amr", "g1"])
    ap.add_argument("--scenario", default="crossing_slow")
    ap.add_argument("--failure", default="sudden_turn")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    robot = args.robot or cfg["simulation"]["robot"]
    vid_dir = OUTPUT_DIR / "videos"

    clips = [(args.scenario, "reactive"), (args.scenario, "predictive"), (args.failure, "predictive")]
    paths = []
    for scen, mode in clips:
        p = vid_dir / f"{robot}_{scen}_{mode}_s{args.seed}.mp4"
        m = run_episode(cfg, scen, mode, args.seed, robot_name=robot, headless=True, record_path=p)
        print(f"{scen:15s} {mode:10s} success={m.get('collision_free_success')} "
              f"collision={m.get('collision')} min_clear={m.get('min_clearance_m'):.2f} -> {p}")
        paths.append(p)

    import imageio.v2 as imageio
    out = OUTPUT_DIR / f"demo_backup_{robot}.mp4"
    with imageio.get_writer(str(out), fps=30, codec="libx264", quality=7, macro_block_size=8) as w:
        for p in paths:
            for frame in imageio.get_reader(str(p)):
                w.append_data(frame)
    print(f"backup video: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
