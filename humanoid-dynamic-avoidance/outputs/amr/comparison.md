# Evaluation results (amr)

Generated 2026-10-16 16:08 by `scripts/evaluate.py` from simulation logs (config: `configs/default.yaml`, seeds: [0, 1, 2, 3, 4]). Failed episodes: 0 (excluded from statistics, see `episodes.csv`).

Rates are over completed episodes. Min clearance uses a circle approximation of robot and actor with ground truth (evaluation only). Time-to-goal averages reached episodes only.

| scenario | mode | episodes | collision_free_success | collision_rate | mean_min_clearance_m | mean_time_to_goal_s |
|---|---|---:|---:|---:|---:|---:|
| empty | reactive | 5 | 100% | 0% | n/a | 15.80 |
| empty | predictive | 5 | 100% | 0% | n/a | 15.80 |
| crossing_slow | reactive | 5 | 0% | 100% | -0.01 | 20.34 |
| crossing_slow | predictive | 5 | 100% | 0% | 0.37 | 20.76 |
| crossing_fast | reactive | 5 | 0% | 100% | -0.02 | 19.02 |
| crossing_fast | predictive | 5 | 100% | 0% | 0.22 | 19.58 |
| late_crossing | reactive | 5 | 0% | 100% | -0.03 | 19.00 |
| late_crossing | predictive | 5 | 100% | 0% | 0.12 | 19.40 |
| brief_occlusion | reactive | 5 | 0% | 100% | -0.01 | 20.34 |
| brief_occlusion | predictive | 5 | 100% | 0% | 0.66 | 21.86 |
| sudden_turn_early | reactive | 5 | 0% | 100% | -0.01 | 22.24 |
| sudden_turn_early | predictive | 5 | 100% | 0% | 0.23 | 22.94 |
| sudden_turn | reactive | 5 | 0% | 100% | -0.05 | 26.40 |
| sudden_turn | predictive | 5 | 0% | 100% | -0.04 | 24.80 |


## Additional metrics

false stop = YIELD entry after which GT clearance stays > 1.5 m for the next 3.0 s (see evaluation/metrics.py).

| scenario | mode | goal_rate | fall_rate | mean_yield_count | mean_total_yield_s | mean_false_stop_count | mean_track_pos_error_m | mean_pred_ade_m | mean_pred_fde_m | errors |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| empty | reactive | 100% | 0% | 0.00 | 0.00 | 0.00 | n/a | n/a | n/a | 0 |
| empty | predictive | 100% | 0% | 0.00 | 0.00 | 0.00 | n/a | n/a | n/a | 0 |
| crossing_slow | reactive | 100% | 0% | 1.00 | 3.80 | 0.00 | 0.21 | 0.33 | 0.56 | 0 |
| crossing_slow | predictive | 100% | 0% | 1.00 | 3.30 | 0.00 | 0.21 | 0.31 | 0.51 | 0 |
| crossing_fast | reactive | 100% | 0% | 1.00 | 2.38 | 0.00 | 0.20 | 0.45 | 0.86 | 0 |
| crossing_fast | predictive | 100% | 0% | 1.00 | 3.04 | 0.00 | 0.20 | 0.44 | 0.84 | 0 |
| late_crossing | reactive | 100% | 0% | 1.00 | 2.24 | 0.00 | 0.20 | 0.48 | 0.93 | 0 |
| late_crossing | predictive | 100% | 0% | 1.00 | 2.90 | 0.00 | 0.20 | 0.47 | 0.91 | 0 |
| brief_occlusion | reactive | 100% | 0% | 1.00 | 3.80 | 0.00 | 0.71 | 0.79 | 0.92 | 0 |
| brief_occlusion | predictive | 100% | 0% | 1.00 | 2.78 | 0.00 | 1.09 | 1.22 | 1.35 | 0 |
| sudden_turn_early | reactive | 100% | 0% | 1.20 | 5.66 | 0.00 | 0.21 | 0.51 | 1.02 | 0 |
| sudden_turn_early | predictive | 100% | 0% | 2.00 | 4.94 | 0.00 | 0.21 | 0.50 | 1.01 | 0 |
| sudden_turn | reactive | 100% | 0% | 2.00 | 6.78 | 0.00 | 0.21 | 0.56 | 1.15 | 0 |
| sudden_turn | predictive | 100% | 0% | 2.00 | 5.90 | 0.00 | 0.20 | 0.54 | 1.10 | 0 |
