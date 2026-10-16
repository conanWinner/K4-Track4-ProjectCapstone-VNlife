# Evaluation results (g1)

Generated 2026-10-16 16:13 by `scripts/evaluate.py` from simulation logs (config: `configs/default.yaml`, seeds: [0, 1, 2, 3, 4]). Failed episodes: 0 (excluded from statistics, see `episodes.csv`).

Rates are over completed episodes. Min clearance uses a circle approximation of robot and actor with ground truth (evaluation only). Time-to-goal averages reached episodes only.

| scenario | mode | episodes | collision_free_success | collision_rate | mean_min_clearance_m | mean_time_to_goal_s |
|---|---|---:|---:|---:|---:|---:|
| empty | reactive | 5 | 100% | 0% | n/a | 16.50 |
| empty | predictive | 5 | 100% | 0% | n/a | 16.50 |
| crossing_slow | reactive | 5 | 20% | 80% | -0.05 | 20.74 |
| crossing_slow | predictive | 5 | 100% | 0% | 0.36 | 21.38 |
| crossing_fast | reactive | 5 | 0% | 100% | -0.06 | 19.78 |
| crossing_fast | predictive | 5 | 100% | 0% | 0.20 | 20.18 |
| late_crossing | reactive | 5 | 0% | 100% | -0.06 | 20.22 |
| late_crossing | predictive | 5 | 100% | 0% | 0.11 | 19.98 |
| brief_occlusion | reactive | 5 | 20% | 80% | -0.05 | 20.74 |
| brief_occlusion | predictive | 5 | 100% | 0% | 0.73 | 31.94 |
| sudden_turn_early | reactive | 5 | 0% | 100% | -0.04 | 22.88 |
| sudden_turn_early | predictive | 5 | 100% | 0% | 0.15 | 23.56 |
| sudden_turn | reactive | 5 | 0% | 100% | -0.29 | n/a |
| sudden_turn | predictive | 5 | 0% | 100% | -0.21 | 29.30 |


## Additional metrics

false stop = YIELD entry after which GT clearance stays > 1.5 m for the next 3.0 s (see evaluation/metrics.py).

| scenario | mode | goal_rate | fall_rate | mean_yield_count | mean_total_yield_s | mean_false_stop_count | mean_track_pos_error_m | mean_pred_ade_m | mean_pred_fde_m | errors |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| empty | reactive | 100% | 0% | 0.00 | 0.00 | 0.00 | n/a | n/a | n/a | 0 |
| empty | predictive | 100% | 0% | 0.00 | 0.00 | 0.00 | n/a | n/a | n/a | 0 |
| crossing_slow | reactive | 100% | 0% | 1.00 | 3.76 | 0.00 | 0.21 | 0.32 | 0.54 | 0 |
| crossing_slow | predictive | 100% | 0% | 1.00 | 3.52 | 0.00 | 0.21 | 0.30 | 0.50 | 0 |
| crossing_fast | reactive | 100% | 0% | 1.00 | 2.44 | 0.00 | 0.20 | 0.45 | 0.84 | 0 |
| crossing_fast | predictive | 100% | 0% | 1.00 | 3.00 | 0.00 | 0.20 | 0.44 | 0.82 | 0 |
| late_crossing | reactive | 100% | 0% | 1.00 | 2.30 | 0.00 | 0.20 | 0.46 | 0.90 | 0 |
| late_crossing | predictive | 100% | 0% | 1.00 | 2.92 | 0.00 | 0.20 | 0.46 | 0.90 | 0 |
| brief_occlusion | reactive | 100% | 0% | 1.00 | 3.76 | 0.00 | 0.77 | 0.86 | 1.00 | 0 |
| brief_occlusion | predictive | 100% | 0% | 1.20 | 8.70 | 0.20 | 1.91 | 2.01 | 2.11 | 0 |
| sudden_turn_early | reactive | 100% | 0% | 2.00 | 5.64 | 0.00 | 0.21 | 0.51 | 1.02 | 0 |
| sudden_turn_early | predictive | 100% | 0% | 2.00 | 5.38 | 0.00 | 0.21 | 0.50 | 1.02 | 0 |
| sudden_turn | reactive | 0% | 100% | 2.00 | 5.18 | 0.00 | 0.21 | 0.50 | 0.90 | 0 |
| sudden_turn | predictive | 100% | 0% | 2.00 | 6.42 | 0.00 | 0.21 | 0.50 | 1.02 | 0 |
