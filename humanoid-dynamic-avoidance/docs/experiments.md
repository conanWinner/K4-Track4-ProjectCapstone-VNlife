# Experiments

All numbers come from `outputs/{amr,g1}/episodes.csv` produced by `scripts/evaluate.py` on 2026-10-16
(Windows 11, Python 3.11.7, MuJoCo 3.16.0, torch 2.6.0+cpu, unitree_rl_gym @ 276801e). Config:
`configs/default.yaml` + `configs/scenarios.yaml` at that state.

## Protocol
- Robots: `g1` (Unitree G1 12-DoF, pretrained LSTM policy at 50 Hz, PD at 500 Hz) and `amr`
  (cylinder proxy with velocity actuators). Same scene, sensor, tracker, heading controller, limits.
- Modes: `reactive` (latest detections in a 0.9 m × ±0.6 m front box → YIELD) vs `predictive`
  (confirmed Kalman tracks → TCA/dCA with capped uncertainty margin → SLOW/YIELD).
- Paired runs: same seed ⇒ same actor timing/speed jitter and same sensor noise stream for both modes.
- Seeds 0–4, 7 scenarios, 2 modes ⇒ 70 episodes per robot. Timeout 45 s.
- Metrics (see `evaluation/metrics.py`): collision = any robot–actor contact; collision-free success =
  goal reached ∧ no collision ∧ no fall; min clearance = circle approximation with GT (eval only).

## Results — G1

| scenario | mode | success | collision | fall | min clearance (m) | time to goal (s) |
|---|---|---:|---:|---:|---:|---:|
| empty | reactive | 100% | 0% | 0% | – | 16.50 |
| empty | predictive | 100% | 0% | 0% | – | 16.50 |
| crossing_slow | reactive | 20% | 80% | 0% | −0.05 | 20.74 |
| crossing_slow | predictive | **100%** | 0% | 0% | 0.36 | 21.38 |
| crossing_fast | reactive | 0% | 100% | 0% | −0.06 | 19.78 |
| crossing_fast | predictive | **100%** | 0% | 0% | 0.20 | 20.18 |
| late_crossing | reactive | 0% | 100% | 0% | −0.06 | 20.22 |
| late_crossing | predictive | **100%** | 0% | 0% | 0.11 | 19.98 |
| brief_occlusion | reactive | 20% | 80% | 0% | −0.05 | 20.74 |
| brief_occlusion | predictive | **100%** | 0% | 0% | 0.73 (0.57*) | 31.94 |
| sudden_turn_early | reactive | 0% | 100% | 0% | −0.04 | 22.88 |
| sudden_turn_early | predictive | **100%** | 0% | 0% | 0.15 | 23.56 |
| sudden_turn | reactive | 0% | 100% | 100% | −0.29 | – |
| sudden_turn | predictive | 0% | 100% | 0% | −0.21 | 29.30 |

\* Since commit `3ae782a` (static-obstacle routing), `clearance_gt_m` and the contact check also include
static boxes and the occluder. Re-running G1 `brief_occlusion` and `crossing_slow` (seeds 0–4) with that
code gives identical success/collision rates and time to goal; only the `brief_occlusion` predictive min
clearance changes, 0.73 → 0.57 m, because the robot now passes closer to the occluder than to the actor.

## Results — AMR proxy

| scenario | reactive success | predictive success | min clearance R → P (m) |
|---|---:|---:|---:|
| empty | 100% | 100% | – |
| crossing_slow | 0% | **100%** | −0.01 → 0.37 |
| crossing_fast | 0% | **100%** | −0.02 → 0.22 |
| late_crossing | 0% | **100%** | −0.03 → 0.12 |
| brief_occlusion | 0% | **100%** | −0.01 → 0.66 |
| sudden_turn_early | 0% | **100%** | −0.01 → 0.23 |
| sudden_turn | 0% | 0% | −0.05 → −0.04 |

## Static obstacles — `obstacle_course` (G1, predictive, added in `3ae782a`)
Three boxes, goal `[6.4, 0]`, no actor. `route_from_scan`: occupancy grid (0.2 m) built only from range
hits, each blocking cells within robot radius + 0.45 m; 8-connected A* to the goal; the controller follows the next
waypoint and stops (YIELD, "no free route") when A* finds nothing. Box poses are never given to the planner.

| seed | success | collision | min clearance (m) | time to goal (s) |
|---|---|---|---:|---:|
| 0 | yes | no | 0.56 | 38.4 |
| 1 | yes | no | 0.56 | 38.4 |
| 2 | yes | no | 0.56 | 38.4 |
| 3 | yes | no | 0.56 | 39.1 |
| 4 | **no** (timeout, 2.1 m from goal) | no | 0.36 | – |

Seed 4 spends 14.2 s in "no free route". Likely cause (not yet verified from the log): hits are kept
forever and each blocks every cell within 0.87 m (G1 radius 0.42 + 0.3 + 0.15), so a few noisy hits can
close a gap between boxes. Known limits: static scenes only
(`route_from_scan` with an actor raises), no map decay, unseen space treated as free.

## Observations
1. Prediction is what makes the difference for side-crossing actors: reactive only reacts once the actor
   is inside the front box, i.e. when the actor is ~0.6 m from the robot path, too late for a lateral
   approach. Predictive yields 1–2 s earlier, when TCA ≤ 2 s and dCA < required clearance.
2. Cost is small: +0.4–0.7 s time-to-goal in crossing scenarios; no false stops except one
   in G1 `brief_occlusion` (0.2 per episode on average).
3. Failure case reproduced deterministically (5/5): late reversal beats the CV model's observation delay.
4. Margin model: the spec formula `F P Fᵀ + Q(τ)` at τ = horizon gives σ ≈ 2 m (σ_a = 0.7 m/s², 2.5–3 s),
   which made a parked actor 1.4 m beside the path look dangerous (robot never reached the goal). Capping
   the uncertainty part at 0.35 m fixed it — documented as a heuristic, not a probabilistic guarantee.
5. G1 specifics needed for parity with the AMR: velocity EMA (gait sway), 0.42 m footprint, YIELD
   station keeping (policy drifts at zero command).

## Integration log (what changed and why, with evidence)
| Problem seen in logs | Fix |
|---|---|
| Robot yields forever next to a parked actor (margin 4.8 m) | `max_uncertainty_margin_m: 0.35` |
| Predictive success but clearance 0.07 m: actor starts 1.5 m from path, robot has < 2 s | corridor ±2.5 m, actor parks at ±2.2 m, scenario re-timed |
| `sudden_turn` actor stops in the path → no one can pass | actor leaves after the turn; hard version made the real failure case |
| SLOW never triggered (tca clipped at horizon = yield threshold) | horizon 3.0 s, slow ≤ 3.0 s, yield ≤ 2.0 s |
| G1 collisions in `sudden_turn_early` (arms outside 0.35 m) | G1 radius 0.42 m |
| G1 stuck beside occluder (drift + sway flips "approaching") | velocity EMA + YIELD station keeping |
| Videos: walls and actor invisible | enable geom group 3 in render/viewer options |

## Reproduce
```powershell
.venv\Scripts\python.exe scripts\evaluate.py --robot g1 --scenarios empty crossing_slow crossing_fast late_crossing brief_occlusion sudden_turn_early sudden_turn --modes reactive predictive --seeds 0 1 2 3 4 --out outputs\g1
.venv\Scripts\python.exe scripts\evaluate.py --robot amr --scenarios empty crossing_slow crossing_fast late_crossing brief_occlusion sudden_turn_early sudden_turn --modes reactive predictive --seeds 0 1 2 3 4 --out outputs\amr
```
G1: ~285 s wall for 70 episodes; AMR: ~85 s.
