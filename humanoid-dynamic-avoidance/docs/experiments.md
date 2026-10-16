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
| brief_occlusion | predictive | **100%** | 0% | 0% | 0.73 | 31.94 |
| sudden_turn_early | reactive | 0% | 100% | 0% | −0.04 | 22.88 |
| sudden_turn_early | predictive | **100%** | 0% | 0% | 0.15 | 23.56 |
| sudden_turn | reactive | 0% | 100% | 100% | −0.29 | – |
| sudden_turn | predictive | 0% | 100% | 0% | −0.21 | 29.30 |

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
