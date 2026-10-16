# Architecture

## Pipeline (one simulation loop, one clock = `data.time`)

```text
ScenarioController ──(GT, writes mocap_pos only)──► MuJoCo physics (dt 0.002 s)
                                                        │
          RobotAdapter.pre_step  ◄── NavCommand ──┐     │ every physics step
     (G1: pretrained policy @50 Hz + PD @500 Hz;  │     │
      AMR: velocity actuators)                    │     ▼
                                                  │  RangeSensor (10 Hz, 61 rays, 180°, mj_ray,
                                                  │  only perceivable geom group, noise + dropout)
                                                  │     │ Scan (sensor frame, timestamp)
                                                  │     ▼
                                                  │  sensor → robot → world  (geometry/transforms)
                                                  │  map-wall filter + distance-gap clustering
                                                  │     │ Detection (visible-surface centroid, radius, R)
                                                  │     ▼
                                                  │  MultiTracker: CV Kalman, Mahalanobis gate χ²(2)=9.21,
                                                  │  Hungarian assignment, tentative/confirmed/deleted
                                                  │     │ Track [x y vx vy], P (4×4)
                                                  │     ▼
                                                  │  Risk: TCA / dCA with uncertainty margin
                                                  │        m0 + k·sqrt(λmax(Pxy(τ*)))
                                                  │     ▼
                                                  └── NavController FSM (10 Hz)
                                                      GO / SLOW / YIELD / RESUME / STOP / GOAL_REACHED
```

Evaluation-only path: `ScenarioController.gt_actor_xy_from_sim` → CSV columns
`obstacle_gt_*_eval_only`, `clearance_gt_m`, read **after** the planner decided in that tick.

## Course pillars → code

| Pillar | Where | Evidence |
|---|---|---|
| Representation | `perception/clustering.py`: rays → points → clusters → Detection; `types.py` state `[x,y,vx,vy]` | detections in overlay / top-down plot |
| Geometry | `geometry/transforms.py`, sensor pose from adapter → world points | `tests/test_transforms.py` |
| Uncertainty | `tracking/kalman_cv.py` (Q, R, Joseph form, NIS), gating, covariance-based margin in `prediction/risk.py` | covariance discs in viewer, `track_P_*` columns |
| Time | sim-time `dt` in KF, sensor/planner cadence, stale-scan stop, hysteresis/ramp in FSM | `sensor_latency_ms`, `planner_latency_ms` columns |
| Evidence | paired reactive vs predictive runs (same seed, scene, controller, sensor), failure case | `outputs/comparison.csv`, figures |
| Learning | Unitree pretrained G1 locomotion policy (not trained by us) | `docs/unitree_integration.md` |
| Data | CSV per tick + metadata.json per run, seeded scenarios | `outputs/runs/*` |

## Why Kalman + prediction instead of distance-only

A distance-only (reactive) rule sees an obstacle only once it is *already in front* of the
robot. A person crossing from the side enters the robot's path laterally: by the time it is
inside the front box, the closing speed leaves too little room — or the person walks into the
robot's flank, which the front box never covers. The tracker estimates the actor velocity, so
the planner can compute *where the two will be closest* (TCA/dCA) and yield **before** the
conflict, then resume once the predicted path is clear. The covariance tells the planner how
much to trust the prediction (larger margin when the track is young or occluded).

Both modes share the same scene, sensor, heading controller and command limits; only the risk
logic differs, so the comparison isolates the effect of prediction.

## Honest assumptions

- Ego pose from the simulator (localization oracle). Not a localization claim.
- Detection = visible-surface centroid, not object center → biased toward the sensor by up to one radius; compensated by margin.
- Constant-velocity model: wrong during sudden turns (see failure case).
- Circle approximations for robot and actor; humanoid foot sweep not modelled.
- Research simulation, not a safety-certified system.
