# Module interfaces (contract between parallel workstreams)

Package root: `src/humanoid_avoidance/` (import as `humanoid_avoidance`). Shared types in
`types.py`, config in `config.py` (`load_config()`, `load_scenarios()`), all thresholds in
`configs/default.yaml` and `configs/scenarios.yaml`. Python interpreter: `.venv\Scripts\python.exe`
(venv with system site-packages: numpy, scipy, torch, matplotlib, pandas, pytest, yaml + mujoco).

Hard rules (from the project README):
- Perception / Tracker / Planner never receive ground-truth actor pose. GT lives only in
  `sim/scenario_controller.py` and `evaluation/`.
- Time = simulation time (`data.time`). Wall clock only for latency profiling.
- G1 is moved only by the pretrained policy through joint PD targets. `qpos` touched only at reset.
- No hard-coded thresholds: read from the cfg dict passed in.
- Every public function documented with units and shapes. No fabricated metrics.

## Ownership (one writer per file)

| Workstream | Files |
|---|---|
| **A. G1 integration** | `sim/g1_adapter.py`, `scripts/setup.ps1`, `scripts/setup.sh`, `configs/upstream_version.txt`, `docs/unitree_integration.md`, `tests/test_g1_adapter.py`, `third_party/` |
| **B. Estimation core** | `geometry/transforms.py`, `tracking/kalman_cv.py`, `tracking/multi_tracker.py`, `prediction/risk.py`, `tests/test_transforms.py`, `tests/test_kalman.py`, `tests/test_association.py`, `tests/test_tca.py` |
| **C. Navigation + evaluation** | `navigation/controller.py`, `evaluation/metrics.py`, `evaluation/plots.py`, `logging/csv_logger.py`, `scripts/evaluate.py`, `tests/test_planner.py`, `tests/test_metrics.py` |
| **D. Sim scene + perception** | `sim/scene_builder.py`, `sim/amr_adapter.py`, `sim/scenario_controller.py`, `sim/robot_base.py`, `perception/range_sensor.py`, `perception/clustering.py`, `tests/test_range_sensor.py`, `tests/test_clustering.py`, `tests/test_scenario.py` |
| **Lead (integration)** | `types.py`, `config.py`, `configs/*`, `sim/runner.py`, `sim/overlay.py`, `scripts/run_demo.py`, `scripts/record_demo.py`, `tests/test_integration_smoke.py`, `README.md`, `docs/architecture.md`, `docs/experiments.md`, `docs/known_limitations.md` |

If you need a change in a file you do not own, do not edit it — state the needed change in your report.

## B. Estimation core

### `geometry/transforms.py`
```python
wrap_angle(a: float | np.ndarray) -> same          # to [-pi, pi)
rot2d(yaw: float) -> np.ndarray (2,2)
se2(x: float, y: float, yaw: float) -> np.ndarray (3,3)     # homogeneous, maps local -> parent
se2_inverse(T: (3,3)) -> (3,3)
transform_points_2d(T: (3,3), pts: (N,2)) -> (N,2)
pose3(pos: (3,), R: (3,3)) -> (4,4)
transform_points_3d(T: (4,4), pts: (N,3)) -> (N,3)
quat_wxyz_to_rotmat(q: (4,)) -> (3,3)              # MuJoCo convention w,x,y,z
yaw_from_rotmat(R: (3,3)) -> float
quat_wxyz_to_yaw(q: (4,)) -> float
body_to_world_vel(v_body: (2,), yaw) -> (2,);  world_to_body_vel(v_world: (2,), yaw) -> (2,)
polar_to_points(ranges: (N,), angles: (N,)) -> (N,2)  # sensor-frame XY
```

### `tracking/kalman_cv.py` (functional, constant-velocity, piecewise-constant white accel)
```python
F_cv(dt) -> (4,4);  H_CV -> (2,4) constant
Q_cv(dt, sigma_a) -> (4,4)       # sigma_a^2 * [[dt^4/4, dt^3/2], [dt^3/2, dt^2]] per axis
kf_predict(x (4,), P (4,4), dt, sigma_a) -> (x, P)          # dt >= 0, dt==0 is a no-op
kf_update(x, P, z (2,), R (2,2)) -> (x, P, nis: float)      # Joseph form, P symmetrized
mahalanobis_sq(x, P, z, R) -> float                          # (z-Hx)^T S^-1 (z-Hx)
init_state(z (2,), R (2,2), init_velocity_sigma) -> (x, P)  # v = 0, large velocity covariance
```

### `tracking/multi_tracker.py`
```python
class MultiTracker:
    def __init__(self, cfg_tracking: dict): ...
    def step(self, time_s: float, detections: list[Detection]) -> list[Track]:
        """Predict all tracks to time_s (sim time, monotonic; raise ValueError if time goes back),
        gate with Mahalanobis (cfg mahalanobis_gate_sq), assign with linear_sum_assignment on the
        gated cost matrix (no forced assignment outside gate), update matched, spawn tentative tracks
        for unmatched detections, lifecycle tentative -> confirmed (min_confirmations within
        confirmation_window) -> deleted after max_misses consecutive misses.
        Call with [] to predict-only (missed update). Returns ALL live tracks (copies)."""
    def confirmed_tracks(self) -> list[Track]
    def reset(self) -> None
```
Track.radius_m: exponential average of detection radii.

### `prediction/risk.py`
```python
time_of_closest_approach(q: (2,), w: (2,), horizon_s, eps=1e-6) -> (tau_s, dca_m)
    # q = p_obs - p_robot, w = v_obs - v_robot; tau = clip(-(q.w)/(w.w+eps), 0, horizon)
predict_track(track: Track, horizon_s, step_s, sigma_a) -> (times (K,), means (K,2), covs (K,2,2))
assess_track_risk(robot: RobotState, track: Track, robot_radius_m, cfg_prediction, sigma_a) -> RiskAssessment
    # margin = m0 + k*sqrt(lambda_max(Pxy(tau*))); approaching = (q.w < -eps*|q|) or currently overlapping;
    # risk = approaching and dca < r_robot + r_obs + margin (or currently overlapping)
    # score in [0,1]: e.g. clip(required_clearance / max(dca, 1e-3), 0, 1) * (1 - tau/horizon) if approaching
assess_risk(robot, tracks: list[Track], robot_radius_m, cfg_prediction, sigma_a) -> RiskAssessment
    # worst over confirmed tracks (risk first, then smallest tau); NO_RISK if none
```

## C. Navigation + evaluation

### `navigation/controller.py`
```python
class NavController:
    def __init__(self, cfg: dict, mode: str):        # full cfg dict; mode in {"reactive","predictive","manual"}
    def reset(self) -> None
    def step(self, time_s: float, robot: RobotState, goal_xy: np.ndarray,
             tracks: list[Track], detections: list[Detection],
             last_scan_time_s: float | None) -> NavCommand
    debug: dict   # after each step: {"state","risk","tca_s","dca_m","score","margin_m","reason"}
```
- FSM states: GO, SLOW, YIELD, RESUME, STOP_SENSOR_FAILURE, GOAL_REACHED (MANUAL in manual mode:
  constant forward command toward goal, no obstacle logic — used to verify locomotion).
- `reactive`: ONLY latest `detections` positions; YIELD if any detection center in body frame has
  0 < x < reactive.stop_distance_m and |y| < reactive.corridor_half_width_m. No velocity/TCA.
- `predictive`: uses `prediction.risk.assess_risk(robot, confirmed tracks, ...)`. risk and
  tca <= yield_tca_s -> YIELD; risk and tca <= slow_tca_s -> SLOW.
- Both: stale scan (time_s - last_scan_time_s > stale_scan_timeout_s, or None after 1 s) ->
  STOP_SENSOR_FAILURE; goal within tolerance -> GOAL_REACHED (zero command).
- RESUME: risk-free continuously for safe_resume_duration_s, then ramp vx over resume_ramp_s -> GO.
- Heading: yaw_rate = clip(k_yaw*wrap(heading_goal - yaw)); reduce vx when |err| > heading_slowdown_rad.
- Slew-limit vx (accel_limit_mps2) and yaw_rate (yaw_accel_limit_rps2) using sim time dt.
- Clip vx to min(navigation.max_vx_mps, robot limit passed via cfg["navigation"]["max_vx_mps"]).

### `logging/csv_logger.py`
```python
LOG_COLUMNS: list[str]   # exactly the per-tick columns listed below
class CsvLogger:
    def __init__(self, path: Path, metadata: dict): ...   # writes metadata.json next to CSV
    def log(self, row: dict) -> None                      # missing keys -> "" ; extra keys ignored
    def close(self) -> None
```
Columns: `sim_time_s, scenario, seed, mode, robot_x, robot_y, robot_yaw, robot_vx, robot_vy,
obstacle_gt_x_eval_only, obstacle_gt_y_eval_only, detection_x, detection_y, num_detections,
track_id, track_x, track_y, track_vx, track_vy, track_P_xx, track_P_yy, tca_s, dca_m, margin_m, risk,
risk_score, nav_state, cmd_vx, cmd_vy, cmd_wz, contact_flag, fall_flag, goal_distance_m,
clearance_gt_m, sensor_latency_ms, planner_latency_ms`

### `evaluation/metrics.py`
```python
robot_actor_contact(model, data, robot_body_ids: set[int], actor_geom_ids: set[int]) -> bool
episode_metrics(df: pandas.DataFrame, cfg: dict) -> dict
    # goal_reached, collision, fall, collision_free_success, min_clearance_m, time_to_goal_s (nan if not
    # reached), yield_count (entries into YIELD), total_yield_s, false_stop_count, mean/p95 latencies
summarize(episodes: pandas.DataFrame) -> pandas.DataFrame   # group by scenario, mode -> table of README sec. 8
```

### `evaluation/plots.py`
```python
plot_episode_topdown(csv_path, out_png, cfg)   # robot path, GT actor (eval only, dashed grey), detections
                                               # (cyan), track (yellow), nav state colouring, walls
plot_comparison(summary_df, out_png)           # bars: success rate, min clearance, time to goal by mode
plot_prediction_error(csv_path, out_png)       # track vs GT position error over time (eval only)
```

### `scripts/evaluate.py`
CLI: `--robot amr|g1 --scenarios ... --modes ... --seeds ... --out outputs/`. For each combination
calls `humanoid_avoidance.sim.runner.run_episode(...)` (owned by Lead, see below), collects
metrics into `outputs/episodes.csv`, writes `outputs/comparison.csv` via `summarize`, plots into
`outputs/figures/`.

## D. Sim scene + perception

### `sim/robot_base.py` — robot adapter protocol (implemented by AMR and G1 adapters)
```python
class RobotAdapter(Protocol):
    name: str
    radius_m: float
    max_vx_mps: float
    def base_spec(self) -> mujoco.MjSpec          # robot + floor + light, nothing else; robot geoms NOT in perceivable group
    def bind(self, model, data) -> None           # resolve ids after compile
    def reset(self, model, data, start_xy, start_yaw) -> None   # only place where qpos may be written
    def set_command(self, cmd: NavCommand) -> None
    def pre_step(self, model, data) -> None       # called before EVERY mj_step; applies ctrl (G1: policy at its own decimation)
    def robot_state(self, model, data) -> RobotState   # localization oracle + measured base velocity
    def sensor_pose(self, model, data) -> tuple[np.ndarray, np.ndarray]   # (pos (3,), R (3,3)) of range sensor mount
    def robot_body_ids(self, model) -> set[int]
    def fallen(self, model, data) -> bool
```

### `sim/scene_builder.py`
```python
@dataclass
class SceneHandles:
    model: mujoco.MjModel; data: mujoco.MjData
    actor_mocap_id: int        # -1 if no actor
    actor_geom_ids: set[int]
    wall_geom_ids: set[int]
    perceivable_group: int
    wall_segments: np.ndarray  # (M, 2, 2) known map wall lines (for perception map filter)
build_scene(robot: RobotAdapter, cfg: dict, scenario_cfg: dict) -> SceneHandles
    # robot.base_spec() + corridor walls + optional occluder + actor (mocap body, cylinder radius
    # actor_radius_m, height actor_height_m, contype/conaffinity enabled), all in perceivable group.
    # Calls robot.bind(model, data).
```

### `sim/amr_adapter.py`
`AmrAdapter(cfg)`: cylinder robot (radius navigation.nominal_robot_radius_m) with slide-x, slide-y,
hinge-yaw joints driven by velocity actuators (body-frame command rotated to world each step) —
physical motion, no teleport. Sensor mount at mount_height_m / mount_forward_m.

### `sim/scenario_controller.py`
```python
class ScenarioController:
    def __init__(self, scenario_cfg: dict, seed: int, actor_max_speed_mps: float = 2.0)
    def actor_xy(self, time_s) -> np.ndarray | None   # GT (eval only)
    def actor_velocity(self, time_s) -> np.ndarray | None
    def apply(self, model, data, handles: SceneHandles) -> None   # writes mocap_pos each step
        # interactive scenario: sets the initial position once, then never overwrites (viewer drags it)
    def gt_actor_xy_from_sim(self, data, handles) -> np.ndarray | None  # reads mocap_pos (eval only; works for interactive)
```

### `perception/range_sensor.py`
```python
class RangeSensor:
    def __init__(self, cfg: dict, rng: np.random.Generator)
    period_s: float
    def due(self, time_s) -> bool                    # fixed cadence on sim time
    def scan(self, model, data, time_s, sensor_pos (3,), sensor_R (3,3), geomgroup_mask) -> Scan
        # mj_ray per ray (or mj_multiRay), horizontal fan in sensor frame, only perceivable group,
        # gaussian range noise, dropout -> valid_hit False; no-hit -> range = max_range, valid False
```

### `perception/clustering.py`
```python
scan_to_world_points(scan: Scan) -> np.ndarray (K,2)       # valid hits only, via geometry.transforms
filter_map_walls(points (K,2), wall_segments (M,2,2), margin) -> mask (K,)
cluster_scan(scan: Scan, wall_segments, cfg: dict) -> list[Detection]
    # distance-gap clustering on consecutive valid hits (angular order), drop walls via map filter
    # and max_obstacle_extent_m, centroid + radius + R = diag(max(sigma^2, var)) per axis
```

## Lead: `sim/runner.py`
```python
run_episode(cfg: dict, scenario_name: str, mode: str, seed: int, robot_name: str = "amr",
            headless: bool = True, record_path: Path | None = None, run_dir: Path | None = None,
            realtime: bool = False) -> dict   # returns metrics dict + "csv_path"
```
