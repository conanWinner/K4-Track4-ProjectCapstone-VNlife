# Dynamic Obstacle Avoidance for Humanoid Robots Using Perception and Motion Prediction

> **README / Implementation Contract dành cho coding agent**  
> Ngôn ngữ: Python 3.10+ · Robot: Unitree G1 · Simulator: MuJoCo · Walking: pretrained Unitree policy · Không tự train model từ đầu.  
> **Ưu tiên:** sản phẩm chạy được, đúng pipeline, có baseline và số đo; không làm UI đẹp trước khi có demo end-to-end.

## 0. YÊU CẦU BẮT BUỘC CHO AGENT

Hãy **tự triển khai mã nguồn hoàn chỉnh** theo README này, kiểm thử từng phase và tạo artifact demo. **Không dừng ở pseudocode**. Chỉ hỏi người dùng khi thực sự vướng quyền truy cập/hardware hoặc dependency bất khả thi. Không tự báo “đã chạy thành công” khi chưa có log thực thi. Khi không thể chạy G1 trong môi trường hiện tại, hoàn thiện các module độc lập + unit tests + integration adapter, đánh dấu rõ `SIM_NOT_VERIFIED`, không bịa metric.

**Quy tắc trung thực kỹ thuật:**
1. G1 phải **được điều khiển bằng pretrained locomotion policy và joint/PD commands**; tuyệt đối không chỉnh `qpos`/root pose để giả đi bộ. Chỉ được khởi tạo/reset vị trí đầu episode.
2. **Perception không được đọc pose/vận tốc ground-truth của obstacle từ MuJoCo.** Dùng ray measurements/depth để phát hiện; ground truth actor chỉ dùng cho scripted scenario và evaluation.
3. MVP cho phép đọc **ego pose của G1 từ simulator như localization oracle** (nêu rõ trong báo cáo), để tập trung vào tracking/avoidance. Không tuyên bố đã giải quyết localization; sau đó có thể thay bằng IMU/odometry.
4. Dự đoán CV + Kalman dùng timestamp thật của simulation, không mặc định frame rate cố định và không dùng dữ liệu tương lai.
5. Demo phải có **baseline** (không prediction hoặc reactive), **proposed** (Kalman + predictive risk), **metric thực đo**, **failure case**, **video dự phòng**.
6. Không dùng dữ liệu cá nhân. Người/xe đẩy trong sim là actor giả lập; không thu thập video người thật.
7. Luôn ưu tiên dừng an toàn khi cảm biến stale, lost track không thể đảm bảo clearance, hoặc risk quá cao. Đây là mô phỏng nghiên cứu, **không phải safety-certified robotics system**.

## 1. MỤC TIÊU & PHẠM VI

**Bài toán:** Humanoid Unitree G1 đi từ điểm A đến B trong hành lang nhà máy giả lập. Một actor (người/xe đẩy biểu diễn bằng collision geoms) băng ngang đường; G1 dùng perception để phát hiện, theo dõi, dự đoán và quyết định **GO / SLOW / YIELD / RESUME**. Sau MVP, thêm **AVOID_LEFT / AVOID_RIGHT** nếu an toàn, nhưng không đánh đổi độ ổn định của demo đầu tiên.

**Đạt MVP nếu:**
- Scene không vật cản: G1 tự đi tới goal.
- Scene actor cắt ngang: cảm biến nhìn thấy actor; Kalman filter xuất `x,y,vx,vy,P`; planner YIELD trước khi tiến vào vùng nguy hiểm; sau đó RESUME tới goal.
- Có **hai mode** `reactive` và `predictive`, chạy cùng seed/scenario, xuất CSV + bảng số liệu.
- Có failure case đổi hướng đột ngột hoặc occlusion, phân tích được nguyên nhân.
- Có video MP4 bằng screen recording/camera renderer của sim.

**Không thuộc MVP:** VLM, YOLO thật, SLAM hoàn chỉnh, obstacle footstep planner 3D, tự huấn luyện gait, nhiều người đông đúc, chứng nhận an toàn.

## 2. CÁC TRỤ CỘT CỦA KHÓA HỌC

| Trụ cột | Nơi triển khai | Minh chứng |
|---|---|---|
| Representation (Day 6–7) | `perception/` chuyển range rays -> cụm obstacle -> centroid | Plot ray hits, detections |
| Geometry (Day 4) | `geometry/` sensor frame -> robot frame -> world frame | Unit test transform/round trip |
| Uncertainty (Day 5) | `tracking/` Kalman state covariance Q/R, gating | Covariance ellipses, residuals |
| Time (Day 4, 8) | timestamps, update/predict, stale detection | Log sensor rate, latency, Δt |
| Evidence (Day 9, 12) | `evaluation/` paired scenarios, failure case | CSV, success/collision/min-distance |
| Learning (Day 1) | Pretrained G1 locomotion only | Model source/version documented |
| Data (Day 9) | CSV traces, scenario seeds | Reproducible logs |

Tối thiểu **Representation + Time + Uncertainty** cần thực sự có code; Geometry và Evidence bắt buộc cho chất lượng đồ án.

## 3. STACK KỸ THUẬT / THAM KHẢO

- Official Unitree walking baseline: https://github.com/unitreerobotics/unitree_rl_gym
- Official MuJoCo simulator: https://github.com/google-deepmind/mujoco
- Optional G1 specific sim: https://github.com/unitreerobotics/unitree_mujoco
- Python: `numpy`, `scipy`, `mujoco`, `torch` (bản phù hợp hệ thống), `matplotlib`, `pandas`, `pytest`, `pyyaml`. Optional `opencv-python` / `imageio[ffmpeg]`.

**Không tự giả định file config cụ thể:** trước khi tích hợp, agent phải `git clone`/inspect repo Unitree ở version đã pin commit, tìm runner deployment, policy weights, joint order, obs dimensions, command scales, sim step và control decimation. Repo upstream có thể thay đổi. Ghi phát hiện thực tế vào `docs/unitree_integration.md`. Nếu pretrained weights không tồn tại trong checkout, báo lỗi rõ và dùng URL nguồn chính thức, **không tạo weights giả**.

## 4. KIẾN TRÚC DỮ LIỆU / THỜI GIAN

```text
MuJoCo (500 Hz physics, VERIFY actual config)
   ├── G1 state: base pose, velocity, joint state → pretrained policy → joint targets
   ├── simulated range sensor (e.g. 10 Hz) → 2D hit points + timestamp
   │       ↓
   │  sensor_frame → robot_frame → world_frame (localization oracle for MVP)
   │       ↓
   │  clustering → detections (position, radius, confidence proxy)
   │       ↓
   │  gated assignment → CV Kalman multi-object tracks
   │       ↓
   │  predicted trajectories, covariance, 0–3 s horizon
   │       ↓
   └── Local planner (e.g. 10 Hz) → [vx, vy, yaw_rate] command
                                              ↓
                                    G1 pretrained locomotion policy (~50 Hz)
                                              ↓
                                          physics loop
```

Sử dụng **simulation time** (`data.time`) làm clock chung. Không dùng wall-clock cho Kalman `dt`; wall-clock chỉ phục vụ profiling FPS/latency. Trong mỗi vòng: `mj_step` -> cập nhật đúng cadence -> phát hành command -> policy. Tránh duplicate update Kalman nhiều lần trên cùng scan.

`state` và quy ước:
- Global ENU-like XY ground plane, góc yaw dương ngược chiều kim đồng hồ.
- Robot frame: `+x` hướng tới trước, `+y` trái, `+z` lên (phải **xác minh** với URDF/MJCF).
- `twist=[vx, vy, yaw_rate]` trong **body frame**; verify runner scaling, clip theo giới hạn policy.
- Units: m, s, rad, m/s, rad/s. Mọi measurement/track có `timestamp`.

### Data contracts (dataclasses hoặc TypedDict)

```python
@dataclass
class Scan:
    time_s: float
    angles_rad: np.ndarray     # shape (N,), relative to sensor +x
    ranges_m: np.ndarray       # shape (N,), finite hit / max range
    valid_hit: np.ndarray      # shape (N,), bool (max range != obstacle)

@dataclass
class Detection:
    time_s: float
    xy_world_m: np.ndarray     # (2,)
    radius_m: float
    covariance_xy: np.ndarray  # (2,2)

@dataclass
class Track:
    track_id: int
    time_s: float
    state: np.ndarray          # [x,y,vx,vy] in world frame
    covariance: np.ndarray     # 4x4
    radius_m: float
    confirmed: bool
    missed_updates: int

@dataclass
class NavCommand:
    vx_mps: float
    vy_mps: float
    yaw_rate_rps: float
    mode: str                  # GO / SLOW / YIELD / RESUME / AVOID / STOP
```

Unit tests phải assert shape, finiteness, units, monotonic timestamp.

## 5. CHI TIẾT MODULE (CÔNG THỨC CẦN IMPLEMENT)

### 5.1 Scene + actor trajectory

- Scene phẳng, giới hạn hành lang (ví dụ 8 x 4 m), start `(0,0)`, goal `(5,0)`; tùy chỉnh theo workspace G1.
- Actor hình trụ/box có collision hình học hợp lệ, di chuyển từ `(2.5,-1.6)` lên `(2.5,+1.6)` theo vận tốc tùy chọn. Dùng MuJoCo mocap body, actuator/path controller hoặc phương pháp tương đương, **chỉ cho obstacle, không cho G1**. Tránh actor teleport xuyên người robot giữa bước sim; dùng bước cập nhật nhỏ, giới hạn vận tốc, log contact.
- Actor phải có geom/group sensor thấy được, physics collision hoạt động.
- Scenario: `empty`, `crossing_slow`, `crossing_fast`, `late_crossing`, `sudden_turn`, `brief_occlusion`; parameterized seed/time/trajectory.
- Ground truth actor chỉ truyền vào `ScenarioController` và `Evaluator`, tuyệt đối không vào `Perception` / `Tracker` / `Planner`.

### 5.2 Simulated range perception

**MVP:** fan of horizontal rays (e.g. 61 rays, FOV 180°, range 0.1–4 m), có thể 2–3 lớp độ cao. Gắn sensor frame vào torso/head bằng pose quaternion/rotation từ MuJoCo; tính endpoint bằng `mujoco.mj_ray` / API phù hợp version. Loại trừ robot geoms; không đánh đồng ray max-range với hit. Với ground/ceiling/self-hit, dùng geom filtering/group đúng API.

Từ mỗi valid range `r_i` tại góc `α_i`:

\[
{}^{S}\mathbf p_i = (r_i\cos\alpha_i,\ r_i\sin\alpha_i,\ 0)^T.
\]

\[
{}^{W}\mathbf p_i = {}^{W}\mathbf T_R\,{}^{R}\mathbf T_S\,{}^S\tilde{\mathbf p}_i.
\]

`T_R_S` lấy từ pose sensor attachment, `T_W_R` từ ego pose (localization oracle MVP). Nếu ray không nằm ngang, dùng 3D direction đầy đủ thay vì công thức 2D.

Nhóm điểm hit bằng DBSCAN đơn giản hoặc **distance-gap clustering** cho scan góc đều; xuất centroid `(x,y)`, extent/radius và `R` dựa trên range noise/cluster dispersion. Kiểm tra hiện tượng tường bị chia nhiều cluster hoặc người bị nhập với tường. **Không cần YOLO** trong MVP.

**Lưu ý observability:** Một scan đơn lẻ chỉ đo được bề mặt nhìn thấy, không trực tiếp có tâm actor. Tạm dùng centroid/visible-surface position và tăng safety margin; không được trình bày là ground-truth object center.

### 5.3 Multi-object data association (ban đầu 1 actor)

- Với một actor và ít clutter: nearest neighbor trên vị trí dự đoán + gating Mahalanobis.
- Khi nhiều actor: tạo cost matrix rồi `scipy.optimize.linear_sum_assignment`; assignment không hợp lệ ngoài gate bị loại, không gán cưỡng bức.

\[
d^2(z,\hat z)=(z-H\hat x)^\top S^{-1}(z-H\hat x),\quad S=HPH^\top+R.
\]

Dùng ngưỡng chi-square 2 DoF, ví dụ `9.21` (xấp xỉ 99%); viết test với detection nhiễu, misses và đổi ID. Quản lý track states `tentative/confirmed/lost/deleted`, ví dụ xác nhận sau 2/3 cập nhật, xóa sau 5 misses; ngưỡng configurable.

### 5.4 Kalman Filter constant velocity

Trạng thái `x=[p_x,p_y,v_x,v_y]^T`, measurement `z=[p_x,p_y]^T`.

\[
F(\Delta t)=\begin{bmatrix}1&0&\Delta t&0\\0&1&0&\Delta t\\0&0&1&0\\0&0&0&1\end{bmatrix},\quad
H=\begin{bmatrix}1&0&0&0\\0&1&0&0\end{bmatrix}.
\]

Với white acceleration noise có phương sai cường độ `σ_a²` trên mỗi trục:

\[
Q=\sigma_a^2\begin{bmatrix}
\Delta t^4/4&0&\Delta t^3/2&0\\
0&\Delta t^4/4&0&\Delta t^3/2\\
\Delta t^3/2&0&\Delta t^2&0\\
0&\Delta t^3/2&0&\Delta t^2
\end{bmatrix}.
\]

Đây là giả thiết gia tốc ngẫu nhiên *piecewise-constant per interval*; phải nêu rõ vì white-noise acceleration liên tục có Q khác. Chọn `R=diag(σ_x²,σ_y²)` từ mô hình đo hoặc thống kê residual, không tự nhận là calibrated.

**Predict:** `x_pred=F x`, `P_pred=F P F.T+Q`.

**Update:** `y=z-H x_pred`; `S=H P_pred H.T+R`; `K=P_pred H.T inv(S)`; `x=x_pred+K y`.

Dùng **Joseph form** cho covariance:

\[
P=(I-KH)P_{pred}(I-KH)^\top+KRK^\top.
\]

Khi mất measurement: predict-only, covariance tăng. Test với quỹ đạo thẳng, noise, dropout, variable dt, `P` symmetric/PSD. Khởi tạo velocity ban đầu 0 với covariance lớn, không kết luận tốc độ từ một frame.

### 5.5 Motion prediction và collision risk

Với prediction horizon `Horizon = 2.0–3.0s`, ở các bước `τ`:

\[
\hat p_o(t+\tau)=p_o(t)+v_o(t)\tau.
\]

Dự đoán covariance vị trí bằng `F(τ) P F(τ)^T + Q(τ)`; margin tăng nếu uncertainty tăng.

Với robot và obstacle, đặt `q = p_o-p_r`, `w=v_o-v_r`. **Time to Closest Approach (TCA)** (không gọi nhầm là Time to Collision):

\[
\tau^*=\operatorname{clip}\left(-\frac{q\cdot w}{w\cdot w+\varepsilon},0,Horizon\right),
\qquad d_{ca}=\|q+w\tau^*\|.
\]

Với `w≈0`, chọn `τ*=0`, và kiểm tra khoảng cách hiện tại. Rủi ro nếu `d_ca < r_robot+r_obstacle+margin` **và** hai đối tượng đang tiến vào vùng xung đột trong horizon. Đây là mô hình hình tròn 2D, chưa tính đầy đủ gait/foot sweep.

**Uncertainty-aware margin (heuristic, không phải bảo đảm xác suất):**

\[
margin=m_0+k\sqrt{\lambda_{max}(P_{o,xy}(\tau^*))}.
\]

Đánh giá giới hạn: đổi hướng đột ngột làm CV sai; che khuất lâu làm P tăng; robot yaw/acceleration thực tế khác commanded velocity. Với v_robot trong risk, ưu tiên **ước lượng vận tốc base thực tế**, không dùng lệnh `cmd` làm vận tốc thật. Với các candidate commands, có thể giả lập rollout kinematic xấp xỉ rồi chấm risk; nêu giả định.

### 5.6 Local planner / finite state machine

States: `GO`, `SLOW`, `YIELD`, `RESUME`, `STOP_SENSOR_FAILURE`, tùy chọn `AVOID_LEFT/RIGHT`.

- `GO`: đi tới goal theo heading error.
- `SLOW`: giảm tốc nếu TCA/TCA-distance tới ngưỡng.
- `YIELD`: command gần 0, tiếp tục perception và tracking; không đi vòng qua người khi không đủ clearance.
- `RESUME`: sau khi risk thấp liên tục `0.5–1.0s`, tăng tốc theo ramp để tránh trạng thái nhảy giật.
- `STOP_SENSOR_FAILURE`: nếu dữ liệu scan quá cũ (ví dụ >0.3 s) hoặc pose không hợp lệ, dừng.
- `GOAL_REACHED`: trong tolerance e.g. 0.4 m, về zero command.

Giới hạn lệnh theo **policy training command range đã inspect**; không tự gán tốc độ quá phạm vi. Low-pass / slew-limit `cmd` theo thời gian. Heading controller ví dụ `yaw_rate=clip(k_yaw*wrap(heading_goal-yaw), -wmax,wmax)`; khi hướng sai lớn giảm `vx`. **Không yêu cầu G1 lùi hoặc sidestep nếu pretrained policy chưa ổn định**. Vòng lặp prediction khác với reactive baseline như sau:

- `reactive`: dùng **chỉ vị trí obstacle mới nhất**, dừng nếu khoảng cách trước mặt dưới một ngưỡng; không dùng velocity/TCA.
- `predictive`: dùng state và P của tracker, TCA/dCA, quyết định YIELD sớm hơn.

Cả hai sử dụng **cùng scene, cùng controller, cùng sensor**. Khác biệt chỉ ở logic risk/planner.

### 5.7 Control adapter vào Unitree G1

1. Clone repo official, pin SHA vào `configs/upstream_version.txt`.
2. Kiểm tra entrypoint Sim2Sim; chạy pretrained G1 trên mặt phẳng **trước khi thêm actor/perception**.
3. Xác định biến command của runner (thường body velocities x/y/yaw); chèn adapter ở đúng chỗ **trước khi build observation**; chắc chắn shape/scale khớp config thực.
4. **Không sửa shape observation 47 thành 47+N**; planner nằm bên ngoài policy. Không fine-tune weights.
5. Nếu cần sửa scene MJCF có obstacle, giữ nguyên robot actuator, joint order, root body/collision và PD gains.
6. Cung cấp `--mode manual|reactive|predictive`, `--scenario ...`, `--headless`, `--seed`, `--record`.
7. Đảm bảo actor injection, perception scan và policy action xảy ra đúng cadence trong một simulation loop.

## 6. CẤU TRÚC REPOSITORY BẮT BUỘC

```text
humanoid-dynamic-avoidance/
├── README.md                      # README này + quickstart thực tế
├── requirements.txt
├── pyproject.toml                 # nếu phù hợp
├── configs/
│   ├── default.yaml
│   ├── scenarios.yaml
│   └── upstream_version.txt
├── scripts/
│   ├── setup.sh
│   ├── run_demo.py               # --mode / --scenario / --seed
│   ├── evaluate.py               # batch evaluation
│   └── record_demo.py            # xuất MP4 (nếu renderer hỗ trợ)
├── src/humanoid_avoidance/
│   ├── types.py
│   ├── geometry/transforms.py
│   ├── sim/g1_adapter.py
│   ├── sim/scenario_controller.py
│   ├── perception/range_sensor.py
│   ├── perception/clustering.py
│   ├── tracking/kalman_cv.py
│   ├── tracking/multi_tracker.py
│   ├── prediction/risk.py
│   ├── navigation/controller.py
│   ├── evaluation/metrics.py
│   └── logging/csv_logger.py
├── tests/
│   ├── test_transforms.py
│   ├── test_range_sensor.py
│   ├── test_kalman.py
│   ├── test_association.py
│   ├── test_tca.py
│   ├── test_planner.py
│   └── test_integration_smoke.py
├── docs/
│   ├── architecture.md
│   ├── unitree_integration.md
│   ├── experiments.md
│   └── known_limitations.md
└── outputs/
    ├── runs/                      # one folder per run, CSV + metadata
    ├── figures/
    ├── comparison.csv
    └── demo_backup.mp4
```

`outputs/` và clone upstream nên gitignore, ngoại trừ các báo cáo đã chọn nộp.

## 7. CÁC PHASE THỰC HIỆN (PHẢI THEO THỨ TỰ)

### P0 — Environment & pretrained locomotion
- [ ] Cài dependency, lưu phiên bản thực tế.
- [ ] Clone/pin official Unitree repo; kiểm tra G1 model và policy weights.
- [ ] Chạy G1 trong MuJoCo với velocity command nhỏ, không vật cản.
- [ ] Xác nhận root pose đi được bằng **joint-controlled policy**, ghi trajectory và clip 10s.
- [ ] Tạo `docs/unitree_integration.md` mô tả entrypoint, mapping, observation dim/action dim/rates theo repo đã inspect.

**Gate:** G1 chạy được trên mặt phẳng; nếu không, KHÔNG tuyên bố hoàn thiện humanoid demo.

### P1 — Scene + sensor
- [ ] Thêm actor di chuyển cắt ngang với collisions.
- [ ] Implement ray scan + filtering self/ground; ray return finite/no-hit rõ ràng.
- [ ] Plot points trên top-down view; không dùng obstacle pose trực tiếp.
- [ ] Test actor stationary/moving, sensor mất tầm nhìn.

**Gate:** scan thay đổi theo khoảng cách actor; hit locations sai số hợp lý so với bề mặt obstacle (chỉ test/eval mới so GT).

### P2 — Geometry + detection + tracking
- [ ] Chuyển sensor points sang world bằng transform đúng.
- [ ] Cluster thành detection; `R` có đơn vị m².
- [ ] Implement Kalman predict/update, association và track lifecycle.
- [ ] Plot observed vs predicted trajectory; log velocities, P, NIS nếu có.

**Gate:** unit tests `pytest` pass, track velocity hội tụ trên actor đi thẳng; không dùng future trajectory.

### P3 — Risk + navigation
- [ ] TCA/dCA + uncertainty margin; xử lý stationary and receding actor.
- [ ] FSM GO/SLOW/YIELD/RESUME/STOP; command smooth.
- [ ] Tạo baseline reactive tương thích.
- [ ] Nối với G1 policy, kiểm tra không bị fall/stall vì command quá gắt.

**Gate:** crossing actor làm predictive chuyển YIELD và resume; empty scene đi tới goal.

### P4 — Evaluation + demo
- [ ] Chạy paired tests cùng seeds cho `reactive` và `predictive` (tối thiểu 5–10 episodes mỗi scenario/mode nếu khả thi).
- [ ] Ghi metrics, logs, tables, clips.
- [ ] Tạo `sudden_turn` failure case và giải thích CV assumption.
- [ ] Dựng video dự phòng, chụp plot trajectories và predicted intercept.

**Gate:** `comparison.csv` thực, một scene success và một failure video, metrics tái lập bằng command line.

## 8. METRICS, LOGGING VÀ ABLATION

### Episode metrics

1. `goal_reached`: final distance to goal ≤ threshold.
2. `collision`: contact event G1-vs-actor/environment (lọc ground/normal foot contact, không đếm mọi tiếp xúc sàn).
3. `fall`: base height/tilt dưới/ngoài ngưỡng trong khoảng thời gian đủ dài, loại lúc robot ngồi/điểm khởi động.
4. **Collision-free success** = (`goal_reached and not collision and not fall`); tính tỷ lệ trên tổng episodes.
5. `min_clearance_m`: khoảng cách biên hình học tối thiểu G1–obstacle trong episode (sử dụng GT **chỉ cho evaluator**; nếu tính approximate circle, ghi là approximation).
6. `time_to_goal_s`, `yield_count`, `total_yield_s`, `false_stop_count` (phải định nghĩa rõ, ví dụ dừng khi không có nguy cơ trong horizon theo GT), `sensor_latency_ms`, `planner_latency_ms`.
7. Prediction error khi có GT: ADE/FDE hoặc velocity error qua horizon, đo trên các track đã matched (không rò GT vào algorithm).

### Log mỗi tick chính

`sim_time_s, scenario, seed, mode, robot_x, robot_y, robot_yaw, robot_vx, robot_vy, obstacle_gt_x_eval_only, obstacle_gt_y_eval_only, detection_x, detection_y, track_x, track_y, track_vx, track_vy, track_P_xx, track_P_yy, tca_s, dca_m, risk, nav_state, cmd_vx, cmd_vy, cmd_wz, contact_flag, fall_flag, goal_distance_m`

Trong code, GT fields nằm sau bước algorithm, **chỉ cho logging/evaluation**; không để tracker/planner nhận struct có GT.

### Bảng kết quả (không bịa số)

| Scenario | Mode | Episodes | Collision-free success | Collision rate | Mean min clearance (m) | Mean time-to-goal (s) |
|---|---|---:|---:|---:|---:|---:|
| crossing_slow | reactive | TODO | TODO | TODO | TODO | TODO |
| crossing_slow | predictive | TODO | TODO | TODO | TODO | TODO |
| crossing_fast | reactive | TODO | TODO | TODO | TODO | TODO |
| crossing_fast | predictive | TODO | TODO | TODO | TODO | TODO |
| sudden_turn | predictive | TODO | TODO | TODO | TODO | TODO |

**Ablation tối thiểu:** reactive vs predictive. Nếu dư thời gian, predictive with uncertainty margin vs without margin. Báo cáo điều kiện mà prediction **không cải thiện**, ví dụ CV sai trong sudden turn.

## 9. FAILURE CASE CẦN THỂ HIỆN

**Sudden turn / reversal:** actor đang đi ngang đột ngột đổi hướng tại `t=t_flip`; Kalman CV cần một số measurement để cập nhật velocity, nên quỹ đạo dự đoán sai. Log plot `predicted path` vs `ground truth`, TCA, dCA và thời điểm planner chuyển mode. Giải thích đây là giới hạn của mô hình CV và độ trễ quan sát, không gán nhãn “Kalman sai” một cách chung chung.

**Biện pháp có thể thử:** tăng margin theo covariance, giảm horizon khi track chưa ổn định, innovation-based maneuver detection, constant-turn/IMM, ưu tiên stop nếu uncertainty cao. Nhưng cần đo tradeoff false stops.

## 10. FILE CONFIG GỢI Ý

```yaml
simulation:
  physics_dt_s: 0.002             # verify actual Unitree runner
  seed: 42
  episode_timeout_s: 35.0
sensor:
  frequency_hz: 10
  fov_deg: 180
  num_rays: 61
  max_range_m: 4.0
  noise_std_m: 0.025
  dropout_probability: 0.01
tracking:
  sigma_accel_mps2: 0.7
  measurement_sigma_m: 0.10
  mahalanobis_gate_sq: 9.21
  min_confirmations: 2
  max_misses: 5
prediction:
  horizon_s: 2.5
  nominal_margin_m: 0.35
  uncertainty_k: 2.0
navigation:
  max_vx_mps: 0.20          # clip additionally by actual policy training limits
  max_yaw_rate_rps: 0.40
  goal_tolerance_m: 0.4
  stale_scan_timeout_s: 0.3
  safe_resume_duration_s: 0.6
  nominal_robot_radius_m: 0.35
```

Tất cả giá trị trên là **starting defaults, không phải kết quả tuning hoặc tiêu chuẩn an toàn**. Tránh có hard-coded threshold rải rác trong modules; đưa tất cả vào YAML. Với tốc độ đi thực tế khác target, phải đo và chỉnh sau.

## 11. LỆNH CLI CẦN ĐƯỢC AGENT HỖ TRỢ

```bash
# 1. Setup dependencies + fetch pinned policy/simulator
bash scripts/setup.sh

# 2. Unit tests, không cần khởi động simulator
python -m pytest -q

# 3. Test G1 without obstacle
python scripts/run_demo.py --scenario empty --mode manual --seed 42

# 4. Baseline (no motion prediction)
python scripts/run_demo.py --scenario crossing_slow --mode reactive --seed 42 --record

# 5. Proposed (Kalman + prediction)
python scripts/run_demo.py --scenario crossing_slow --mode predictive --seed 42 --record

# 6. Failure case
python scripts/run_demo.py --scenario sudden_turn --mode predictive --seed 42 --record

# 7. Batch evaluation and summary plots
python scripts/evaluate.py --scenarios crossing_slow crossing_fast sudden_turn --modes reactive predictive --seeds 0 1 2 3 4
```

Đây là **required CLI contract cho agent implement**, không phải khẳng định các file/lệnh đã tồn tại ngay bây giờ. `scripts/setup.sh` cần xử lý đường dẫn upstream rõ ràng và báo lỗi thân thiện nếu thiếu weights.

## 12. TEST PLAN CHI TIẾT

- Geometry: `T^-1(T(p))≈p`, yaw ±90°/180°, sensor offset, world-vs-body velocities.
- Range: empty scene no hit, obstacle at known distance, no self-hit, near/far range, occlusion.
- Kalman: CV constant velocity convergence; variable dt; missing 2–5 frames; covariance remains symmetric PSD; innovation behaves sensibly.
- Association: two crossing detections, out-of-gate rejects, track ID persistence.
- TCA: head-on approaching risk, crossing path risk, parallel similar-speed no risk, moving away no false TTC, overlapping currently risky.
- Planner: stale scan stops, goal reached stops, risk → YIELD, risk clears for hysteresis → RESUME, no aggressive oscillation.
- G1: shape input/output matches actual weight, command clipping, robot advances without changing root pose directly, no repeated unexplained falls.
- Reproducibility: same seed/config reproduces scenario parameters and evaluation close enough within simulation determinism limitations.

## 13. GIAO DIỆN DEMO TỐI THIỂU

Hiển thị trực tiếp trên MuJoCo viewer hoặc thêm top-down debug window:
- G1 và dynamic actor;
- current obstacle hit points (cyan), Kalman tracked point (yellow), predicted trajectory (dashed orange), target (green);
- status `GO/SLOW/YIELD/RESUME`, risk score / `t_ca`, `d_ca`, velocity command;
- HUD episode result: collision, fall, elapsed time, distance to goal.

**Ưu tiên video dự phòng** 20–40 giây: baseline reactive và proposed predictive với cùng crossing scenario, rồi 10–15 giây failure case. Nếu không xuất MP4 được qua offscreen renderer, quay màn hình thật; không fake video bằng animation không chạy simulator.

## 14. ĐẦU RA BẮT BUỘC KHI AGENT HOÀN THÀNH

1. Toàn bộ source + config + tests trong cấu trúc trên.
2. README `Quick Start` **đã xác minh** (OS, Python, versions, repo SHA, chính xác command).
3. `docs/architecture.md`: diagram pipeline, giải thích tại sao Kalman + prediction thay vì distance-only.
4. `docs/unitree_integration.md`: gốc model và policy, observation/action mapping và nguồn tham khảo.
5. `outputs/comparison.csv` + plots từ **runs thực**; nếu chưa chạy được không chế số.
6. `outputs/demo_backup.mp4` nếu chạy được simulator.
7. `docs/known_limitations.md`: failure case, hạn chế sensor, velocity vs actual motion, limitation of CV and circular robot proxy.
8. Một báo cáo tóm tắt: **PASS/FAIL theo từng phase**, test output, file thực tế tạo được, blockers chính xác.

## 15. DEMO SCRIPT 3–5 PHÚT

1. **0:00–0:30**: Bài toán người/xe đẩy cắt ngang đường G1 trong hành lang.
2. **0:30–1:00**: Trụ cột Representation/Geometry/Time/Uncertainty nằm ở module nào.
3. **1:00–2:00**: Chạy `reactive`, actor băng ngang; quan sát lúc GO/YIELD và kết quả.
4. **2:00–3:00**: Chạy `predictive` cùng seed, bật trajectory predictions/TCA, quan sát robot dừng sớm và đi tiếp.
5. **3:00–4:00**: Show metric đo thực tế và một failure `sudden_turn`.
6. **4:00–5:00**: Giải thích lựa chọn, hạn chế, và vì sao không tuyên bố an toàn tuyệt đối.

## 16. TÀI LIỆU THAM KHẢO NỀN TẢNG

- R. E. Kalman, “A New Approach to Linear Filtering and Prediction Problems”, 1960 — Kalman recursion.
- H. W. Kuhn, “The Hungarian Method for the Assignment Problem”, 1955 — assignment, dùng khi tracking nhiều actor.
- Y. Bar-Shalom et al., *Estimation with Applications to Tracking and Navigation*, Wiley, 2001 — linear state estimation, tracking/data association.
- Official MuJoCo API documentation: https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html
- Official Unitree pretrained policy repo: https://github.com/unitreerobotics/unitree_rl_gym
- Unitree MuJoCo simulation repo: https://github.com/unitreerobotics/unitree_mujoco

Các phương trình là **mô hình phổ biến và giả thiết được ghi rõ**, không được tuyên bố trích từ một paper cụ thể nếu không đối chiếu nguồn gốc chính xác. TCA là bài toán chuyển động tương đối hình học, không đồng nghĩa thời điểm va chạm thực và không bảo đảm an toàn theo dynamics của humanoid.

---

## 17. LỜI NHẮN CUỐI CHO CODING AGENT

**Thứ tự ưu tiên:** chạy G1 pretrained → actor và range sensor → Kalman tracking → prediction risk → FSM → integration → metrics/videos. Mỗi phase phải có bằng chứng chạy/test. Nếu không thể làm đầy đủ, hoàn thành tối đa code có thể test độc lập và ghi rõ tình trạng; **không dùng placeholder có vẻ chạy được để che lỗi**. Mọi quyết định lớn (giới hạn perception, simulator oracle, proxy collision geometry, model limitation) phải được ghi trong tài liệu nộp.
