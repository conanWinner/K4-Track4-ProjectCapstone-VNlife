# Humanoid tránh vật cản động bằng Perception + Motion Prediction

**Vin Perception Hackathon — Track 4 · Đơn vị: Vin Robotics**
Unitree G1 (pretrained walking policy) đi trong hành lang nhà máy mô phỏng (MuJoCo). Một công nhân / xe đẩy
băng ngang. Robot **phát hiện bằng cảm biến range giả lập → theo dõi bằng Kalman → dự đoán điểm gặp nhau
gần nhất (TCA/dCA) → quyết định GO / SLOW / YIELD / RESUME**, so sánh với baseline phản xạ (reactive).

> Mô phỏng nghiên cứu, **không phải hệ thống an toàn được chứng nhận**. Mọi số liệu dưới đây lấy từ log
> chạy thật (`outputs/*/episodes.csv`), tái lập bằng lệnh ở mục Quick Start.

## 1. Bài toán & người dùng

- **Người dùng:** đội vận hành robot của Vin Robotics triển khai humanoid/AMR chở linh kiện trong hành lang
  nhà máy (ví dụ dây chuyền VinFast), nơi công nhân và xe đẩy đi cắt ngang liên tục.
- **Nỗi đau:** luật "thấy vật trước mặt thì dừng" (reactive) chỉ phản ứng khi người đã ở ngay trước robot.
  Người đi ngang từ bên hông bước vào sườn robot → va chạm, hoặc robot phanh gấp sát người.
- **Giải pháp:** ước lượng vận tốc người bằng Kalman filter, dự đoán *khi nào và cách bao xa* hai bên gần
  nhau nhất, nhường đường **trước** khi vào vùng xung đột rồi đi tiếp khi an toàn.

## 2. Kết quả (5 seed × 7 kịch bản × 2 chế độ, mỗi robot 70 episode)

**Unitree G1 (pretrained policy, điều khiển bằng PD khớp — không chỉnh qpos):**

| Kịch bản | Reactive: không va chạm | Predictive: không va chạm | Min clearance R → P (m) |
|---|---:|---:|---:|
| empty | 100% | 100% | – |
| crossing_slow (0.5 m/s) | 20% | **100%** | −0.05 → 0.36 |
| crossing_fast (0.9 m/s) | 0% | **100%** | −0.06 → 0.20 |
| late_crossing | 0% | **100%** | −0.06 → 0.11 |
| brief_occlusion | 20% | **100%** | −0.05 → 0.73¹ |
| sudden_turn_early | 0% | **100%** | −0.04 → 0.15 |
| **sudden_turn (ca lỗi)** | 0% (100% **ngã**) | **0%** | −0.29 → −0.21 |

¹ Clearance tới người. Từ commit `3ae782a`, clearance tính cả hộp che tĩnh nên chạy lại cho **0.57 m**
(tỉ lệ thành công và time-to-goal không đổi; đã kiểm tra lại G1 seed 0–4).

AMR proxy (hình trụ, cùng pipeline) cho kết quả cùng xu hướng: reactive 0% ở cả 6 kịch bản có người,
predictive 100% ở 5/6, cả hai 0% ở `sudden_turn`. Bảng đầy đủ (yield count, false stop, sai số dự đoán,
latency): [`outputs/g1/comparison.md`](outputs/g1/comparison.md), [`outputs/amr/comparison.md`](outputs/amr/comparison.md).

- Chi phí của predictive: time-to-goal +0.4–0.7 s so với reactive ở các kịch bản băng ngang (G1 `late_crossing`: −0.2 s);
  `brief_occlusion` trên G1 chậm hơn ~11 s (chờ cạnh hộp tĩnh — xem giới hạn).
- Latency (G1): perception + tracking trung bình 1.45 ms, planner 0.49 ms mỗi tick 10 Hz.
  Sim chạy ~5× realtime (G1), ~26× (AMR).

**Ca lỗi `sudden_turn`:** người đi gần hết hành lang (CV dự đoán "đang rời đi" → robot RESUME) rồi quay
đầu 1.2 m/s đúng lúc robot tới vạch giao cắt. Kalman CV cần vài scan để đảo dấu vận tốc → YIELD đến muộn
→ va chạm 5/5. Đây là giới hạn của **mô hình vận tốc không đổi + độ trễ quan sát**, không phải "Kalman sai".
Cùng cú quay đầu nhưng xảy ra khi robot còn đang nhường ở xa (`sudden_turn_early`) thì predictive xử lý được
100%. Hình: `outputs/g1/figures/topdown_sudden_turn_predictive_s0.png`,
`prediction_error_sudden_turn_predictive_s0.png`.

## 3. Trụ cột Track 4 nằm ở đâu

| Trụ cột | Module | Minh chứng |
|---|---|---|
| Biểu diễn | `perception/clustering.py`: tia → điểm → cụm → Detection; state `[x,y,vx,vy]` | overlay cyan/vàng, top-down plot |
| Hình học | `geometry/transforms.py`: sensor → robot → world (SE2/SE3, quaternion MuJoCo) | `tests/test_transforms.py` |
| Bất định | `tracking/kalman_cv.py` (Q, R, Joseph form, NIS), gating Mahalanobis χ²=9.21, margin theo λmax(P) có chặn | đĩa covariance trong viewer, cột `track_P_*` |
| Thời gian | dt từ sim time, cadence 10 Hz, stale-scan stop, hysteresis + ramp, lọc vận tốc dáng đi | cột latency, `sensor.blackout_s` |
| Bằng chứng | so sánh cặp reactive/predictive cùng seed, ca lỗi có phân tích | `outputs/*/comparison.*`, figures |
| Học | Unitree pretrained G1 locomotion (LSTM, không tự train) | `docs/unitree_integration.md` |
| Dữ liệu | CSV mỗi tick + metadata.json mỗi run, scenario có seed | `outputs/runs/` |

Vì sao Kalman + dự đoán thay vì chỉ đo khoảng cách: xem [`docs/architecture.md`](docs/architecture.md).

## 4. Impact (ước lượng, giả định nêu rõ)

Phép tính để thuyết trình — thay số thật của nhà máy khi có:

- Giả định 1 robot gặp **N = 40** lượt người băng ngang mỗi ca (8 h), quy mô **R = 20** robot, 2 ca/ngày.
- Trong mô phỏng, tỉ lệ va chạm/near-miss khi có người cắt ngang: reactive 80–100%, predictive 0% (trừ ca quay đầu đột ngột).
- Số tình huống có rủi ro được chuyển thành "nhường đường có kế hoạch" mỗi ngày ≈ N × R × 2 = **1,600** —
  mỗi tình huống là một lần phanh gấp / va chạm tiềm năng được tránh.
- Chi phí thời gian: +0.4–0.7 s mỗi lượt giao cắt → ≈ 40 × 0.7 s ≈ 28 s/robot/ca (~0.1% thời gian ca).
- **Rủi ro & mở rộng:** sim-to-real (cảm biến thật nhiễu hơn, người đông), localization thật thay oracle,
  mô hình chuyển động người tốt hơn (IMM/constant-turn) cho ca quay đầu, thêm AVOID_LEFT/RIGHT.

## 5. Quick Start (đã xác minh: Windows 11, Python 3.11.7, MuJoCo 3.16.0, torch 2.6.0+cpu)

Demo mới trên Linux: **3 khối hộp**, robot tự chọn đường vòng bằng A* trên bản đồ
tích lũy từ tia range. Không đưa tọa độ hộp trong YAML vào bộ tìm đường.

```bash
cd humanoid-dynamic-avoidance
.venv/bin/python scripts/run_demo_no_hud.py --robot g1 --scenario obstacle_course --mode predictive
```

Các hộp xanh nằm trong `configs/scenarios.yaml` → `obstacle_course.obstacles`;
`size` là **nửa kích thước** hộp theo mét. Hộp cần đủ cao để cắt mặt phẳng quét cảm biến.
Đích của cảnh này là `[6.4, 0]`. Robot giảm tốc khi rẽ, đi vòng hộp và tiếp tục tới đích.
Va chạm với hộp cũng được tính trong log, ngoài va chạm với người.
Kiểm tra G1 (seed 0, 1, 2, 3, 4, 42): 6/6 tới đích, không va chạm/ngã,
khoảng cách nhỏ nhất 0.52–0.56 m, thời gian tới đích 38.4–40.6 s mô phỏng.
Chạy lại bằng `scripts/evaluate.py` trên Windows (timeout 45 s, seed 0–4): **4/5** tới đích
(38.4–39.1 s, clearance 0.56 m); seed 4 không va chạm nhưng kẹt ở trạng thái "không còn đường trống"
14.2 s và hết giờ cách đích 2.1 m. Khi demo nên dùng `--seed 0`.

Giới hạn: `route_from_scan` hiện chỉ dùng cho **cảnh vật cản tĩnh**, bản đồ không xóa
điểm đã quan sát và vùng chưa thấy được coi là trống. Khi không tìm được đường, lệnh
tiến được giảm về 0; G1 vẫn bước tại chỗ và có thể trôi. Cảnh có người đi ngang tiếp tục
dùng cơ chế giảm tốc/nhường đường cũ. Trình chạy Linux tắt chữ HUD do lỗi MuJoCo;
các lớp cảm biến, track và quỹ đạo vẫn hiển thị.

```powershell
cd humanoid-dynamic-avoidance
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1   # venv + deps + unitree_rl_gym @ pinned SHA
.venv\Scripts\python.exe -m pytest -q                        # 127 tests
```

Demo có cửa sổ MuJoCo (realtime, overlay + HUD):

```powershell
.venv\Scripts\python.exe scripts\run_demo.py --robot g1 --scenario crossing_slow --mode reactive
.venv\Scripts\python.exe scripts\run_demo.py --robot g1 --scenario crossing_slow --mode predictive
.venv\Scripts\python.exe scripts\run_demo.py --robot g1 --scenario sudden_turn --mode predictive
.venv\Scripts\python.exe scripts\run_demo.py --robot g1 --scenario interactive --mode predictive
```

`interactive`: giám khảo tự kéo "người" (trụ cam): **double-click** vào trụ, giữ **Ctrl + chuột phải** để kéo
ngang trước robot. Robot vẫn chỉ "thấy" qua tia range.

Đánh giá hàng loạt và video dự phòng:

```powershell
.venv\Scripts\python.exe scripts\evaluate.py --robot g1 --scenarios crossing_slow crossing_fast sudden_turn --modes reactive predictive --seeds 0 1 2 3 4 --out outputs\g1
.venv\Scripts\python.exe scripts\record_demo.py --robot g1 --seed 0      # -> outputs\demo_backup_g1.mp4
```

Thêm `--headless` để chạy không cửa sổ, `--robot amr` để dùng robot proxy (nhanh hơn ~5×).

## 6. Overlay trong viewer / video

Tia trúng: chấm **cyan** · Track Kalman: đĩa **vàng** + đĩa mờ 2σ · Quỹ đạo dự đoán: gạch **cam** ·
Vòng dưới robot: **xanh** = không rủi ro, **đỏ** = rủi ro · HUD: chế độ, trạng thái FSM, TCA, dCA,
clearance GT (chỉ để đánh giá), va chạm.

## 7. Giả định trung thực

- Ego pose lấy từ simulator (localization oracle) — không tuyên bố đã giải quyết localization.
- Perception không đọc pose actor; ground truth chỉ dùng cho scenario và cột `*_eval_only` sau khi planner đã quyết định.
- G1 chỉ di chuyển bằng pretrained policy qua PD khớp; `qpos` chỉ ghi lúc reset.
- Detection là tâm bề mặt nhìn thấy, không phải tâm vật. Robot/người xấp xỉ hình tròn (G1: 0.42 m).
- **Dữ liệu an toàn:** toàn bộ dữ liệu do simulator tự sinh ("người" là một khối trụ điều khiển theo kịch bản).
  Không dùng camera, hình ảnh hay dữ liệu thật của công nhân, bệnh nhân hoặc sinh viên.
- Chi tiết: [`docs/known_limitations.md`](docs/known_limitations.md), [`docs/experiments.md`](docs/experiments.md).

## 8. Cấu trúc

```
configs/            default.yaml (mọi ngưỡng), scenarios.yaml, upstream_version.txt
src/humanoid_avoidance/
  geometry/ perception/ tracking/ prediction/ navigation/   thuật toán (không biết GT)
  sim/       scene_builder, amr_adapter, g1_adapter, scenario_controller, runner, overlay
  evaluation/ logging/                                       metrics, plots, CSV logger
scripts/            run_demo.py, evaluate.py, record_demo.py, setup.ps1/.sh
tests/              130 tests (transforms, kalman, association, tca, planner, metrics, sensor, G1, smoke)
docs/               architecture, interfaces, unitree_integration, experiments, known_limitations
outputs/            amr/, g1/ (episodes.csv, comparison.csv/.md, figures/), videos/
```
