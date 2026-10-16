# Kịch bản demo 10 phút

Chuẩn bị trước: mở sẵn 1 terminal ở `humanoid-dynamic-avoidance`, chạy thử 1 lần mỗi lệnh dưới đây trên
**đúng máy demo**; mở sẵn `outputs/demo_backup_g1.mp4` và `outputs/g1/figures/comparison.png`.

## 0:00–1:00 · Bài toán & người dùng
- Vin Robotics đưa humanoid/AMR chở linh kiện trong hành lang nhà máy; công nhân, xe đẩy cắt ngang liên tục.
- Luật "thấy trước mặt thì dừng" phản ứng quá muộn với người đi từ bên hông.
- Chúng tôi: G1 + cảm biến range → Kalman → dự đoán điểm gặp gần nhất → nhường đường *trước*.

## 1:00–5:00 · Demo trực tiếp
```powershell
.venv\Scripts\python.exe scripts\run_demo.py --robot g1 --scenario crossing_slow --mode reactive
.venv\Scripts\python.exe scripts\run_demo.py --robot g1 --scenario crossing_slow --mode predictive
```
- Reactive: robot chỉ dừng khi người đã ở ngay trước → người va vào sườn (HUD `collision=YES`).
- Predictive: chỉ vào **đĩa vàng** (track Kalman, mũi tên vận tốc trong nhãn), **gạch cam** (quỹ đạo dự
  đoán), TCA/dCA trên HUD → robot chuyển SLOW/YIELD sớm (vòng đỏ), người đi qua, RESUME → tới đích.
- Mời giám khảo: `--scenario interactive` → double-click trụ cam, **Ctrl + chuột phải** kéo ngang trước robot.
- (Tuỳ thời gian, ~1 phút) Vật cản tĩnh: robot tự tìm đường vòng 3 hộp bằng A* trên bản đồ tích luỹ từ tia
  range, không đọc toạ độ hộp trong YAML. Đích `[6.4, 0]`, mất ~40 s mô phỏng nên chỉ chạy khi còn dư giờ:
  ```powershell
  .venv\Scripts\python.exe scripts\run_demo.py --robot g1 --scenario obstacle_course --mode predictive --seed 0
  ```
  Đã đo: 4/5 seed tới đích; seed 4 kẹt vì bản đồ chặn hết đường (ca lỗi phụ nếu bị hỏi).
  Trên Linux dùng `scripts/run_demo_no_hud.py` (tắt chữ HUD do lỗi MuJoCo). Giới hạn: chỉ cho cảnh tĩnh,
  bản đồ không xoá điểm cũ, vùng chưa thấy coi là trống.
- Nếu demo trục trặc: chiếu `outputs/demo_backup_g1.mp4`, nói rõ đây là video quay từ sim thật.

## 5:00–7:00 · Số đo, ca lỗi, trụ cột, impact
- Bảng G1 (README mục 2): crossing_slow 20% → **100%**, crossing_fast 0% → **100%**, late_crossing 0% → **100%**;
  chi phí +0.4–0.7 s/lượt.
- Ca lỗi: `--scenario sudden_turn --mode predictive` (hoặc hình `prediction_error_sudden_turn_*`):
  người quay đầu khi robot đã RESUME → CV cần vài scan để đảo vận tốc → va chạm 5/5. Hướng sửa: phát hiện
  manoeuvre bằng NIS, IMM.
- Trụ cột: Hình học (sensor→world), Bất định (Kalman, gating χ², margin theo covariance), Thời gian
  (sim time dt, stale-scan stop, hysteresis), Biểu diễn (tia → cụm → track), Bằng chứng (so sánh cặp).
- Impact: 20 robot × 40 lượt cắt ngang/ca × 2 ca ≈ 1,600 tình huống/ngày chuyển từ phanh gấp/va chạm thành
  nhường đường có kế hoạch; tốn ~28 s/robot/ca.

## 7:00–10:00 · Hỏi đáp — ai trả lời phần nào
Phân công: **Thắng** (trưởng nhóm; G1, sim, vật cản tĩnh) · **Hiếu** (perception, tracking, prediction) ·
**Chương** (planner, đánh giá, số liệu, impact). Ai cũng phải nói được luồng tổng thể
cảm biến → cụm → Kalman → TCA/dCA → FSM.

| Câu hỏi có thể gặp | Người trả lời / ý chính |
|---|---|
| Vì sao Kalman mà không chỉ khoảng cách? | **Hiếu**: cần vận tốc để biết *sẽ* gần nhau khi nào; khoảng cách chỉ là hiện tại |
| Q, R chọn thế nào? | **Hiếu**: R từ nhiễu cảm biến + độ tán cụm; Q gia tốc ngẫu nhiên σ_a = 0.7 m/s²; NIS để kiểm tra |
| Vì sao chặn margin? | **Hiếu**: Q(τ) ở 3 s cho σ ≈ 2 m > nửa hành lang → robot không bao giờ đi; cap 0.35 m là heuristic |
| Perception có "ăn gian" GT không? | **Hiếu**: chỉ `mj_ray` lên geom group 3; GT chỉ ghi cột `*_eval_only` sau khi planner quyết định |
| G1 có đi thật không? | **Thắng**: pretrained LSTM policy 50 Hz + PD 500 Hz, qpos chỉ ghi khi reset; trôi khi đứng yên → station keeping |
| Vật cản tĩnh né thế nào? | **Thắng**: lưới 0.2 m, mỗi tia trúng chặn các ô trong bán kính robot + 0.45 m, A* 8 hướng tới đích, bám waypoint; không có đường → dừng |
| GO/SLOW/YIELD quyết định thế nào? Đo số liệu ra sao? | **Chương**: TCA ≤ 3 s → SLOW, ≤ 2 s + dCA < margin → YIELD, hysteresis để RESUME; so sánh cặp cùng seed, 5 seed × 7 kịch bản, `scripts/evaluate.py` |
| Impact tính thế nào? | **Chương**: 20 robot × 40 lượt/ca × 2 ca ≈ 1,600 tình huống/ngày; giả định ghi rõ trong README mục 4 |
| Dữ liệu có an toàn không? | **Chương**: toàn bộ do simulator sinh, "người" là khối trụ theo kịch bản; không dùng hình ảnh/dữ liệu thật của ai |
| Ra đời thật cần gì? | **Thắng**: localization thật, lidar thật, nhiều người, AVOID trái/phải, mô hình người tốt hơn |
