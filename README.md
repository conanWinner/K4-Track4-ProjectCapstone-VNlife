# K4 Track 4 Capstone — Humanoid tránh vật cản động (Vin Robotics)

> **Bài nộp của team nằm ở thư mục [`humanoid-dynamic-avoidance/`](humanoid-dynamic-avoidance/).**
> Bắt đầu đọc từ [`humanoid-dynamic-avoidance/README.md`](humanoid-dynamic-avoidance/README.md).

Unitree G1 (pretrained walking policy) đi trong hành lang nhà máy mô phỏng bằng MuJoCo. Robot phát hiện người
băng ngang bằng cảm biến range giả lập, theo dõi bằng Kalman filter, dự đoán điểm gặp nhau gần nhất (TCA/dCA)
rồi nhường đường **trước** khi va chạm. Kết quả được so sánh với baseline phản xạ trên cùng seed.

| | |
|---|---|
| Kết quả chính (G1, 5 seed) | Không va chạm: reactive 0–20% → predictive **100%** ở 5 kịch bản có người băng ngang |
| Ca lỗi | `sudden_turn`: người quay đầu khi robot đã đi tiếp → va chạm 5/5 (giới hạn mô hình vận tốc không đổi) |
| Trụ cột Track 4 | Hình học, Bất định, Thời gian, Biểu diễn, Bằng chứng (vị trí trong code: README project, mục 3) |
| Video dự phòng | [`humanoid-dynamic-avoidance/outputs/demo_backup_g1.mp4`](humanoid-dynamic-avoidance/outputs/demo_backup_g1.mp4) |
| Kịch bản thuyết trình | [`humanoid-dynamic-avoidance/docs/demo_script.md`](humanoid-dynamic-avoidance/docs/demo_script.md) |
| Dữ liệu | Chỉ dữ liệu mô phỏng tự sinh; không dùng hình ảnh hay dữ liệu thật của người nào |

## Team VNBlife

| Thành viên | GitHub | Vai trò | Phụ trách (code + hỏi đáp) |
|---|---|---|---|
| **Đoàn Quang Thắng** (Trưởng nhóm) | [conanWinner](https://github.com/conanWinner) | Tích hợp Unitree G1 + vật cản tĩnh | `sim/` (g1_adapter, scene_builder, runner), `navigation/scan_route.py` |
| Phạm Minh Hiếu | [hieulovecat](https://github.com/hieulovecat) | Perception + Kalman/TCA | `perception/`, `geometry/`, `tracking/`, `prediction/` |
| Nguyễn Hữu Chương | [hchuong04](https://github.com/hchuong04) | Planner + đánh giá + thuyết trình | `navigation/controller.py`, `evaluation/`, `docs/` |

Chạy nhanh (Windows PowerShell):

```powershell
cd humanoid-dynamic-avoidance
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
.venv\Scripts\python.exe scripts\run_demo.py --robot g1 --scenario crossing_slow --mode predictive
```

---

## Tài liệu đề bài hackathon (giữ nguyên từ repo gốc)

- `index.html`: trang đề bài (một file, không cần build). Mở trực tiếp bằng trình duyệt.
- `huong-dan-lab-capstone.md`: bản markdown của hướng dẫn lab, gồm quy định nộp bài trên VLearn.
- `docs/README_Humanoid_Dynamic_Obstacle_Avoidance.md`: đặc tả đề tài ban đầu của team.
