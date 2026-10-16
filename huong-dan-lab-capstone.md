# Đề bài Capstone Track 4 — Vin Perception Hackathon

> Nguồn: VLearn, khóa `k04-l34-p2-t4`, D13, lab `lab-5d4a4679` (phần s02 → s11)
> https://vlearn.dev/course/k04-l34-p2-t4/reader?day=D13&part=lab-5d4a4679-s02-doc
>
> Repo đề bài chính thức: https://github.com/VinUni-AI20k/K4-Track4-ProjectCapstone

---

## 1. Tổng quan đề bài Capstone

### Về bài lab này
Trang đề bài hackathon cuối khóa Track 4 (Computer Vision & Robotics): 2 giờ build theo team (3–5 người/đội), 10 phút demo mỗi team, đề tài gắn với hệ sinh thái Vin (VinFast, Vin Robotics, VinMec, VinUni, Vinhomes, VinBus), vận dụng tối thiểu 2 trụ cột Track 4 kèm một số đo và một ca lỗi.

### Bạn làm được gì sau bài này
- Xây dựng một demo Computer Vision hoặc Robotics chạy thật trên webcam, cảm biến hoặc sim cho một bài toán của hệ sinh thái Vin.
- Chỉ rõ và vận dụng đúng bản chất ít nhất 2 trong 7 trụ cột kiến thức của Track 4 trong hệ thống.
- Đo lường một điều có thể kiểm chứng (số đo định lượng) và phân tích một trường hợp hệ thống làm sai kèm lý do (ca lỗi).
- Đảm bảo an toàn dữ liệu: thành viên tự đóng vai, tuyệt đối không dùng dữ liệu thật của bệnh nhân hay sinh viên.
- Thực hiện quy trình làm việc nhóm chuẩn qua Git: Trưởng nhóm tạo repository, mời các thành viên vào làm việc chung; sau đó tất cả thành viên lấy repo chung tự nộp trên tài khoản VLearn cá nhân.
- Trình bày và bảo vệ sản phẩm trong 10 phút (1 phút bài toán & người dùng, 4 phút demo trực tiếp, 2 phút số đo/ca lỗi/trụ cột/impact, 3 phút hỏi đáp).

### Cần chuẩn bị
- Hoàn thành các kiến thức và kỹ năng thực hành từ Day 01 đến Day 27 của Track 4 (Computer Vision, Camera Geometry, State Estimation, Robotics, VLM).
- Kỹ năng lập trình Python, xử lý luồng video và các thư viện thị giác máy tính / mô phỏng (OpenCV, MediaPipe, YOLO, NumPy, PyBullet/MuJoCo).
- Kỹ năng làm việc nhóm trên GitHub (tạo repo, mời Collaborators, phân chia branch, merge code).
- Thiết bị: laptop có webcam ổn định, điện thoại (nếu dùng camera/IMU), hoặc mô phỏng (sim).

Công cụ: Git & GitHub · Python 3.10+ · OpenCV, NumPy, Ultralytics YOLO, MediaPipe · Giao diện demo: Streamlit, Gradio hoặc cửa sổ OpenCV · Cảm biến (nếu dùng): Physics Toolbox Sensor Suite / Sensor Logger.

**Mục tiêu cốt lõi:** Xây một demo computer vision hoặc robotics chạy thật, cho một bài toán của hệ sinh thái Vin, chỉ bằng những gì đã học trong Track 4.

### Bốn con số định hình Vin Perception Hackathon
| | |
|---|---|
| **2 giờ** | build, làm theo team (3–5 người) — ra một vòng end-to-end chạy sớm, sau đó mới cải thiện |
| **10 phút** | mỗi team, gồm demo trực tiếp và hội đồng hỏi đáp |
| **≥ 2** | trụ cột kiến thức phải dùng — chỉ rõ chúng nằm ở đâu trong hệ thống |
| **1 + 1** | một số đo có thể kiểm chứng và một ca lỗi kèm lý do hệ thống làm sai |

### Minh họa hệ thống Perception mẫu (giao thông thông minh)
- **Khung hình camera (30 Hz):** phát hiện xe (car 0,91), người đi bộ (person 0,87), tính thời gian tới va chạm (TTC 1,8 s).
- **Cảm biến IMU (gia tốc dọc):** hai đỉnh dao động mạnh tương ứng hai lần xe cán qua ổ gà.
- **Sensor Fusion:** camera nhìn thấy hư hại mặt đường + IMU xác nhận độ xóc nảy cơ học.

### ⚠️ BẮT BUỘC: Quy định làm việc nhóm và nộp bài trên VLearn
- **Quy mô đội:** 3–5 người. Một bạn làm Trưởng nhóm tạo repository trên GitHub, mời tất cả thành viên làm Collaborator.
- **Tất cả thành viên đều phải tự nộp bài trên VLearn:** sau khi hoàn thiện mã nguồn trên repo chung, **MỖI THÀNH VIÊN ĐỀU PHẢI LẤY LINK GITHUB REPO CHUNG ĐÓ TỰ NỘP TRÊN TÀI KHOẢN VLEARN/LMS CỦA MÌNH.** VLearn chấm theo từng tài khoản cá nhân — ai không nộp sẽ không có bản nộp và không được ghi nhận điểm.

---

## 2. Chuẩn bị môi trường và Quy trình phối hợp nhóm

Khác với các bài Lab cá nhân Day 01–27, Capstone Day 28 làm theo nhóm qua một repository Git chung.

**Quy trình:**
1. **Trưởng nhóm khởi tạo** — tạo repo GitHub (Public) & mời tất cả thành viên vào Collaborators.
2. **Cả nhóm cùng phát triển** — chấp nhận lời mời, clone repo chung, tạo branch riêng và merge vào `main`.
3. **Chốt sản phẩm trên `main`** — kiểm tra code, demo, video dự phòng, chốt phiên bản hoàn chỉnh.
4. **Lấy URL repo chung** (vd. `https://github.com/leader/repo-name`) → Trưởng nhóm và từng thành viên đều tự nộp URL trên VLearn cá nhân.

### 2.1. Dành cho Trưởng nhóm
1. **Tạo repository:**
   - Đăng nhập GitHub → **New repository**.
   - Tên: `K4-Track4-Capstone-Team<Tên_Nhóm>` (hoặc fork từ `VinUni-AI20k/K4-Track4-ProjectCapstone`).
   - **Bắt buộc chọn Public** (không chọn Private vì BGK và LMS sẽ không truy cập được).
   - Khởi tạo kèm `README.md`.
2. **Mời thành viên:** Settings → Collaborators → **Add people**, nhập username GitHub từng thành viên.
3. **Gửi đường dẫn repo** cho cả nhóm để clone.

### 2.2. Dành cho thành viên
1. **Chấp nhận lời mời:** mở email/thông báo GitHub → **Accept invitation**.
2. **Clone repo:**
   ```bash
   git clone https://github.com/<leader-username>/<repo-name>.git
   cd <repo-name>
   ```
3. **Thiết lập môi trường Python** (Windows PowerShell):
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install --upgrade pip
   pip install opencv-python ultralytics mediapipe streamlit gradio numpy filterpy
   ```
   (Trang gốc có thêm tab macOS/Linux — tương đương: `python3 -m venv .venv && source .venv/bin/activate`, rồi các lệnh `pip` như trên.)
4. **Phân chia nhánh:** mỗi thành viên tạo branch riêng theo nhiệm vụ:
   ```bash
   git checkout -b feature/<tên-nhiệm-vụ>
   # Viết code và commit
   git add -A
   git commit -m "feat: bổ sung module phát hiện"
   git push origin feature/<tên-nhiệm-vụ>
   ```
   Trưởng nhóm kiểm tra và merge vào `main`.

### 2.3. Quy trình nộp bài trên VLearn cho từng người (cực kỳ quan trọng)
Khi buổi build kết thúc:
- Trưởng nhóm copy đường dẫn repo chung (vd. `https://github.com/leader/K4-Track4-Capstone-TeamVinPerception`) và gửi cho cả nhóm.
- **TỪNG BẠN (CẢ TRƯỞNG NHÓM VÀ TOÀN BỘ THÀNH VIÊN) ĐĂNG NHẬP VLEARN CỦA MÌNH, VÀO MỤC BÀI NỘP DAY 28, DÁN ĐÚNG LINK REPO CHUNG VÀ BẤM NỘP.**
- Ghi chú: `Team <Tên_Nhóm>, Trưởng nhóm: <Tên_Trưởng_Nhóm>, Vai trò cá nhân: <Nhiệm_vụ>`.

---

## 3. Đề tài: Chọn một bài toán gắn với Vin

Tự chọn bài toán cụ thể, miễn có camera, cảm biến hoặc robot và gắn với một đơn vị trong hệ sinh thái Vin.

| Đơn vị Vin | Hướng bài toán gợi ý | Ví dụ có thể làm demo |
|---|---|---|
| **VinFast** | Camera và cảm biến trong xe, hỗ trợ lái, giám sát tài xế, sạc xe điện. | Nhận diện tài xế buồn ngủ (EAR qua MediaPipe Face Mesh); ước lượng góc nghiêng đầu 3D (solvePnP); cảnh báo va chạm trước (TTC); nhận diện ổ gà kết hợp IMU; căn lề trạm sạc ArUco. |
| **Vin Robotics** | Robot di động, cánh tay robot, cảm biến, mô phỏng. | AMR tránh chướng ngại vật; cánh tay robot gắp phân loại linh kiện (Pick and Place); bám theo người vận hành; mô phỏng trong PyBullet hoặc MuJoCo. |
| **VinMec** | Giám sát an toàn bằng camera, robot hỗ trợ trong bệnh viện. | Phát hiện té ngã cho bệnh nhân cao tuổi (MediaPipe Pose); đo góc khớp phục hồi chức năng sau mổ; robot vận chuyển đồ trong hành lang bệnh viện. |
| **VinUni** | Camera và cảm biến trong khuôn viên, thể thao, phòng lab. | Phân tích tư thế thể thao (ném bóng rổ, cầu lông, nâng tạ); giám sát an toàn phòng lab (kính bảo hộ, áo blouse); đếm người ra vào thư viện. |
| **Khác** | Vinhomes, VinBus hoặc đơn vị Vin khác. | Đếm hành khách lên xuống VinBus theo hướng di chuyển; giám sát an toàn cửa xe bus; kiểm soát an ninh khu đô thị Vinhomes. |

---

## 4. Yêu cầu: Bốn điều mỗi demo phải có

Ngoài bốn điều này, cách làm hoàn toàn tự do:

1. **Chạy thật** — demo trực tiếp trên webcam, cảm biến hoặc sim. Mang theo video quay sẵn để dự phòng.
2. **Dùng kiến thức của khóa** — chỉ rõ ít nhất 2 trụ cột và chúng nằm ở đâu trong hệ thống.
3. **Một số đo, một ca lỗi** — đo một điều có thể kiểm chứng; cho xem một trường hợp hệ thống làm sai, kèm lý do.
4. **Dữ liệu an toàn** — thành viên tự đóng vai; không dùng dữ liệu thật của bệnh nhân hay sinh viên.

---

## 5. Bảy trụ cột của Track 4

Chọn ít nhất hai trụ cột.

| # | Trụ cột | Ý nghĩa | Ví dụ trong demo |
|---|---|---|---|
| 1 | **Học** | Model nhận ra gì, học từ dữ liệu nào | YOLOv8/v11 phát hiện vật thể, MediaPipe trích mốc cơ thể, VLM mô tả cảnh |
| 2 | **Hình học** | Hiệu chuẩn camera, đổi hệ tọa độ, khoảng cách | Góc xoay đầu 3D qua `cv2.solvePnP`; ước lượng khoảng cách theo tiêu cự; marker ArUco |
| 3 | **Bất định** | Nhiễu cảm biến, độ tin cậy, khi nào nghi ngờ | Ngưỡng confidence; lọc nhiễu gia tốc kế; cảnh báo "không chắc chắn" khi ngược sáng/rung |
| 4 | **Biểu diễn** | Mã hóa thành box, mốc khớp, điểm, trạng thái | 468 mốc khuôn mặt; vector trạng thái `[x, y, vx, vy]` |
| 5 | **Thời gian** | Đồng bộ cảm biến, tần số, độ trễ ảnh → quyết định | Kalman filter làm mượt & dự đoán khi bị che; đồng bộ camera 30 Hz với IMU 100 Hz; đo độ trễ end-to-end |
| 6 | **Dữ liệu** | Log, replay, nhãn, chất lượng dữ liệu | Ghi log cảm biến ra CSV; replay video test; kiểm thử nhiều điều kiện ánh sáng |
| 7 | **Bằng chứng** | Số đo, ca lỗi, giới hạn đã chứng minh | Biểu đồ FPS; phân tích ca lỗi khi đeo kính râm/quay mặt > 60°; nêu giới hạn khoảng cách (0.5 m – 2 m) |

---

## 6. Công cụ: Gợi ý để bắt đầu nhanh

Hai giờ không đủ để train từ đầu → dùng model pretrained. Công cụ do team tự chọn:

- **Đầu vào:** webcam laptop; camera và IMU điện thoại (Physics Toolbox Sensor Suite, Sensor Logger — xuất CSV); video dashcam/hành lang; sim PyBullet hoặc MuJoCo.
- **Model có sẵn:** phát hiện vật (YOLO), tư thế và mốc khuôn mặt (MediaPipe), phân đoạn, VLM mô tả cảnh.
- **Hình học và hợp nhất:** OpenCV, marker ArUco, hiệu chuẩn camera; Kalman filter bằng NumPy.
- **Giao diện demo:** cửa sổ OpenCV, Streamlit hoặc Gradio — chọn thứ chạy ổn trên máy dùng để demo.

---

## 7. Chấm điểm: Sáu tiêu chí, tổng 100 điểm

Mỗi giám khảo chấm độc lập từng tiêu chí theo ba mức, rồi lấy trung bình. Hai tiêu chí đầu chiếm một nửa tổng điểm.

### 1. Vận dụng kiến thức đã học — 25 điểm
*Sản phẩm này có làm được không nếu chưa học Track 4?*
- **Xuất sắc (20–25):** dùng ≥ 2 trụ cột đúng bản chất, chỉ ra vị trí trong hệ thống và giải thích vì sao chọn cách đó thay vì phương án khác.
- **Khá (13–19):** có dùng hai trụ cột nhưng ở mức áp dụng công thức/thư viện, ít giải thích lựa chọn.
- **Cần cải thiện (0–12):** chủ yếu gọi model có sẵn; kiến thức khóa không thể hiện trong thiết kế.

### 2. Demo chạy được — 25 điểm
*Có thấy hệ thống chạy trên đầu vào thật không?*
- **Xuất sắc (20–25):** chạy trực tiếp trên webcam/cảm biến/sim, ổn định, xử lý được đầu vào do giám khảo đưa ra tại chỗ.
- **Khá (13–19):** chạy được nhưng cần kịch bản cố định, hoặc phải chuyển sang video một lúc.
- **Cần cải thiện (0–12):** chủ yếu là video, ảnh chụp hoặc dữ liệu giả; hoặc hỏng giữa chừng không phục hồi.

### 3. Độ hữu ích — 15 điểm
*Ai sẽ dùng nó vào ngày mai, và họ được gì?*
- **Xuất sắc (12–15):** người dùng và nỗi đau rất cụ thể, gắn với một đơn vị Vin; kết quả dùng được ngay không cần giải thích.
- **Khá (8–11):** bài toán rõ nhưng người dùng/cách dùng còn chung chung.
- **Cần cải thiện (0–7):** chỉ là demo kỹ thuật, khó nói ai cần và để làm gì.

### 4. Impact — 15 điểm
*Nếu triển khai ở Vin thì thay đổi được gì, ở quy mô nào?*
- **Xuất sắc (12–15):** ước lượng tác động bằng con số (an toàn, thời gian, chi phí, số người), nêu đường mở rộng và rủi ro thực tế.
- **Khá (8–11):** nêu tác động nhưng chưa có con số, hoặc chưa nói tới mở rộng.
- **Cần cải thiện (0–7):** chỉ nói chung chung là hữu ích/tiềm năng.

### 5. Bằng chứng — 10 điểm
*Dựa vào đâu để tin kết quả này?*
- **Xuất sắc (8–10):** có số đo kiểm chứng được, ≥ 1 ca lỗi được phân tích nguyên nhân, nói rõ giới hạn.
- **Khá (5–7):** có số đo nhưng ca lỗi chưa giải thích, hoặc chưa nêu giới hạn.
- **Cần cải thiện (0–4):** không có số đo; kết luận dựa trên cảm giác.

### 6. Trình bày và hỏi đáp — 10 điểm
*Cả team có kể rõ và trả lời được không?*
- **Xuất sắc (8–10):** mạch lạc trong 10 phút, đúng giờ; mọi thành viên trả lời được về phần mình và cả hệ thống.
- **Khá (5–7):** rõ ý nhưng có thành viên không trả lời được, hoặc quá giờ.
- **Cần cải thiện (0–4):** khó theo dõi, không trả lời được câu hỏi hội đồng.

### Gợi ý chia 10 phút demo
| Thời lượng | Nội dung |
|---|---|
| 1 phút | Bài toán Vin và người dùng là ai |
| 4 phút | Demo trực tiếp, cho xem hệ thống chạy |
| 2 phút | Một số đo, một ca lỗi, trụ cột đã dùng & tác động ước lượng |
| 3 phút | Hội đồng hỏi đáp (mọi thành viên có thể được hỏi) |

### Khi hai team bằng điểm
- Trước: team có điểm **Demo chạy được** cao hơn.
- Sau: team có điểm **Bằng chứng** cao hơn.

---

## 8. Lịch trình: Một buổi, từ phát đề đến công bố

Mốc tính từ lúc bắt đầu; giờ thực tế do BTC thông báo.

| Mốc | Giai đoạn | Nội dung |
|---|---|---|
| 0:00 | Phát đề, chọn hướng | 15 phút. Mỗi team chốt bài toán và phân vai. Trưởng nhóm tạo repo GitHub, mời các bạn vào repo. |
| 0:15 | Build | 120 phút. Mục tiêu là một vòng end-to-end chạy sớm, rồi mới cải thiện. |
| 1:15 *(mốc then chốt)* | Mốc kiểm tra | Giảng viên đi qua từng team. Cần có thứ gì đó chạy từ đầu vào đến đầu ra, dù thô. |
| 2:15 | Dừng code, chuẩn bị demo | 15 phút. Quay video dự phòng, chốt số đo và ca lỗi. **Tất cả thành viên nộp link repo chung lên VLearn cá nhân.** |
| 2:30 | Demo | 10–12 team, mỗi team tối đa 10 phút. |
| Cuối buổi | Chấm điểm và công bố | BGK tổng hợp theo thang điểm và công bố giải thưởng. |

---

## 9. Checklist: Trước khi lên demo

- [ ] 1. Demo chạy trên đúng máy sẽ dùng, đã thử webcam hoặc cảm biến ở phòng demo.
- [ ] 2. Có video quay sẵn để dự phòng nếu demo trực tiếp hỏng.
- [ ] 3. Có một con số đo, và cả team nói được nó đo như thế nào.
- [ ] 4. Có ít nhất một ca lỗi, kèm lý do vì sao hệ thống sai.
- [ ] 5. Chỉ ra được ít nhất 2 trụ cột và vị trí của chúng trong hệ thống.
- [ ] 6. Không dùng dữ liệu thật của bệnh nhân hoặc sinh viên.
- [ ] 7. Mỗi thành viên trả lời được vì sao team chọn cách làm này.
- [ ] 8. Nói được người dùng là ai và họ được gì khi dùng sản phẩm.
- [ ] 9. Có một ước lượng tác động nếu triển khai ở Vin, dù chỉ là phép tính đơn giản.

---

## 10. Câu hỏi thường gặp (FAQ)

**Đề tài có bị giới hạn không?**
Không. Hướng gợi ý chỉ để tiết kiệm thời gian nghĩ. Bài toán nào có camera, cảm biến hoặc robot và gắn với một đơn vị Vin đều hợp lệ.

**Hai team chọn cùng bài toán thì sao?**
Được. Mỗi team làm theo cách riêng; so sánh hai cách làm thường là phần thú vị của buổi demo.

**Có được train model không?**
Được, nhưng 2 giờ rất ngắn. Nên bắt đầu từ model pretrained và dành thời gian cho hình học, hợp nhất cảm biến và đánh giá.

**Demo bị lỗi giữa chừng thì sao?**
Chuyển sang video quay sẵn và nói rõ đó là video. Ca lỗi có chủ đích vẫn được tính điểm, vì nó là bằng chứng.

**Không có xe hay robot thật thì sao?**
Không cần. Webcam, điện thoại và sim là đủ để làm một demo tốt.

**Thành viên không tự nộp link repo trên VLearn cá nhân thì sao?**
VLearn ghi nhận điểm theo từng tài khoản. Dù dùng chung repo GitHub, mỗi thành viên bắt buộc phải tự đăng nhập và nộp link repo đó, nếu không sẽ bị tính là thiếu bài nộp (**0 điểm**).
