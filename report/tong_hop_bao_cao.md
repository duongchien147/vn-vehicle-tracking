# Tổng hợp tư liệu để viết báo cáo

Tài liệu nguồn để viết báo cáo vào `docs/DL AND CV.docx`. Mọi con số dưới đây đọc lại từ file kết quả gốc (đường dẫn
ghi kèm) ngày 9/10/2026; khi viết báo cáo, chép số từ đây hoặc từ file gốc, không từ trí nhớ.

Quy ước số liệu:
- **mAP50 chính** = trung bình AP50 của **xe máy + ô tô**. Buýt và tải quá ít mẫu trên các tập Việt Nam (test Hà Nội
  chỉ có 5 buýt, 2 tải), AP của chúng nhảy giữa 0 và 1. Tập UVH-26, test Roboflow dùng cả 4 lớp; UA-DETRAC dùng
  xe máy + ô tô + buýt.
- Số trong ngoặc vuông là khoảng tin cậy bootstrap (20 lần lấy mẫu lại ảnh test).
- Tên mô hình: **v1–v5** là YOLO26n fine-tune qua 5 phiên bản dữ liệu; **v5** là mô hình triển khai; "COCO" là
  YOLO26n gốc chưa fine-tune; "@1280" là suy luận ở ảnh 1280.

---

## 1. Thông tin chung

| Mục | Nội dung |
|---|---|
| Tên đề tài (đúng như bìa) | ỨNG DỤNG HỌC SÂU TRONG PHÁT HIỆN, PHÂN LOẠI VÀ THEO DÕI PHƯƠNG TIỆN GIAO THÔNG |
| Học phần | Học sâu (bài tập lớn) |
| Ngành / chuyên ngành | Khoa học máy tính / Trí tuệ nhân tạo và khoa học dữ liệu |
| Sinh viên | Dương Minh Chiến, MSV 12423062, lớp 124231 |
| Giảng viên hướng dẫn | PGS.TS. Nguyễn Minh Tiến |
| Trường | Đại học Công nghệ Kỹ thuật Hưng Yên, 2026 |

**Phần phải sửa ngoài 4 chương** (đang còn nội dung đề tài cũ "rủi ro tín dụng"):

| Chỗ | Việc cần làm |
|---|---|
| Header mọi trang | Đổi "ỨNG DỤNG HỌC MÁY XÂY DỰNG HỆ THỐNG DỰ ĐOÁN RỦI RO TÍN DỤNG" → tên đề tài mới |
| Lời cam đoan | Đổi tên đề tài trong ngoặc kép; đổi ngày (đang là 26/5/2026) |
| Lời cảm ơn | Viết lại (bản cũ nhắc bộ môn, đề tài cũ) |
| Danh mục từ viết tắt | Thay bằng mục 7 |
| Danh sách bảng / hình | Xoá mục cũ, để Word tự sinh lại sau khi chèn chú thích SEQ mới |
| Mở đầu, 4 chương | Viết mới theo mục 3 |
| Tài liệu tham khảo | Thay bằng mục 8 (nhập vào Word citation manager, kiểu IEEE) |

## 2. Quy cách trình bày (rút từ file mẫu)

| Thành phần | Quy cách |
|---|---|
| Trang | A4; lề trên 3 cm, dưới 3 cm, trái 3.5 cm, phải 2 cm; header/footer 1.27 cm |
| Header | Tên đề tài VIẾT HOA, có gạch dưới; trang đầu section không header |
| Số trang | Giữa chân trang, đánh liên tục từ bìa (trang Nhận xét = 3) |
| Chữ thân bài | Times New Roman 13 pt, giãn dòng 1.5, căn đều, thụt đầu dòng ~1 cm, gần như không cách đoạn |
| Tên chương | Heading 1, 15 pt, đậm, không nghiêng, VIẾT HOA, căn giữa: "CHƯƠNG 1. TÊN CHƯƠNG", sang trang mới |
| Tiêu đề phần đầu | MỞ ĐẦU, MỤC LỤC, LỜI CẢM ƠN…: Heading 1, 14 pt, đậm, căn giữa |
| Mục cấp 2, 3 | "1.1. …" và "1.1.1 …": đậm, căn trái |
| Đề mục nhỏ không số | Đoạn chữ đậm (ví dụ "Nguyên lý hoạt động", "Ưu, nhược điểm") |
| Hình | Hình ở giữa; chú thích **dưới** hình: "Hình 3. 2 Tên hình", 12 pt nghiêng, căn giữa, style Caption, số bằng field `SEQ Hình_3.` |
| Bảng | Chú thích **trên** bảng: "Bảng 4. 1 Tên bảng"; bảng kẻ lưới, hàng tiêu đề nền xanh (5B9BD5) chữ đậm |
| Công thức | Bảng 2 cột không viền: công thức Word (cột trái) + "(2.1)" (cột phải); dòng sau "Công thức (2.1): mô tả"; rồi "Với:" + gạch đầu dòng giải thích ký hiệu |
| Danh sách | Gạch "-" hoặc "•"; đoạn liệt kê "Thứ nhất, … Thứ hai, …" |
| Trích dẫn | "[n]" kiểu IEEE qua Word citation manager; danh mục bằng field BIBLIOGRAPHY |
| Mỗi chương | Kết thúc bằng mục "Kết luận chương" |

**Sửa so với bản mẫu:** bản cũ dùng lẫn Heading 1/Heading 2 cho mục "n.m" (chương 1 dùng H2, chương 2–4 dùng H1),
nên mục lục bị lệch cấp. Thống nhất: tên chương = H1, mục n.m = H2, mục n.m.k = H3. Hình thức nhìn giữ nguyên.

**Đúng 4 chương, không có chương "Kết luận" riêng:** kết quả chính, hạn chế, hướng phát triển nằm ở mục cuối chương 4
(như bản mẫu).

## 3. Dàn ý chi tiết

### MỞ ĐẦU

**Lý do chọn đề tài**
- Giao thông đô thị Việt Nam: xe máy chiếm đa số, mật độ dày, che khuất nhiều; nhu cầu đếm và giám sát tự động từ
  camera giao thông.
- Mô hình phát hiện huấn luyện trên COCO chạy trực tiếp cho kết quả kém ở bối cảnh này (số liệu: test Hà Nội mAP50
  chính COCO@1280 chỉ 0.686, AP50 xe máy 0.530; bảng 4.1).
- Dữ liệu có nhãn đúng bối cảnh Việt Nam rất ít, nhất là góc camera giao thông và cảnh đêm.

**Mục tiêu đề tài**
- Tổng quát: xây dựng hệ thống phát hiện, phân loại (4 lớp `motorbike, car, bus, truck`) và theo dõi phương tiện từ
  video camera giao thông Việt Nam; đánh giá theo chuẩn (mAP kiểu COCO cho phát hiện, HOTA/MOTA/IDF1 theo
  MOTChallenge cho theo dõi).
- Cụ thể:
  - Thu thập, gộp và gán nhãn dữ liệu; tự xây tập test Việt Nam có người kiểm (ngày và đêm) và nhãn tracking.
  - Fine-tune YOLO26n qua nhiều phiên bản dữ liệu, so với baseline COCO.
  - So sánh kiến trúc CNN (YOLO26n/s) với Transformer (RT-DETRv2-S) theo giao thức công bằng.
  - Chọn và tinh chỉnh tracker (ByteTrack, BoT-SORT, OC-SORT) trên tập dev, chấm một lần trên tập test.
  - Ứng dụng: đếm xe qua vạch theo lớp và hướng; web demo Streamlit; đo tốc độ trên CPU laptop và GPU.

**Giới hạn và phạm vi**
- Phạm vi: camera cố định nhìn chéo từ trên cao (kiểu camera giao thông); ban ngày và ban đêm; 4 lớp xe.
- Không tối ưu cho góc drone nhìn thẳng xuống và góc ngang tầm mắt (chỉ dùng để phân tích lỗi).
- Phần cứng: laptop chỉ có CPU (i5-8250U) để suy luận; huấn luyện trên GPU Tesla T4 của Kaggle (quota 30 giờ/tuần).
- Tập test nhỏ (20–30 frame mỗi tập; 1 clip tracking 20 s); mỗi cấu hình train 1 lần.

**Nội dung thực hiện**
1. Dữ liệu: gộp 6 nguồn, chia theo cảnh; gán nhãn tay tập test; nhãn tự động cho frame train.
2. Huấn luyện: 5 phiên bản YOLO26n (v1–v5), YOLO26s, RT-DETRv2-S trên Kaggle.
3. Theo dõi: so 3 tracker, chỉnh tham số trên dev VnTraffic, test trên clip PVD gán tay bằng CVAT.
4. Ứng dụng: pipeline đếm xe, web demo, video demo xử lý trên GPU.
5. Đánh giá: E1 phát hiện, E2 tốc độ, E3 tracking + sai số đếm, E4 phân tích lỗi.

**Phương pháp tiếp cận**
- Nghiên cứu tài liệu (YOLO, DETR, SORT và các biến thể, chỉ số MOT).
- Thực nghiệm có kiểm soát: mọi lựa chọn (mô hình, tracker, imgsz) chốt trên tập **dev** theo **tiêu chí đặt trước**;
  tập **test** chỉ chấm một lần cấu hình đã chọn. Hoà thì chọn phương án nhanh hơn.
- Mỗi quyết định đối chiếu với chuẩn của benchmark lớn (COCO, KITTI, VisDrone, MOTChallenge) — mục 6.

### CHƯƠNG 1. GIỚI THIỆU BÀI TOÁN PHÁT HIỆN, PHÂN LOẠI VÀ THEO DÕI PHƯƠNG TIỆN GIAO THÔNG

**1.1. Giới thiệu bài toán**
- Ba bài toán con: phát hiện (vị trí box), phân loại (4 lớp), theo dõi đa đối tượng (gán ID không đổi cho từng xe
  qua các frame — tracking-by-detection).
- Đầu vào: video camera cố định 1080p; đầu ra: box + lớp + ID mỗi frame, sự kiện xe qua vạch.
- Hình đề xuất: frame video demo có vẽ box/ID/vạch đếm (mục 5, H1.1).

**1.2. Ứng dụng của bài toán**
- Đếm lưu lượng theo lớp và hướng; thống kê giao thông; hỗ trợ điều khiển tín hiệu; giám sát vi phạm (ngoài phạm vi);
  đầu vào cho hệ thống giao thông thông minh.

**1.3. Khó khăn và thách thức** (mỗi ý có bằng chứng trong dự án)
- Mật độ xe máy rất cao, che khuất lẫn nhau: một frame test Hà Nội có ~100 xe (test 30 frame: 2.982 xe máy, 539 ô tô).
- Xe rất nhỏ ở xa: tập `vn_extra` ngày có 52% xe cao < 32 px.
- Ban đêm: đèn pha/đèn hậu chói, xe tối màu; chưa có tập đêm Việt Nam có nhãn công khai.
- Lệch miền dữ liệu: mô hình học trên dataset có sẵn kém ở clip mục tiêu (v1: test Roboflow mAP50 0.861 nhưng test Hà
  Nội mAP50 chính chỉ 0.395, AP50 xe máy 0.045).
- Thiếu dữ liệu đúng góc camera giao thông; định nghĩa lớp không thống nhất giữa các dataset (xe van, minibus).
- Tài nguyên: suy luận trên CPU laptop, train bị giới hạn quota GPU.

**1.4. Kết chương**

### CHƯƠNG 2. CƠ SỞ LÝ THUYẾT

**2.1. Mạng nơ-ron tích chập và họ YOLO**
- CNN: tích chập, chia sẻ trọng số, trường tiếp nhận, backbone–neck–head.
- Phát hiện một giai đoạn kiểu YOLO [7]; YOLO26 [1]: end-to-end, **không cần NMS** (mỗi đối tượng một dự đoán), gán nhãn
  STAL, ProgLoss, optimizer MuSGD (cho batch lớn; với dữ liệu này ultralytics tự chọn AdamW — mục 3.3).
- Các cỡ n/s/m/l/x; dự án dùng n (2.5 M tham số) và s (10.0 M).
- Ưu, nhược điểm.

**2.2. Transformer và họ DETR**
- Cơ chế attention [17] (công thức Attention(Q,K,V) = softmax(QKᵀ/√d_k)V).
- DETR [3]: phát hiện như dự đoán tập hợp, ghép cặp Hungarian giữa dự đoán và nhãn, không NMS; nhược điểm hội tụ chậm.
- RT-DETR [4] và RT-DETRv2 [5]: bộ mã hoá lai (AIFI + CCFM), chọn truy vấn theo IoU, deformable attention;
  RT-DETRv2-S dùng backbone ResNet-18vd [18].
- Loss: phân lớp Varifocal + box L1 + GIoU, denoising.
- Ưu, nhược điểm (chính xác box, chậm trên CPU).

**2.3. IoU và hậu xử lý**
- Công thức IoU = |A ∩ B| / |A ∪ B|; NMS (vì sao YOLO26/DETR bỏ được); WBF [14] (dùng khi gán nhãn tự động).

**2.4. Theo dõi đa đối tượng**
- Tracking-by-detection; bộ lọc Kalman [19] (trạng thái vị trí + vận tốc, bước dự đoán/cập nhật);
  thuật toán Hungarian [20] (ghép theo ma trận chi phí 1 − IoU).
- SORT [8]; ByteTrack [9] (ghép hai bước: box điểm cao rồi box điểm thấp — lý do dùng conf 0.1);
  BoT-SORT [10] (Kalman cải tiến + bù chuyển động camera GMC + ReID tuỳ chọn); OC-SORT [11] (quan sát trung tâm).
- Bỏ phiếu lớp theo track (đóng góp của pipeline dự án — có thể đặt ở chương 3).

**2.5. Các chỉ số đánh giá** (ghi công thức theo mẫu)
- Precision, Recall; AP = diện tích dưới đường P–R (nội suy 101 điểm kiểu COCO [12]); mAP50, mAP75, mAP50-95.
- Vùng ignore (iscrowd của COCO, DontCare của KITTI [13]).
- MOTA [15] = 1 − (FN + FP + IDSW) / GT.
- IDF1 [16] = 2·IDTP / (2·IDTP + IDFP + IDFN); IDP, IDR.
- HOTA [6] = √(DetA·AssA), trung bình trên ngưỡng α ∈ {0.05,…,0.95}; DetA, AssA.
- Khoảng tin cậy bootstrap.

**2.6. Kết luận chương**

### CHƯƠNG 3. XÂY DỰNG HỆ THỐNG

**3.1. Dữ liệu**
- 3.1.1 Nguồn dữ liệu huấn luyện và giấy phép (bảng mục 9).
- 3.1.2 Chia tập: val/test chỉ lấy từ các cảnh `vnvehicle` chưa xuất hiện trong train, nhóm ảnh gần trùng bằng dHash
  (khoảng cách ≤ 10); `cantho`, `indo` chỉ dùng train vì mỗi bộ gần như một camera. **Không** dùng cách chia sẵn của
  Roboflow (test cùng cảnh với train làm mAP cao giả tạo).
- 3.1.3 Thống kê dataset v5 (Bảng 3.1, `data/processed/vn_vehicles/stats.csv`).
- 3.1.4 Tập đánh giá tự xây: test Hà Nội, `vn_extra` ngày/đêm, nhãn tracking PVD (Bảng 3.2); tập ngoài: VnTraffic (dev
  tracking), val UVH-26, UA-DETRAC.

**3.2. Gán nhãn**
- Công cụ: X-AnyLabeling (ảnh), CVAT (tracking video).
- Quy trình nhãn sơ bộ bằng mô hình → người sửa → rà soát bằng mô hình khác (tập `vn_extra` dùng YOLO26x, không dùng
  mô hình đang so sánh, để không thiên vị).
- Quy tắc: box xe máy bao cả người lái; xe mơ hồ (cây che gần hết…) gán lớp `ignore` (theo DontCare/iscrowd/ignored
  region); quy tắc lớp xe van (van nhỏ, MPV → car; van cỡ Ford Transit → bus).
- Nhãn tự động cho 69 frame train Hà Nội: ghép 4 nguồn dự đoán (v2@960, v2@1280, YOLO26x@1280, v2 + SAHI [21] 2×2)
  bằng WBF + lật ngang; ngưỡng giữ 0.15 chọn trên val. Đo trên val: P 0.905 / R 0.853 / F1 0.878, recall xe máy 0.841
  (không SAHI: 0.802).
- Bổ sung nhãn UVH-26 (thiếu ≥ 15% xe ở xa) bằng YOLO26x conf ≥ 0.5 ở chỗ chưa có box.
- Nhãn tracking clip PVD: nhãn sơ bộ từ tracker → người gán sửa trên CVAT → soát không dùng mô hình của dự án; quy ước
  vùng ignore dải xa (xe máy < 25 px theo KITTI). Lịch sử sửa: `data/tracking_gt/pvd_goc_a/cvat_backup/edit_log.json`.

**3.3. Huấn luyện mô hình phát hiện**
- Môi trường: Kaggle T4, ultralytics 8.4.163, fine-tune từ trọng số COCO, 100 epoch, dừng sớm 20, seed 42,
  `optimizer=auto` → AdamW lr 0.00125, batch hiệu dụng 64 (cộng dồn gradient tới nbs = 64).
- Lịch sử 5 phiên bản dữ liệu và lý do từng bước (Bảng 3.3, mục 4.6).
- Chọn mô hình: theo tiêu chí chốt trước — mAP50 chính trên val Hà Nội cho detector, IDR trên dev VnTraffic cho tracking,
  hoà (chênh < 0.005) thì chọn nhanh hơn.

**3.4. Giao thức so sánh kiến trúc CNN – Transformer** (`docs/giao_thuc_so_sanh_kien_truc.md`)
- Đồng nhất: dữ liệu, nhãn, pretrain COCO, imgsz 960, 100 epoch + dừng sớm 20, chọn checkpoint EMA theo mAP50-95 val,
  AdamW, AMP, seed, tối đa 300 phát hiện, không TTA, một đường chấm chung.
- Điều chỉnh RT-DETRv2 cho giống YOLO: imgsz 640 → 960; warmup 2000 bước → 3 epoch; tắt augmentation 10 epoch cuối.
- Không đồng nhất được (gắn với từng họ): lr và lịch lr, weight decay, batch hiệu dụng (64 vs 16), augmentation (mosaic
  vs không), cách đưa ảnh về kích thước (letterbox 544×960 vs kéo giãn 960×960), loss, gán nhãn.
- Đối chứng cùng số điểm ảnh: YOLO @1280 (1280×720 ≈ 960×960).
- Kiểm chứng đường chấm (bước 0): RT-DETRv2 lệch < 0.001 so với log train; YOLO26s: số ultralytics in khi train
  (0.873) cao hơn đường chấm dự án (0.847) do `yolo val rect=True` đệm thêm nửa stride — không phải lỗi.

**3.5. Theo dõi và chọn tracker**
- Dev = VnTraffic (nhãn thiếu → chỉ dùng AssA, IDR, số lần đổi ID); test = clip PVD 20 s gán tay (chấm một lần).
- Chấm không phân lớp (tracker ghép theo IoU, không xét lớp).
- So 3 tracker → chỉnh lưới tham số BoT-SORT (`track_buffer` 30/60/90, ReID bật/tắt) → `botsort_tuned`
  (`track_buffer` 90, không ReID) → tắt GMC (camera đứng yên) → `botsort_tuned_nogmc`.
- Lọc track ngắn < 3 frame trong pipeline video (= min_hits của SORT).

**3.6. Pipeline hệ thống** (sơ đồ H3.x — cần vẽ)
- Video → YOLO26 (conf 0.1) → BoT-SORT → ánh xạ lớp theo tên → bỏ phiếu lớp theo track (trọng số độ tin cậy) →
  LineZone (supervision, anchor đáy giữa box) đếm vào/ra theo lớp → vẽ box/ID/vạch → video H.264 + CSV sự kiện.
- Vì sao conf 0.1: ByteTrack/BoT-SORT cần box điểm thấp cho bước ghép thứ hai; tạo track mới do `new_track_thresh`.
- Vì sao bỏ phiếu lớp: một xe có thể bị gọi lúc car lúc truck; supervision đếm theo lớp tại lúc cắt vạch.
- ID hiển thị đánh lại #1, #2… theo thứ tự xuất hiện (ID gốc của tracker lên tới hàng nghìn).
- Web demo Streamlit; xuất OpenVINO để tăng tốc CPU.

**3.7. Kết luận chương**

### CHƯƠNG 4. KẾT QUẢ THỰC NGHIỆM

**4.1. Tiêu chí và tập đánh giá** — liệt kê các tập (Bảng 3.2), chỉ số, nguyên tắc dev/test, các thiên lệch nhãn (4.9).

**4.2. Kết quả phát hiện (E1)** — Bảng 4.1–4.5 (mục 4.1 tài liệu này).

**4.3. So sánh kiến trúc CNN – Transformer** — Bảng 4.6, 4.7; đồ thị mAP – ms/frame (cần vẽ).

**4.4. Tốc độ (E2)** — Bảng 4.8 (CPU), 4.9 (GPU).

**4.5. Kết quả theo dõi (E3)** — Bảng 4.10–4.13.

**4.6. Sai số đếm xe qua vạch** — Bảng 4.14.

**4.7. Cảnh đêm** — `vn_extra` đêm, UA-DETRAC đêm/ngày, phân tích clip đêm không nhãn.

**4.8. Phân tích lỗi (E4)** — 7 ảnh `outputs/error_analysis/`.

**4.9. Ứng dụng web và video demo** — ảnh chụp app, frame video demo GPU, số lượt qua vạch.

**4.10. Nhận xét và thảo luận** — thiên lệch nhãn sơ bộ; vì sao chọn v5 + BoT-SORT; đánh đổi độ chính xác/tốc độ.

**4.11. Kết luận chương**
- 4.11.1 Kết quả chính.
- 4.11.2 Hạn chế (mục 10).
- 4.11.3 Hướng phát triển: thêm dữ liệu đúng góc camera giao thông (bộ Đà Nẵng nếu được cấp), gán nhãn tracking cảnh
  đêm, nhiều seed, ReID chuyên cho xe, TensorRT/FP16, export RT-DETRv2 sang OpenVINO, imgsz theo cảnh.

---

## 4. Bảng số liệu

### 4.1 Phát hiện — test Hà Nội (30 frame gán tay; 2.982 xe máy / 539 ô tô / 5 buýt / 2 tải)

Nguồn: `outputs/eval_hanoi/all_models/test/results.csv`, `outputs/eval_hanoi/v3_vs_v5/test/results.csv`.

| Mô hình | imgsz | mAP50 chính [CI] | AP50 xe máy | AP50 ô tô |
|---|---|---|---|---|
| COCO (YOLO26n gốc) | 1280 | 0.686 [0.674–0.701] | 0.530 | 0.842 |
| v1 | 640 | 0.395 [0.368–0.414] | 0.045 | 0.744 |
| v2 | 960 | 0.890 [0.882–0.901] | 0.819 | 0.961 |
| v3 | 960 | 0.921 [0.915–0.934] | 0.865 | 0.978 |
| v4 | 1280 | 0.919 [0.910–0.927] | 0.862 | 0.975 |
| **v5** | 960 | **0.923** [0.915–0.932] | 0.864 | 0.982 |

Val Hà Nội (10 frame, dùng để chọn): v3 0.917 [0.904–0.927], v4 0.916 [0.902–0.926], v5 0.919 [0.909–0.928];
v5@1280 0.922 [0.914–0.931] (`outputs/eval_hanoi/v3_vs_v4/val`, `v3_vs_v5/val`, `v5_imgsz_dev/val`).
v5 OpenVINO trên val: 0.916 so với PyTorch 0.919 (`outputs/eval_hanoi/openvino_check/val`).

### 4.2 Phát hiện — `vn_extra` (cảnh mới, nhãn người, nhãn sơ bộ từ YOLO26x nên mAP50-95 dùng được)

Nguồn: `outputs/eval_hanoi/vn_extra/{day,night}/results.csv`, `outputs/eval_hanoi/vn_extra_v5_1280_analysis/day/`.
Ngày = `pvd_goc_b/c` 30 frame (552 xe máy / 156 ô tô / 0 buýt / 2 tải); đêm = `ham_chui_dem_2/3` 30 frame
(468 / 157 / 6 / 9, sau khi sửa nhãn 8/10/2026).

| Mô hình | Ngày mAP50 [CI] | Ngày mAP50-95 | Đêm mAP50 [CI] | Đêm mAP50-95 |
|---|---|---|---|---|
| COCO @1280 | 0.458 [0.432–0.489] | 0.286 | 0.437 [0.396–0.499] | 0.308 |
| v1 @640 | 0.081 | 0.035 | 0.183 | 0.106 |
| v2 | 0.736 | 0.424 | 0.496 | 0.306 |
| v3 | 0.757 | 0.460 | 0.437 | 0.263 |
| v4 @1280 | 0.808 [0.779–0.824] | 0.517 | 0.481 | 0.321 |
| **v5** | 0.745 [0.715–0.769] | 0.438 | **0.628** [0.598–0.659] | 0.404 |
| v5 @1280 (chỉ phân tích) | 0.834 [0.808–0.842] | 0.519 | 0.657 [0.619–0.691] | 0.449 |

Số v5@1280 đêm lấy từ `outputs/eval_hanoi/arch_pixel_control/night` (đã sửa nhãn); thư mục
`vn_extra_v5_1280_analysis/night` là bản trước khi sửa nhãn (0.656) — không dùng.

Phân tích (script tạm, conf 0.25, IoU 0.5, làm trước khi sửa nhãn đêm): ngày — recall xe < 20 px: v3 0.29, v5 0.31,
v4 0.45; xe ≥ 32 px mọi mô hình Việt Nam ~0.85–0.89. Đêm — xe ≥ 64 px: v5 recall 0.79, v3 0.50, v4 0.48; v3 nhầm xe
máy thành ô tô 36 lần. **Không có file kết quả lưu lại** cho phân tích này (chỉ ghi trong ghi chú dự án) — nếu đưa vào
báo cáo dưới dạng bảng thì nên chạy lại và lưu.

### 4.3 Phát hiện — 20 frame clip PVD (TP.HCM; 375 xe máy / 238 ô tô / 12 buýt)

Nguồn: `outputs/eval_clip_detection/pvd_goc_a/results.csv`. Nhãn sơ bộ từ v3 → chỉ dùng mAP50.

| Mô hình | mAP50 chính [CI] |
|---|---|
| COCO @1280 | 0.664 [0.651–0.693] |
| v2 | 0.860 [0.838–0.886] |
| v3 | 0.891 [0.869–0.911] |
| v4 @1280 | 0.907 [0.889–0.920] |
| v5 | 0.892 [0.876–0.915] |

AP50 buýt = 0 với v5 do quy ước lớp van (12 box buýt là Ford Transit), không phải mô hình không nhận được buýt.

### 4.4 Phát hiện — tập ngoài

Nguồn: `outputs/eval_uvh26/v3_vs_v5/`, `outputs/eval_ua_detrac/results.csv`, `outputs/eval_hanoi/arch_roboflow/test/`.

| Tập | COCO @1280 | v3 | v5 |
|---|---|---|---|
| Val UVH-26 (509 ảnh, 4 lớp; bỏ ảnh cùng camera với train) | 0.594 [0.577–0.621] | 0.335 [0.328–0.377] | 0.850 [0.832–0.864] |
| UA-DETRAC đêm (xe máy + ô tô + buýt) | 0.689 [0.659–0.736] | 0.722 [0.664–0.755] | 0.845 [0.790–0.865] |
| UA-DETRAC ngày | 0.690 [0.665–0.723] | 0.709 [0.655–0.756] | 0.812 [0.765–0.854] |
| Test Roboflow `vnvehicle` (308 ảnh, 4 lớp) | 0.255 | — | 0.834 |

UA-DETRAC đêm: 51 xe máy / 1.578 ô tô / 143 buýt / 21 tải; ngày: 35 / 4.003 / 246 / 19.
Đọc cẩn thận: v5 đã học 2.000 ảnh UVH-26 (khác camera) → UVH đo tổng quát sang camera mới cùng miền.

### 4.5 Kiến trúc: YOLO26n (v5) / YOLO26s / RT-DETRv2-S — cùng dữ liệu v5, cùng đường chấm

Nguồn: `outputs/eval_hanoi/arch/{test,day,night}`, `arch_roboflow/test`, `arch_pixel_control/*`,
`outputs/eval_clip_detection/arch/`, `arch_pixel_control/`, `outputs/eval_uvh26/arch/`, `outputs/eval_ua_detrac/arch/`.

mAP50 chính:

| Tập | v5 (n) @960 | YOLO26s @960 | RT-DETRv2-S @960×960 | YOLO26s @1280 (cùng số điểm ảnh) |
|---|---|---|---|---|
| Test Hà Nội | 0.923 | 0.914 | 0.929 [0.925–0.939] | 0.915 |
| `vn_extra` ngày | 0.745 | 0.798 | **0.886** [0.853–0.894] | 0.875 [0.849–0.893] |
| `vn_extra` đêm | 0.628 | 0.713 | **0.816** [0.782–0.840] | 0.800 [0.764–0.826] |
| 20 frame PVD | 0.892 | 0.903 | 0.895 | 0.907 |
| Test Roboflow (4 lớp) | 0.834 | 0.862 | 0.874 | — |
| Val UVH-26 (4 lớp) | 0.850 | 0.859 | **0.894** [0.880–0.902] | 0.865 [0.846–0.876] |
| UA-DETRAC đêm | 0.845 | 0.876 | 0.887 | 0.862 |
| UA-DETRAC ngày | 0.812 | 0.885 | 0.924 [0.875–0.949] | 0.885 |

mAP50-95 trên tập nhãn **độc lập** (không qua mô hình YOLO nào — chỉ ở đây mới so được IoU chặt):

| Tập | v5 | YOLO26s | RT-DETRv2-S |
|---|---|---|---|
| Val UVH-26 | 0.696 | 0.721 | 0.757 |
| UA-DETRAC đêm | 0.538 | 0.554 | 0.588 |
| UA-DETRAC ngày | 0.539 | 0.569 | 0.599 |

Đối chứng cùng số điểm ảnh trên `vn_extra` (mAP50-95): ngày RT-DETRv2 0.578 vs YOLO26s@1280 0.588; đêm 0.547 vs
0.548. Thêm: v5 (n) @1280 trên test Hà Nội 0.927, ngày 0.834, đêm 0.657.

**Kết luận đọc bảng:** (1) cùng imgsz 960, RT-DETRv2-S ≥ YOLO26s ở mọi tập, hơn rõ ở cảnh khó; (2) cho YOLO cùng số
điểm ảnh thì trên dữ liệu Việt Nam 1080p hai họ **hoà** → phần lớn lợi thế là do độ phân giải đầu vào; (3) trên tập
nhãn độc lập RT-DETRv2 có mAP50-95 cao hơn đều ~0.03 → box khít hơn.

### 4.6 Lịch sử huấn luyện

Nguồn: `outputs/kaggle/vn-vehicles-yolo26-train/` (`*_run_config.json`, `runs/*/results.csv`, `*_test_metrics.csv`),
`outputs/kaggle/vn-vehicles-rtdetrv2-train/`. Test = test Roboflow do ultralytics chấm (không so trực tiếp với số
đường chấm dự án — mục 3.4).

| Mô hình | Dữ liệu train | imgsz | batch | Epoch tốt nhất / đã chạy | Val mAP50 / 50-95 (lúc train) | Test Roboflow mAP50 / 50-95 | Giờ T4 |
|---|---|---|---|---|---|---|---|
| v1 (`yolo26n_e100_640`) | v1: Roboflow + VisDrone | 640 | 16 | 81 / 100 | 0.831 / 0.549 | 0.861 / 0.572 | 0.84 |
| v2 (`yolo26n_e100_960`) | v2: cantho/indo 300 ảnh mỗi bộ + 909 ảnh VisDrone ≥ 10 xe máy | 960 | 16 | 86 / 100 | 0.838 / 0.555 | 0.864 / 0.573 | 1.61 |
| v3 | v2 + 69 frame Hà Nội nhãn tự động (×4) + 10 frame val Hà Nội | 960 | 16 | 80 / 100 | 0.853 / 0.616 | 0.870 / 0.573 | 1.79 |
| v4 | như v3 | 1280 | 8 | 83 / 100 | 0.851 / 0.616 | 0.873 / 0.575 | 3.18 |
| **v5** | v3 + 2.000 ảnh UVH-26 (train 4.714 ảnh) | 960 | 16 | 90 / 100 | 0.867 / 0.628 | 0.861 / 0.578 | 2.96 |
| YOLO26s | như v5 | 960 | 8 | 87 / 100 | 0.873 / 0.639 | 0.884 / 0.600 | 4.90 |
| RT-DETRv2-S | như v5 | 960 | 8 × 2 | 61 / 81 (dừng sớm) | 0.878 / 0.651 | 0.874 / 0.601 | 10.7 |

Hình đường học: `outputs/kaggle/vn-vehicles-yolo26-train/runs/<run>/results.png`; RT-DETRv2:
`outputs/kaggle/vn-vehicles-rtdetrv2-train/rtdetrv2_r18_v5_e100_960_epochs.csv` (cần tự vẽ).

Lý do từng phiên bản: v1 → kém ở clip Hà Nội (xe máy nhỏ) → v2 tăng imgsz 960 + VisDrone nhiều xe máy → v3 thêm frame
Hà Nội nhãn tự động → v4 thử imgsz 1280 (không hơn, chậm hơn → giữ v3) → v5 thêm UVH-26 đúng góc camera giao thông
(tracking dev tốt hơn rõ → chọn v5).

### 4.7 Thống kê dataset v5 (`data/processed/vn_vehicles/stats.csv`)

| Tập | Nguồn | Ảnh | Xe máy | Ô tô | Buýt | Tải |
|---|---|---|---|---|---|---|
| train | cantho | 300 | 878 | 842 | 15 | 68 |
| train | indo | 300 | 2.181 | 844 | 49 | 51 |
| train | vnvehicle | 929 | 1.328 | 1.629 | 197 | 1.123 |
| train | visdrone | 909 | 18.050 | 28.702 | 472 | 1.801 |
| train | hanoi (69 frame × 4) | 276 | 18.192 | 4.256 | 68 | 100 |
| train | uvh26 | 2.000 | 12.348 | 7.066 | 1.231 | 3.351 |
| val | vnvehicle | 310 | 451 | 579 | 93 | 352 |
| val | hanoi | 10 | 889 | 156 | 7 | 1 |
| test | vnvehicle | 308 | 453 | 609 | 115 | 303 |

### 4.8 Tốc độ (E2)

**CPU** i5-8250U, video 1080p clip PVD, 5 đợt × 30 frame xen kẽ, trung vị [IQR]
(`outputs/benchmark_speed/e2_arch/results.csv`; tracker lúc đo là `botsort_tuned` có GMC):

| Mô hình | Đầu vào | Tham số (M) | GFLOPs | Detector ms/frame [IQR] | Detector frame/s | Pipeline ms/frame | Pipeline frame/s |
|---|---|---|---|---|---|---|---|
| v5 PyTorch | 544×960 | 2.51 | 7.4 | 107.6 [92.0–129.3] | 9.3 | 206.4 | 4.8 |
| v5 OpenVINO | 544×960 | — | — | 67.6 [64.7–74.1] | 14.8 | 155.7 | 6.4 |
| YOLO26s | 544×960 | 9.95 | 28.7 | 284.8 [263.1–322.2] | 3.5 | 404.4 | 2.5 |
| RT-DETRv2-S | 960×960 | 20.08 | 133.9 | 1561.0 [1441.7–1719.5] | 0.64 | — | — |
| YOLO26s @1280 | 736×1280 | 9.95 | 52.4 | 497.3 [461.6–614.8] | 2.0 | 705.1 | 1.4 |

**GPU** Tesla T4 Kaggle, PyTorch FP32, video VnTraffic 1080p, tracker `botsort_tuned_nogmc`
(`outputs/kaggle/vtrack-gpu-benchmark-run/outputs/benchmark_speed/e2_gpu_t4/results.csv`):

| Mô hình | Detector ms (frame/s) | Pipeline ms (frame/s) |
|---|---|---|
| v5 | 13.2 (75.7) | 23.7 (42.2) |
| YOLO26s | 14.7 (68.0) | 23.5 (42.6) |
| RT-DETRv2-S | 62.5 (16.0) | — |
| YOLO26s @1280 | 24.3 (41.2) | 33.2 (30.2) |

Ghi chú:
- Tắt GMC: pipeline OpenVINO 128 → 92 ms/frame (7.8 → 10.8 frame/s) — số này đo nhanh, **không có file kết quả lưu**.
  README đang ghi "~10.8 frame/s". Nếu đưa vào bảng báo cáo, nên đo lại bằng `scripts/benchmark_speed.py` với tracker
  `botsort_tuned_nogmc` để có file truy ngược.
- Video demo trên GPU (gồm vẽ + mã hoá video): 15.4–20.9 frame/s (`outputs/kaggle/vtrack-demo-gpu/vtrack-demo-gpu.log`).
- Không chạy pipeline với RT-DETRv2 vì tracking dùng `model.track` của ultralytics.

### 4.9 Theo dõi — dev VnTraffic (501 frame, 85 track; nhãn thiếu → so bằng AssA, IDR, IDSW)

**So tracker** (v3, cấu hình mặc định; `outputs/eval_tracking/trackers/results.csv`):

| Tracker | HOTA | AssA | IDR | IDSW | MOTA | frame/s CPU |
|---|---|---|---|---|---|---|
| ByteTrack | 0.480 | 0.615 | 0.708 | 147 | 0.054 | 4.4 |
| **BoT-SORT** | 0.505 | **0.659** | **0.738** | **128** | 0.068 | 3.6 |
| OC-SORT | 0.493 | 0.594 | 0.669 | 206 | 0.227 | 4.0 |

**Chỉnh tham số BoT-SORT** (v3; `outputs/eval_tracking/tuning_vntraffic/results.csv`, 18 cấu hình): `track_buffer`
90 không ReID: IDR 0.744, AssA 0.668, IDSW 128 (mặc định 30: 0.738 / 0.659 / 128). Bật ReID (đặc trưng của chính
detector) làm tệ đi: IDSW 207–211. Lọc track < 3 frame: IDSW 128 → 118, IDR −0.004.

**Đổi detector** (cùng `botsort_tuned`; `detector_v3_vs_v4`, `detector_v3_vs_v5`):

| Detector | HOTA | AssA | IDR | IDSW | FN | Số track dự đoán |
|---|---|---|---|---|---|---|
| v3 | 0.507 | 0.668 | 0.744 | 128 | 2.550 | 467 |
| v4 @1280 | 0.555 | 0.684 | 0.738 | 91 | 2.697 | 344 |
| **v5** | 0.554 | **0.709** | **0.827** | **62** | **1.058** | 334 |
| v5 @1280 | 0.549 | 0.713 | 0.820 | 85 | 1.183 | 341 |

**Tắt GMC** (`gmc_dev`): có GMC IDR 0.827 / IDSW 62; không GMC 0.827 / 61 → hoà → chọn bản nhanh hơn.

### 4.10 Theo dõi — test clip PVD (20 s, 183 track gán tay; chấm một lần)

Nguồn: `outputs/eval_tracking/test_pvd_goc_a_nogmc/results.csv` (cấu hình mặc định hiện tại), `test_pvd_goc_a/`.

| Cấu hình | HOTA | DetA | AssA | MOTA | IDF1 | IDR | IDP | IDSW | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|
| v5 + `botsort_tuned_nogmc` (mặc định) | 0.718 | 0.670 | 0.784 | 0.813 | 0.880 | 0.822 | 0.947 | 26 | 357 | 2.157 |
| v5 + `botsort_tuned` (có GMC) | 0.718 | 0.672 | 0.783 | 0.815 | 0.880 | 0.822 | 0.946 | 28 | 351 | 2.129 |

**Ablation trên test** (chỉ báo cáo, không dùng để chọn; `outputs/eval_tracking/ablation_pvd_goc_a/results.csv`):

| Detector + tracker | HOTA | MOTA | IDF1 | IDSW |
|---|---|---|---|---|
| v5 + BoT-SORT | 0.718 | 0.815 | 0.880 | 28 |
| v5 + ByteTrack | 0.698 | 0.791 | 0.872 | 34 |
| v5 + OC-SORT | 0.692 | 0.786 | 0.865 | 56 |
| v3 + `botsort_tuned` | 0.741 | 0.763 | 0.845 | 44 |

HOTA của v3 cao hơn là do thiên lệch nhãn sơ bộ (box giữ nguyên từ v3; thắng ở IoU ≥ 0.7). Lọc tuổi track 3: v5
IDSW 28 → 23, HOTA 0.718 → 0.708.

### 4.11 Sai số đếm xe qua vạch (clip PVD 20 s; vạch y = 0.6)

Nguồn: `outputs/eval_counting/default/pvd_goc_a/results.csv`. Số thật = track nhãn tay cho qua cùng bộ đếm của
pipeline.

| Lớp | Thật vào | Thật ra | Đếm vào | Đếm ra | Thật tổng | Đếm tổng | Sai số tuyệt đối (theo hướng) |
|---|---|---|---|---|---|---|---|
| Xe máy | 1 | 52 | 0 | 53 | 53 | 53 | 2 |
| Ô tô | 1 | 18 | 0 | 19 | 19 | 19 | 2 |
| Buýt | 0 | 1 | 0 | 0 | 1 | 0 | 1 |
| Tải | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Tổng** | 2 | 71 | 0 | 72 | 73 | 72 | 3 (4.1%) |

Soi từng lượt: 2 lượt "vào" của nhãn là hiện tượng nhãn ở tán cây (box ôm phần nhìn thấy làm đáy box đi ngược qua
vạch); lượt buýt là Ford Transit bị đếm thành ô tô → lỗi thật duy nhất: 1 lượt sai lớp.

### 4.12 Cảnh đêm không nhãn (`outputs/unlabeled_analysis/night/`)

3 clip hầm chui, 20 s mỗi clip, mốc ngày = clip PVD: độ tin cậy trung vị v5 0.59–0.63 (đêm) vs 0.63 (ngày); tỉ lệ track
< 1 s 47–52% vs 54% → v5 gần như không suy giảm; v3 tụt rõ (0.36–0.45; 67–73% track ngắn). Chỉ là chỉ số thay thế.

### 4.13 Video demo GPU (`outputs/kaggle/vtrack-demo-gpu/demo/`)

YOLO26s @960 + `botsort_tuned_nogmc`, 6 clip × 60 s 1080p; lượt qua vạch: pvd_goc_a 194 (chỉ chiều ra), pvd_goc_b
169, pvd_goc_c 254, hầm chui 1/2/3: 69 / 76 / 98 (chưa có nhãn — không phải số đánh giá).

## 5. Danh mục hình

| Mã đề xuất | Nội dung | File | Trạng thái |
|---|---|---|---|
| H1.1 | Frame demo có box, ID, vạch đếm | trích từ `outputs/kaggle/vtrack-demo-gpu/demo/pvd_goc_a_tracked.mp4` | **cần trích** |
| H1.2 | Ví dụ khó khăn: cụm xe máy dày, xe nhỏ ở xa, ban đêm | `outputs/error_analysis/2_…`, `1_…`, `3_…` (dùng lại) hoặc frame gốc | có sẵn |
| H2.x | Sơ đồ YOLO (backbone–neck–head), DETR/RT-DETR, SORT/ByteTrack | — | **cần vẽ** (vẽ lại, không chép hình bài báo nếu không trích nguồn) |
| H3.1 | Sơ đồ pipeline hệ thống | — | **cần vẽ** |
| H3.2 | Ví dụ ảnh từ mỗi nguồn dữ liệu | `data/raw/...`, `data/labeling/...` | **cần chọn** |
| H3.3 | Giao diện gán nhãn X-AnyLabeling / CVAT | — | **cần chụp** |
| H3.4 | Đường học v5 | `outputs/kaggle/vn-vehicles-yolo26-train/runs/yolo26n_v5_e100_960/results.png` | có sẵn |
| H3.5 | Đường học RT-DETRv2 | từ `rtdetrv2_r18_v5_e100_960_epochs.csv` | **cần vẽ** |
| H4.1 | Ma trận nhầm lẫn v5 trên test Roboflow | `outputs/kaggle/vn-vehicles-yolo26-train/runs/yolo26n_v5_e100_960/test/confusion_matrix_normalized.png` | có sẵn |
| H4.2 | Đồ thị mAP – ms/frame các kiến trúc | từ bảng 4.5 + 4.8 | **cần vẽ** |
| H4.3–H4.9 | 7 ảnh phân tích lỗi E4 | `outputs/error_analysis/1_xe_nho_o_xa_bi_sot.jpg` … `7_doi_id.jpg` | có sẵn |
| H4.10 | Giao diện web demo (chế độ GPU xử lý sẵn) | — | **cần chụp** (`streamlit run app/streamlit_app.py`) |
| H4.11 | Frame video demo ban đêm | trích từ `ham_chui_dem_*_tracked.mp4` | **cần trích** |

Chú thích 7 ảnh E4 (`configs/error_examples.yaml`, `outputs/error_analysis/cases.csv`): (1) xe máy rất nhỏ ở xa bị sót;
(2) cụm xe máy chờ đèn đỏ bị sót; (3) ô tô ban đêm chỉ thấy đèn, bị rào chắn che; (4) xe máy ban đêm bị gọi là ô tô;
(5) box lệch ban đêm (chỉ khoanh đèn hậu → vừa sót vừa thừa); (6) Ford Transit (nhãn buýt) bị gọi là ô tô; (7) đổi
ID khi xe bị cây che. Màu: xanh = đúng, đỏ = sót, cam = sai lớp, tím = thừa, xám = vùng ignore.

**Lưu ý bản quyền:** frame clip YouTube chỉ dùng học tập, ghi nguồn dưới hình; không đăng báo cáo công khai kèm frame.

## 6. Các quyết định và lý do (dùng cho chương 3 và hỏi đáp bảo vệ)

| Quyết định | Chuẩn tham chiếu | Lý do / bằng chứng |
|---|---|---|
| Chia val/test theo cảnh, nhóm ảnh gần trùng bằng dHash | Thực hành tránh rò rỉ dữ liệu | Test cùng cảnh với train làm mAP cao giả tạo; v1 test Roboflow 0.861 nhưng Hà Nội 0.395 |
| Luôn đánh giá thêm trên frame Việt Nam tự gán | — | Test Roboflow không đại diện clip mục tiêu |
| Lớp `ignore` cho xe mơ hồ, chấm bằng iscrowd | DontCare (KITTI), iscrowd (COCO), ignored region (VisDrone) | Không phạt bỏ sót / phát hiện ở vùng không chắc chắn |
| mAP50 chính = xe máy + ô tô | — | Buýt/tải 5/2 box: AP nhảy 0 ↔ 1 |
| Chỉ dùng mAP50 trên test Hà Nội / PVD | — | Nhãn sơ bộ từ v2/v3 → IoU chặt thiên vị mô hình đó (v2 mAP50-95 0.815 > v3 0.765) |
| Tập `vn_extra` dò lỗi bằng YOLO26x | — | Không mô hình được so sánh tham gia tạo nhãn |
| Chọn trên dev, test chấm một lần, tiêu chí đặt trước, hoà chọn nhanh hơn | Giao thức benchmark (MOTChallenge test server) | Tránh chọn theo test |
| conf 0.1 | ByteTrack dùng box điểm thấp ở bước ghép 2 | Tạo track mới do `new_track_thresh` |
| Chọn v5 | Tiêu chí đặt trước | mAP50 val hoà (0.919 vs 0.917), IDR dev 0.827 vs 0.744, IDSW 62 vs 128 |
| Giữ imgsz 960 (không 1280) | Tiêu chí đặt trước | Val 0.922 vs 0.919 (hoà); dev IDSW 85 vs 62; chậm hơn ~1.5× |
| BoT-SORT, `track_buffer` 90, không ReID | — | Tốt nhất AssA/IDR/IDSW trên dev; ReID của detector làm IDSW 128 → ~208 |
| Tắt GMC | Camera đứng yên | Dev hoà (IDSW 61 vs 62), nhanh hơn |
| Chấm tracking không phân lớp | Tracker ghép theo IoU, không xét lớp | Phân loại chấm bằng mAP |
| Quy tắc lớp van theo hình dáng thân xe | VisDrone van → car, UVH Tempo-traveller/Mini-bus → bus | Lệch luật VN và KITTI (có lớp van riêng) vì dự án chỉ 4 lớp |
| Batch theo từng họ (YOLO hiệu dụng 64, RT-DETRv2 16) | Công thức chính thức mỗi họ | Ép YOLO về nbs 16 làm `optimizer=auto` đổi sang MuSGD lr 0.01 → train sai công thức |
| Đối chứng YOLO @1280 khi so kiến trúc | — | RT-DETRv2 kéo giãn 960×960 nhìn nhiều điểm ảnh hơn ~1.8× |
| Đo tốc độ theo đợt xen kẽ, khởi động ≥ 15 s | — | Laptop hạ xung khi nóng; OpenVINO cần khởi động lâu |
| Box xe che một phần ôm phần nhìn thấy | Lệch MOTChallenge (ước lượng cả xe) | Ghi rõ là hạn chế |
| Mô hình triển khai: v5 OpenVINO trên CPU; YOLO26s nếu có GPU | — | RT-DETRv2 chậm 23× trên CPU; trên GPU n và s nhanh như nhau |

## 7. Danh mục từ viết tắt

| Tiếng Anh | Viết tắt | Diễn giải |
|---|---|---|
| Convolutional Neural Network | CNN | Mạng nơ-ron tích chập |
| You Only Look Once | YOLO | Họ mô hình phát hiện một giai đoạn |
| DEtection TRansformer | DETR | Mô hình phát hiện dựa trên Transformer |
| Real-Time DEtection TRansformer | RT-DETR | DETR thời gian thực |
| Intersection over Union | IoU | Tỉ lệ giao trên hợp |
| Non-Maximum Suppression | NMS | Loại bỏ box trùng |
| Weighted Boxes Fusion | WBF | Ghép box có trọng số |
| Slicing Aided Hyper Inference | SAHI | Suy luận theo lát cắt ảnh |
| Average Precision / mean AP | AP / mAP | Độ chính xác trung bình / trung bình trên các lớp |
| Exponential Moving Average | EMA | Trung bình trượt hàm mũ của trọng số |
| Multi-Object Tracking | MOT | Theo dõi đa đối tượng |
| Multiple Object Tracking Accuracy | MOTA | Độ chính xác theo dõi đa đối tượng |
| ID F1 score | IDF1 | F1 theo danh tính |
| Higher Order Tracking Accuracy | HOTA | Độ chính xác theo dõi bậc cao |
| Detection / Association Accuracy | DetA / AssA | Độ chính xác phát hiện / liên kết |
| ID Switch | IDSW | Số lần đổi ID |
| Global Motion Compensation | GMC | Bù chuyển động camera |
| Re-Identification | ReID | Nhận dạng lại đối tượng |
| Frames Per Second | FPS | Số frame mỗi giây |
| Giga Floating-point Operations | GFLOPs | Tỉ phép tính dấu phẩy động |
| Graphics Processing Unit | GPU | Bộ xử lý đồ hoạ |
| Automatic Mixed Precision | AMP | Huấn luyện độ chính xác hỗn hợp |
| Test-Time Augmentation | TTA | Tăng cường dữ liệu khi suy luận |
| Confidence Interval | CI | Khoảng tin cậy |

## 8. Tài liệu tham khảo (IEEE)

Đã tra lại ngày 9/10/2026 (nguồn tra ở cuối mục). Mục đánh dấu **[cần kiểm tra]** chưa xác minh đủ thông tin.

1. G. Jocher, J. Qiu, M. Liu, S. Lyu, F. C. Akyon, and M. E. Kalfaoglu, "Ultralytics YOLO26: Unified real-time
   end-to-end vision models," arXiv:2606.03748, 2026.
2. Ultralytics, "Ultralytics YOLO," GitHub repository (phiên bản 8.4.163 dùng trong đề tài), AGPL-3.0.
   https://github.com/ultralytics/ultralytics
3. N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and S. Zagoruyko, "End-to-end object detection with
   transformers," in *Proc. ECCV*, 2020, pp. 213–229.
4. Y. Zhao, W. Lv, S. Xu, J. Wei, G. Wang, Q. Dang, Y. Liu, and J. Chen, "DETRs beat YOLOs on real-time object
   detection," in *Proc. IEEE/CVF CVPR*, 2024, pp. 16965–16974.
5. W. Lv, Y. Zhao, Q. Chang, K. Huang, G. Wang, and Y. Liu, "RT-DETRv2: Improved baseline with bag-of-freebies for
   real-time detection transformer," arXiv:2407.17140, 2024.
6. J. Luiten, A. Ošep, P. Dendorfer, P. Torr, A. Geiger, L. Leal-Taixé, and B. Leibe, "HOTA: A higher order metric for
   evaluating multi-object tracking," *Int. J. Comput. Vis.*, vol. 129, pp. 548–578, 2021.
7. J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You only look once: Unified, real-time object detection," in
   *Proc. IEEE CVPR*, 2016, pp. 779–788.
8. A. Bewley, Z. Ge, L. Ott, F. Ramos, and B. Upcroft, "Simple online and realtime tracking," in *Proc. IEEE ICIP*,
   2016, pp. 3464–3468.
9. Y. Zhang, P. Sun, Y. Jiang, D. Yu, F. Weng, Z. Yuan, P. Luo, W. Liu, and X. Wang, "ByteTrack: Multi-object tracking
   by associating every detection box," in *Proc. ECCV*, 2022. **[cần kiểm tra số trang]**
10. N. Aharon, R. Orfaig, and B.-Z. Bobrovsky, "BoT-SORT: Robust associations multi-pedestrian tracking,"
    arXiv:2206.14651, 2022.
11. J. Cao, J. Pang, X. Weng, R. Khirodkar, and K. Kitani, "Observation-centric SORT: Rethinking SORT for robust
    multi-object tracking," in *Proc. IEEE/CVF CVPR*, 2023. **[cần kiểm tra số trang]**
12. T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. L. Zitnick, "Microsoft COCO:
    Common objects in context," in *Proc. ECCV*, 2014, pp. 740–755.
13. A. Geiger, P. Lenz, and R. Urtasun, "Are we ready for autonomous driving? The KITTI vision benchmark suite," in
    *Proc. IEEE CVPR*, 2012, pp. 3354–3361.
14. R. Solovyev, W. Wang, and T. Gabruseva, "Weighted boxes fusion: Ensembling boxes from different object detection
    models," *Image Vis. Comput.*, vol. 107, Art. no. 104117, 2021.
15. K. Bernardin and R. Stiefelhagen, "Evaluating multiple object tracking performance: The CLEAR MOT metrics,"
    *EURASIP J. Image Video Process.*, vol. 2008, Art. no. 246309, 2008.
16. E. Ristani, F. Solera, R. Zou, R. Cucchiara, and C. Tomasi, "Performance measures and a data set for multi-target,
    multi-camera tracking," in *Proc. ECCV Workshops*, 2016, pp. 17–35.
17. A. Vaswani *et al.*, "Attention is all you need," in *Proc. NeurIPS*, 2017, pp. 5998–6008.
18. K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE CVPR*, 2016,
    pp. 770–778.
19. R. E. Kalman, "A new approach to linear filtering and prediction problems," *J. Basic Eng.*, vol. 82, no. 1,
    pp. 35–45, 1960.
20. H. W. Kuhn, "The Hungarian method for the assignment problem," *Naval Res. Logist. Quart.*, vol. 2, no. 1–2,
    pp. 83–97, 1955.
21. F. C. Akyon, S. O. Altinuc, and A. Temizel, "Slicing aided hyper inference and fine-tuning for small object
    detection," in *Proc. IEEE ICIP*, 2022, pp. 966–970.
22. P. Zhu, L. Wen, D. Du, X. Bian, H. Fan, Q. Hu, and H. Ling, "Detection and tracking meet drones challenge,"
    *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 44, no. 11, pp. 7380–7399, 2022.
23. L. Wen, D. Du, Z. Cai, Z. Lei, M.-C. Chang, H. Qi, J. Lim, M.-H. Yang, and S. Lyu, "UA-DETRAC: A new benchmark and
    protocol for multi-object detection and tracking," *Comput. Vis. Image Underst.*, vol. 193, Art. no. 102907, 2020.
24. A. Sharma *et al.*, "The Urban Vision Hackathon dataset and models: Towards image annotations and accurate vision
    models for Indian traffic," arXiv:2511.02563, 2025. Dataset UVH-26: https://huggingface.co/datasets/iisc-aim/UVH-26
25. T.-N. Le, B.-C. Tran, D.-T. Nguyen, V.-H. Nguyen, and T.-L. Le, "Vehicle Tracking Dataset" (gồm VnTraffic), Zenodo,
    v1, 2026. doi: 10.5281/zenodo.18195750. Bài báo đi kèm: doi 10.1109/MAPR67746.2025.11133912
    **[cần kiểm tra: tên và tác giả bài báo MAPR 2025]**
26. A. Milan, L. Leal-Taixé, I. Reid, S. Roth, and K. Schindler, "MOT16: A benchmark for multi-object tracking,"
    arXiv:1603.00831, 2016.
27. TrackEval (bản PyPI `trackeval` 1.3.0, fork của J. Luiten và A. Hoffhues, "TrackEval," GitHub, 2020).
    **[cần kiểm tra cách trích dẫn bản fork]**
28. Roboflow, "Supervision," GitHub repository (phiên bản 0.30.x). https://github.com/roboflow/supervision
29. Intel, "OpenVINO toolkit." https://github.com/openvinotoolkit/openvino
30. Roboflow Universe datasets (CC BY 4.0): "vietnamese-vehicle" (car-classification, v3); "vehicle-vietnam-cantho"
    (vehicle, v19); "vehicle-detection" (skripsi-1qzlz, v2). URL ở mục 9.
31. P. Skalos, "UA-DETRAC fix masks two-wheelers," Kaggle dataset, CC BY-NC-SA 3.0. **[cần kiểm tra tên đầy đủ tác
    giả và năm]**

Nguồn đã dùng để tra: [UVH-26 arXiv](https://arxiv.org/abs/2511.02563v1),
[Zenodo VnTraffic](https://zenodo.org/records/18195750), [Ultralytics YOLO26 docs](https://docs.ultralytics.com/models/yolo26/),
[RT-DETRv2 arXiv](https://arxiv.org/pdf/2407.17140), [RT-DETR arXiv](https://arxiv.org/pdf/2304.08069),
[UA-DETRAC arXiv](https://arxiv.org/pdf/1511.04136), [VisDrone IEEE](https://ieeexplore.ieee.org/abstract/document/9573394),
[SAHI arXiv](https://arxiv.org/abs/2202.06934v1). Các mục còn lại (3, 6–20, 26) là bài kinh điển, thông tin theo trích
dẫn phổ biến; nên đối chiếu lại một lượt với Google Scholar trước khi nộp.

## 9. Nguồn dữ liệu và giấy phép

| Nguồn | Dùng cho | Giấy phép / ghi chú |
|---|---|---|
| Roboflow `vietnamese-vehicle` v3 — https://universe.roboflow.com/car-classification/vietnamese-vehicle | Train + val + test | CC BY 4.0 |
| Roboflow `vehicle-vietnam-cantho` v19 — https://universe.roboflow.com/vehicle/vehicle-vietnam-cantho-2gxc8 | Train | CC BY 4.0 |
| Roboflow `vehicle-detection` v2 — https://universe.roboflow.com/skripsi-1qzlz/vehicle-detection-ckrxi | Train | CC BY 4.0 |
| VisDrone2019-DET — https://github.com/VisDrone/VisDrone-Dataset | Train (909 ảnh ≥ 10 xe máy) | Chỉ dùng nghiên cứu |
| UVH-26 — https://huggingface.co/datasets/iisc-aim/UVH-26 | Train (2.000 ảnh), val dùng chấm phụ | CC BY 4.0 |
| Video Hà Nội — https://www.youtube.com/watch?v=Lzds95Kfh54 (Raw Street Capture 101) | Frame train (nhãn tự động), val, test (nhãn tay) | YouTube chuẩn — học tập, ghi nguồn, không đăng lại |
| Video Phạm Văn Đồng — https://www.youtube.com/watch?v=VswAfC4fER0 (Vietnam Naturally) | Chỉ test (`pvd_goc_a/b/c`) | như trên |
| Video hầm chui đêm — https://www.youtube.com/watch?v=-Ta6lRLNnnE (Midnight Soundscape) | Chỉ test (`ham_chui_dem_1/2/3`) | như trên |
| VnTraffic — doi 10.5281/zenodo.18195750 | Dev tracking | CC BY 4.0 |
| UA-DETRAC (nhãn gán lại: Kaggle `patrikskalos/ua-detrac-fix-masks-two-wheelers`; ảnh: Kaggle `bratjay/ua-detrac-orig`) | Chấm phụ đêm/ngày | CC BY-NC-SA 3.0 |

Các clip Pexels/Pixabay trong `data/videos/SOURCES.md` chỉ dùng thử pipeline và phân tích lỗi, không dùng trong bảng
số liệu.

## 10. Hạn chế cần nêu trung thực

- Mỗi cấu hình train **1 lần (1 seed)**; bootstrap chỉ đo dao động do mẫu ảnh test.
- Tập test nhỏ: 30 frame Hà Nội, 30 + 30 frame `vn_extra`, 20 frame PVD; tracking test chỉ 1 clip 20 s; sai số đếm chỉ
  73 lượt trên 1 clip.
- Nhãn VnTraffic thiếu xe → HOTA/MOTA trên dev thấp hơn thực tế; chỉ dùng AssA/IDR/IDSW.
- Thiên lệch nhãn sơ bộ: test Hà Nội (sơ bộ từ v2) và PVD (từ v3) → chỉ dùng mAP50; HOTA ở IoU cao thiên vị v3.
- Cảnh đêm Việt Nam có nhãn phát hiện (`vn_extra` đêm) nhưng **chưa có nhãn tracking**; UA-DETRAC đêm là đường có đèn,
  giao thông Trung Quốc.
- Hai tập dev (val Hà Nội, VnTraffic) không có cảnh xe rất nhỏ ở xa → lựa chọn imgsz 960 không phản ánh lợi ích của
  1280 ở cảnh như `pvd_goc_b/c`.
- Lớp xe van không có trong 4 lớp → Ford Transit hay bị gọi là ô tô.
- Box xe che một phần ôm phần nhìn thấy (lệch quy ước MOTChallenge).
- Tốc độ CPU chưa đạt thời gian thực; RT-DETRv2 chưa export OpenVINO; GPU mới đo FP32, chưa TensorRT.
- Tracker chỉnh với v3 rồi dùng cho v5.

## 11. Lệch giữa tài liệu nội bộ và file kết quả (đã kiểm khi tổng hợp)

- ghi chú dự án ghi số box `vn_extra` đêm là 465 / 159 / 7 / 8 — đó là **trước** khi sửa nhãn 8/10/2026; file kết quả hiện
  tại là **468 / 157 / 6 / 9**. Báo cáo dùng số sau.
- Số "pipeline OpenVINO không GMC 10.8 frame/s" (README, ghi chú dự án) không có file kết quả lưu lại (mục 4.8).
- Bảng phân tích recall theo kích thước xe (mục 4.2) không có file kết quả lưu lại.
- ghi chú dự án ghi tốc độ RT-DETRv2-S trên CPU là "1.561 [1.442–1.719] ms" — đúng ra là **giây** (1561 ms/frame).
