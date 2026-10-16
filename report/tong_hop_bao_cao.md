# Tổng hợp tư liệu để viết báo cáo

Tài liệu nguồn để viết báo cáo vào `docs/DL AND CV.docx`. Mọi con số dưới đây đọc lại từ file kết quả gốc (đường dẫn
ghi kèm) ngày 9/10/2026; khi viết báo cáo, chép số từ đây hoặc từ file gốc, không từ trí nhớ.

Quy ước số liệu:
- **mAP50 chính** = trung bình AP50 của **xe máy + ô tô**. Buýt và tải quá ít mẫu trên các tập Việt Nam (test Hà Nội
  chỉ có 5 buýt, 2 tải), AP của chúng nhảy giữa 0 và 1. Tập UVH-26, test Roboflow dùng cả 4 lớp; UA-DETRAC dùng
  xe máy + ô tô + buýt.
- Số trong ngoặc vuông là khoảng tin cậy bootstrap 95% (1.000 lần lấy mẫu lại ảnh test; chấm lại 9–10/10/2026 trong
  các thư mục `ci1000`, thay bản 20 lần trước đó). Frame cùng clip giống nhau nên khoảng thật có thể rộng hơn.
  Với 1.000 lần, hai kết luận cũ yếu đi: v4 so với v5 ban ngày ở `vn_extra` và YOLO26s so với YOLO26n ở `vn_extra` ngày,
  UVH-26, UA-DETRAC giờ có khoảng tin cậy chồng nhau; RT-DETRv2-S so với YOLO26s@1280 ở UVH-26 cũng chồng nhẹ.
- Bootstrap theo khối (10/10/2026, mục 4.14): lấy mẫu lại cả khối 10 giây liên tiếp trong mỗi clip (UA-DETRAC: cả
  video) vì frame gần nhau chung xe. Kết luận chính giữ nguyên; yếu đi thêm hai chỗ: YOLO26s chỉ tách v5 sát nút ở
  `vn_extra` đêm; UA-DETRAC ngày v5 chạm mô hình gốc.
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

**Lý do chọn đề tài** (viết lại 13/10/2026: chỉ nêu lý do, KHÔNG đưa kết quả thực nghiệm của chương 4; kèm Hình 1
`intro_day.jpg` / `intro_night.jpg` do `report/make_figures.py` tạo)
- Thực tế: 77 triệu xe máy đăng ký, ~770 xe/1.000 dân (tính đến 9/2024, Ủy ban ATGT quốc gia, Báo Tiền Phong
  4/11/2024, `tienphong2024xemay`); giao thông hỗn hợp, xe máy dồn cụm ở ngã tư; cần số liệu lưu lượng theo loại xe và
  hướng để điều khiển đèn, cải tạo nút giao.
- Hạ tầng camera: Hà Nội vận hành Trung tâm điều khiển giao thông thông minh 12/2025, 1.837 camera tại 195 nút giao,
  phân tích hình ảnh để xác định mật độ, lưu lượng, chỉnh đèn (VnEconomy 13/12/2025, `vneconomy2025camera`) -> cần máy
  tự phát hiện, phân loại, theo dõi.
- Khoảng trống kỹ thuật: mô hình huấn luyện sẵn trên COCO (ít cảnh xe máy dày nhìn từ trên cao; xe máy và người lái là
  hai đối tượng) -> phải tinh chỉnh và đo lại; dữ liệu Việt Nam có nhãn ít (vài camera, hầu như không có góc camera
  giao thông, không tìm được tập đêm có nhãn); điểm trên tập test có sẵn chưa nói được về camera khác.
- Giá trị học thuật: ghép CNN/Transformer, theo dõi đa đối tượng, chỉ số chuẩn; thu thập và gán nhãn, chia tập, cân đối
  độ chính xác với tốc độ trên laptop không GPU rời.

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
- RT-DETR [4] và RT-DETRv2 [5]: bộ mã hoá lai (AIFI + CCFF), chọn truy vấn có độ bất định nhỏ nhất (tên theo bản CVPR 2024), deformable attention;
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
- Web demo Streamlit; xuất OpenVINO đầu vào 544×960 chạy trên GPU Intel tích hợp (hoặc INT8 trên CPU), mục 4.4.1.

**3.7. Kết luận chương**

### CHƯƠNG 4. KẾT QUẢ THỰC NGHIỆM

**4.1. Tiêu chí và tập đánh giá** — liệt kê các tập (Bảng 3.2), chỉ số, nguyên tắc dev/test, các thiên lệch nhãn (4.9).

**4.2. Kết quả phát hiện (E1)** — Bảng 4.1–4.5 (mục 4.1 tài liệu này).

**4.3. So sánh kiến trúc CNN – Transformer** — Bảng 4.6, 4.7; đồ thị mAP – ms/frame (cần vẽ).

**4.4. Tốc độ (E2)** — Bảng 4.8 (CPU), 4.9 (GPU).

**4.5. Kết quả theo dõi (E3)** — Bảng 4.10–4.13.

**4.6. Sai số đếm xe qua vạch** — Bảng 4.14.

**4.7. Phân tích lỗi (E4)** — 7 ảnh `outputs/error_analysis/`.

**4.8. Ứng dụng web và video demo** — ảnh chụp app, frame video demo, xe cứu hộ, khoá lớp hiển thị.

**4.9. Nhận xét và thảo luận** — thiên lệch nhãn sơ bộ; test Hà Nội và tổng quát; các lần dùng lại tập test;
bootstrap theo khối; các lần huấn luyện (seeds); lựa chọn cuối cùng và đánh đổi. Kết quả cảnh đêm nằm ở 4.2.2 (phát
hiện) và 4.6.2 (đếm xe); mục "Kết quả ở cảnh ban đêm" riêng đã bỏ (16/10/2026) vì lặp số liệu.

**4.10. Kết luận chương**
- 4.10.1 Kết quả chính.
- 4.10.2 Hạn chế (mục 10).
- 4.10.3 Hướng phát triển: thêm dữ liệu đúng góc camera giao thông, gán nhãn tracking cảnh
  đêm, nhiều seed, ReID chuyên cho xe, TensorRT/FP16, export RT-DETRv2 sang OpenVINO, imgsz theo cảnh.

---

## 4. Bảng số liệu

### 4.1 Phát hiện — test Hà Nội (30 frame gán tay; 2.982 xe máy / 539 ô tô / 5 buýt / 2 tải)

**Lưu ý (rà soát 9/10/2026):** frame test, val và train Hà Nội lấy từ **cùng một camera, cùng ngã tư**, chỉ khác thời
điểm và mức zoom. Với v3–v5 (đã học frame Hà Nội), tập này đo độ chính xác **trên camera đã tinh chỉnh**, không đo khả
năng tổng quát. Bằng chứng: v2 → v3 tăng 0.890 → 0.921 ở Hà Nội nhưng chỉ 0.736 → 0.757 ở `vn_extra` ngày và giảm
0.496 → 0.437 ở `vn_extra` đêm. Khả năng tổng quát đọc ở `vn_extra`, PVD, UVH-26, UA-DETRAC (camera mới).

Nguồn: `outputs/eval_hanoi/all_models/test/results.csv`, `outputs/eval_hanoi/v3_vs_v5/test/results.csv`.

| Mô hình | imgsz | mAP50 chính [CI] | AP50 xe máy | AP50 ô tô |
|---|---|---|---|---|
| COCO (YOLO26n gốc) | 1280 | 0.686 [0.668–0.705] | 0.530 | 0.842 |
| v1 | 640 | 0.395 [0.369–0.417] | 0.045 | 0.744 |
| v2 | 960 | 0.890 [0.877–0.901] | 0.819 | 0.961 |
| v3 | 960 | 0.921 [0.911–0.933] | 0.865 | 0.978 |
| v4 | 1280 | 0.919 [0.907–0.929] | 0.862 | 0.975 |
| **v5** | 960 | **0.923** [0.913–0.932] | 0.864 | 0.982 |

Val Hà Nội (10 frame, dùng để chọn): v3 0.917 [0.901–0.932], v4 0.916 [0.899–0.929], v5 0.919 [0.907–0.931];
v5@1280 0.922 [0.912–0.932] (`outputs/eval_hanoi/v3_vs_v4/val`, `v3_vs_v5/val`, `v5_imgsz_dev/val`).
v5 OpenVINO trên val: xem bảng OpenVINO ở mục 4.8 (bản demo 544×960 trên GPU tích hợp 0.918 so với PyTorch 0.919).

### 4.2 Phát hiện — `vn_extra` (cảnh mới, nhãn người, nhãn sơ bộ từ YOLO26x nên mAP50-95 dùng được)

Nguồn: `outputs/eval_hanoi/vn_extra/{day,night}/results.csv`, `outputs/eval_hanoi/vn_extra_v5_1280_analysis/day/`.
Ngày = `pvd_goc_b/c` 30 frame (552 xe máy / 156 ô tô / 0 buýt / 2 tải); đêm = `ham_chui_dem_2/3` 30 frame
(468 / 157 / 6 / 9, sau khi sửa nhãn 8/10/2026).

| Mô hình | Ngày mAP50 [CI] | Ngày mAP50-95 | Đêm mAP50 [CI] | Đêm mAP50-95 |
|---|---|---|---|---|
| COCO @1280 | 0.458 [0.416–0.502] | 0.286 | 0.437 [0.363–0.498] | 0.308 |
| v1 @640 | 0.081 | 0.035 | 0.183 | 0.106 |
| v2 | 0.736 | 0.424 | 0.496 | 0.306 |
| v3 | 0.757 | 0.460 | 0.437 | 0.263 |
| v4 @1280 | 0.808 [0.770–0.843] | 0.517 | 0.481 | 0.321 |
| **v5** | 0.745 [0.699–0.787] | 0.438 | **0.628** [0.575–0.681] | 0.404 |
| v5 @1280 (chỉ phân tích) | 0.834 [0.799–0.866] | 0.519 | 0.657 [0.607–0.705] | 0.449 |

Số v5@1280 đêm lấy từ `outputs/eval_hanoi/arch_pixel_control/night` (đã sửa nhãn); thư mục
`vn_extra_v5_1280_analysis/night` là bản trước khi sửa nhãn (0.656) — không dùng.

Recall theo chiều cao xe (`scripts/analyze_size_recall.py`, `outputs/size_recall/default/results.csv`; conf 0.25,
IoU 0.5; nhãn đêm đã sửa; đêm bỏ khoảng < 20 px vì chỉ có 1 xe):

| Mô hình | Ngày <20 (192) | 20–32 (175) | 32–64 (203) | ≥64 (140) | Đêm 20–32 (37) | 32–64 (316) | ≥64 (286) |
|---|---|---|---|---|---|---|---|
| v3 | 0.29 | 0.52 | 0.84 | 0.86 | 0.11 | 0.34 | 0.43 |
| v4 @1280 | 0.45 | 0.59 | 0.82 | 0.84 | 0.19 | 0.37 | 0.45 |
| v5 | 0.31 | 0.49 | 0.88 | 0.84 | 0.05 | 0.53 | 0.74 |
| v5 @1280 | 0.50 | 0.66 | 0.87 | 0.90 | 0.16 | 0.56 | 0.72 |
| YOLO26s | 0.46 | 0.66 | 0.87 | 0.89 | 0.49 | 0.62 | 0.75 |
| YOLO26s @1280 | 0.72 | 0.78 | 0.92 | 0.88 | 0.57 | 0.70 | 0.79 |
| RT-DETRv2-S | 0.80 | 0.85 | 0.96 | 0.94 | 0.51 | 0.77 | 0.91 |

v3 nhầm xe máy thành ô tô ban đêm 36 lần (16 + 20); nới IoU 0.3 chỉ tăng recall 2–8 điểm. Số ngày trùng bản phân tích
tạm trước đây; số đêm khác (bản cũ: v5 0.79, v3 0.50, v4 0.48) vì làm trước khi sửa nhãn đêm.

### 4.3 Phát hiện — 20 frame clip PVD (TP.HCM; 375 xe máy / 238 ô tô / 12 buýt)

Nguồn: `outputs/eval_clip_detection/pvd_goc_a/results.csv`. Nhãn sơ bộ từ v3 → chỉ dùng mAP50.

| Mô hình | mAP50 chính [CI] |
|---|---|
| COCO @1280 | 0.664 [0.638–0.693] |
| v2 | 0.860 [0.833–0.890] |
| v3 | 0.891 [0.865–0.915] |
| v4 @1280 | 0.907 [0.883–0.929] |
| v5 | 0.892 [0.866–0.923] |

AP50 buýt = 0 với v5 do quy ước lớp van (12 box buýt là Ford Transit), không phải mô hình không nhận được buýt.

### 4.4 Phát hiện — tập ngoài

Nguồn: `outputs/eval_uvh26/v3_vs_v5/`, `outputs/eval_ua_detrac/results.csv`, `outputs/eval_hanoi/arch_roboflow/test/`.

| Tập | COCO @1280 | v3 | v5 |
|---|---|---|---|
| Val UVH-26 (509 ảnh, 4 lớp; bỏ ảnh cùng camera với train) | 0.594 [0.573–0.619] | 0.335 [0.314–0.361] | 0.850 [0.832–0.868] |
| UA-DETRAC đêm (xe máy + ô tô + buýt) | 0.689 [0.654–0.739] | 0.722 [0.668–0.774] | 0.845 [0.797–0.890] |
| UA-DETRAC ngày | 0.690 [0.661–0.753] | 0.709 [0.649–0.764] | 0.812 [0.755–0.870] |
| Test Roboflow `vnvehicle` (308 ảnh, 4 lớp) | 0.255 | — | 0.834 |

UA-DETRAC đêm: 51 xe máy / 1.578 ô tô / 143 buýt / 21 tải; ngày: 35 / 4.003 / 246 / 19.
Đọc cẩn thận: v5 đã học 2.000 ảnh UVH-26 (khác camera) → UVH đo tổng quát sang camera mới cùng miền.

### 4.5 Kiến trúc: YOLO26n (v5) / YOLO26s / RT-DETRv2-S — cùng dữ liệu v5, cùng đường chấm

Nguồn: `outputs/eval_hanoi/arch/{test,day,night}`, `arch_roboflow/test`, `arch_pixel_control/*`,
`outputs/eval_clip_detection/arch/`, `arch_pixel_control/`, `outputs/eval_uvh26/arch/`, `outputs/eval_ua_detrac/arch/`.

mAP50 chính:

| Tập | v5 (n) @960 | YOLO26s @960 | RT-DETRv2-S @960×960 | YOLO26s @1280 (cùng số điểm ảnh) |
|---|---|---|---|---|
| Test Hà Nội | 0.923 | 0.914 | 0.929 [0.921–0.940] | 0.915 |
| `vn_extra` ngày | 0.745 | 0.798 | **0.886** [0.855–0.914] | 0.875 [0.844–0.902] |
| `vn_extra` đêm | 0.628 | 0.713 | **0.816** [0.777–0.854] | 0.800 [0.765–0.832] |
| 20 frame PVD | 0.892 | 0.903 | 0.895 | 0.907 |
| Test Roboflow (4 lớp) | 0.834 | 0.862 | 0.874 | — |
| Val UVH-26 (4 lớp) | 0.850 | 0.859 | **0.894** [0.878–0.907] | 0.865 [0.847–0.881] |
| UA-DETRAC đêm | 0.845 | 0.876 | 0.887 | 0.862 |
| UA-DETRAC ngày | 0.812 | 0.885 | 0.924 [0.874–0.960] | 0.885 |

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
(`outputs/benchmark_speed/e2_arch_nogmc/results.csv`, đo lại 9/10/2026 với tracker
`botsort_tuned_nogmc`, khi chưa có bộ lọc box trùng thêm từ 15/10/2026 — chưa đo lại tốc độ với bộ lọc; thay bảng cũ
`e2_arch` đo khi còn bật GMC):

| Mô hình | Đầu vào | Tham số (M) | GFLOPs | Detector ms/frame [IQR] | Detector frame/s | Pipeline ms/frame | Pipeline frame/s |
|---|---|---|---|---|---|---|---|
| v5 PyTorch | 544×960 | 2.51 | 7.4 | 100.5 [88.4–109.4] | 9.9 | 133.7 | 7.5 |
| YOLO26s | 544×960 | 9.95 | 28.7 | 261.9 [254.8–267.5] | 3.8 | 285.3 | 3.5 |
| RT-DETRv2-S | 960×960 | 20.08 | 133.9 | 1526.2 [1498.6–1602.4] | 0.66 | — | — |
| YOLO26s @1280 | 736×1280 | 9.95 | 52.4 | 455.2 [445.4–472.2] | 2.2 | 493.0 | 2.0 |

Ngoài khâu phát hiện, pipeline tốn thêm ~30 ms/frame (lần đo cũ còn bật GMC: 90–100 ms). Tỉ lệ RT-DETRv2-S chậm hơn:
YOLO26s @960 5.8×, YOLO26s @1280 3.4×, v5 15×. Bảng cũ có dòng "v5 OpenVINO 62.8 ms / 10.9 frame/s": số đó thực ra
đo trên GPU Intel tích hợp (ultralytics để OpenVINO tự chọn thiết bị) — đã bỏ, thay bằng bảng OpenVINO dưới đây.

**GPU** Tesla T4 Kaggle, PyTorch FP32, video VnTraffic 1080p, tracker `botsort_tuned_nogmc`
(`outputs/kaggle/vtrack-gpu-benchmark-run/outputs/benchmark_speed/e2_gpu_t4/results.csv`):

| Mô hình | Detector ms (frame/s) | Pipeline ms (frame/s) |
|---|---|---|
| v5 | 13.2 (75.7) | 23.7 (42.2) |
| YOLO26s | 14.7 (68.0) | 23.5 (42.6) |
| RT-DETRv2-S | 62.5 (16.0) | — |
| YOLO26s @1280 | 24.3 (41.2) | 33.2 (30.2) |

Ghi chú:
- Số "10.8 / 10.9 frame/s" cũ của v5 OpenVINO là đo trên GPU tích hợp, không phải CPU (xem bảng OpenVINO).

**OpenVINO trên laptop** (10/10/2026; `outputs/benchmark_speed/e2_cpu_openvino/` và `e2_igpu_openvino/` — hai lần chạy
riêng, đặt `KMP_BLOCKTIME=0 OMP_WAIT_POLICY=PASSIVE`; độ chính xác: `outputs/eval_hanoi/openvino_device|openvino_rect/val`,
`outputs/eval_tracking/openvino_device_dev|openvino_rect_dev`). Tiêu chí chốt trước: mAP50 val và IDR dev chênh ≤ 0.005
là hoà, hoà thì chọn nhanh hơn.

| Mô hình, cấu hình | Thiết bị | Đầu vào | Phát hiện ms | Pipeline frame/s | mAP50 val | IDR dev | Đổi ID dev |
|---|---|---|---|---|---|---|---|
| v5 PyTorch FP32 | CPU | 544×960 | 117.9 | 7.5 | 0.919 | 0.827 | 61 |
| v5 OpenVINO FP32 | CPU | 960×960 | 112.4 | 7.9 | 0.916 | 0.813 | 75 |
| v5 OpenVINO INT8 | CPU | 960×960 | 72.1 | 11.4 | 0.917 | 0.813 | 78 |
| v5 OpenVINO INT8 | CPU | 544×960 | 33.7 | 18.8 | 0.920 | 0.816 | 71 |
| v5 OpenVINO FP32 | GPU tích hợp | 960×960 | 54.1 | 12.5 | 0.915 | 0.819 | 68 |
| v5 OpenVINO INT8 | GPU tích hợp | 960×960 | 60.6 | 12.0 | 0.916 | 0.814 | 82 |
| **v5 OpenVINO FP32 (demo)** | GPU tích hợp | 544×960 | **32.8** | **19.8** | 0.918 | 0.825 | 71 |
| YOLO26s PyTorch FP32 | CPU | 544×960 | 284.3 | 3.3 | 0.915 | 0.845 | 65 |
| YOLO26s OpenVINO INT8 | CPU | 544×960 | 103.8 | 8.2 | 0.914 | 0.851 | 61 |
| YOLO26s OpenVINO FP32 | GPU tích hợp | 544×960 | 84.5 | 9.4 | 0.915 | 0.848 | 63 |

- Không chỉ định thiết bị thì OpenVINO (qua ultralytics) đưa bản FP32 lên GPU tích hợp → phải ghi `#cpu`/`#gpu`.
- Luồng OpenMP của torch chờ quay vòng, tranh lõi với OpenVINO: v5 INT8 544×960 trên CPU ~90 → 34 ms khi tắt; PyTorch
  trên CPU chậm đi ~15–20% (117.9 so với 100.5 ms ở bảng trên). App và `infer_video.py` tắt sẵn.
- INT8 nhanh hơn FP32 ~1.6 lần trên CPU, không nhanh hơn trên GPU UHD 620; không giảm độ chính xác.
- Đầu vào 544×960 (khung 16:9) nhanh hơn bản vuông 1.6–2.1 lần và gần PyTorch hơn; chỉ dùng cho video 16:9.
- Chọn cho demo: v5 FP32 544×960 trên GPU tích hợp (hoà PyTorch trên cả hai tiêu chí). Máy chỉ có CPU: INT8 544×960
  (IDR 0.816 so với 0.827, đổi ID 71 so với 61 → đánh đổi, không phải hoà). YOLO26s INT8 544×960 trên CPU không đạt
  tiêu chí so với bản vuông (IDR 0.851 so với 0.861).
- Cả quy trình của web demo (tính cả vẽ và ghi video 1080p): ~7.3 frame/s với bản demo, 5.8 với PyTorch → khâu vẽ/ghi
  video là chậm nhất.
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
vạch); lượt buýt là Ford Transit bị đếm thành ô tô → lỗi thật duy nhất: 1 lượt sai lớp. YOLO26s: 71 ra = 71 thật,
sai số theo hướng 2 (2.7%).

**Ban đêm — hai clip hầm chui 60 s (`ham_chui_dem_2`, `_3`), người đếm tay** (`data/count_gt/ham_chui_dem_{2,3}/`,
`scripts/count_by_hand.py`; chấm: `outputs/eval_manual_counts/night_human/`; v5 chạy CPU ra `outputs/manual_count_runs/v5/`,
YOLO26s = video demo GPU). Clip 1 bỏ vì trùng clip 2 và có cắt cảnh. 176 lượt, tất cả chiều "vào": 149 xe máy / 25 ô
tô (gồm 2 xe bán tải — quy ước bán tải 5 chỗ = ô tô, người làm đồ án chốt 12/10/2026, lệch COCO) / 1 buýt / 1 tải.
Kiểm độ tin cậy: một lần đếm độc lập thứ hai trên ảnh gốc (`data/count_gt/ai_review_2026-10-11/`) ghép được 174/176
lượt (lệch thời điểm trung vị 0.2 s), khác lớp đúng 2 xe bán tải; 2 xe máy cuối clip 3 lần đếm thứ hai chưa chốt — phóng
to frame 1782–1799 xác nhận đã qua vạch. (Báo cáo gọi là "đếm thêm một lần độc lập", không nêu tên trợ lý AI — người làm
đồ án chọn 15/10/2026.)

| Clip | Thật | v5 đếm / sót / thừa | YOLO26s đếm / sót / thừa |
|---|---|---|---|
| Hầm chui 2 | 79 | 72 / 7 / 0 | 76 / 3 / 0 |
| Hầm chui 3 | 97 | 89 / 8 / 0 | 98 (95 vào + 3 ra) / 4 / 5 |
| Tổng | 176 | 161 / 15 / 0 → sai số theo hướng 15 (8.5%) | 174 / 7 / 5 → 8 (4.5%) |

Mọi lỗi là xe máy; ô tô/buýt/tải đúng cả số lẫn lớp. Lượt sót tập trung ở cụm xe: 14/15 lượt sót của v5 và 7/7 của
YOLO26s có xe máy khác qua vạch trong ±0.3 s (tỉ lệ nền ở mọi lượt: 55%). 5 lượt thừa của YOLO26s: 2 xe có box dao
động quanh vạch (vào–ra–vào) + 1 xe ở frame cuối bị đếm "ra". Hướng sửa (chưa làm): chỉ tính lượt khi xe ở hẳn bên kia
vạch vài frame (chọn trên dev).

### 4.12 Cảnh đêm không nhãn (`outputs/unlabeled_analysis/night/`)

3 clip hầm chui, 20 s mỗi clip, mốc ngày = clip PVD: độ tin cậy trung vị v5 0.59–0.63 (đêm) vs 0.63 (ngày); tỉ lệ track
< 1 s 47–52% vs 54% → v5 gần như không suy giảm; v3 tụt rõ (0.36–0.45; 67–73% track ngắn). Chỉ là chỉ số thay thế;
không đưa vào báo cáo (16/10/2026) vì đã có tập đêm nhãn người (`vn_extra` đêm, đếm tay hầm chui).

### 4.13 Video demo GPU (`outputs/kaggle/vtrack-demo-gpu/demo/`) — bản cũ, YOLO26s học dữ liệu v5

Từ 15/10/2026 video demo dùng YOLO26s-v6, xử lý sẵn trên máy (mục 4.17); số dưới đây là của bản cũ.

YOLO26s @960 + `botsort_tuned_nogmc`, 6 clip × 60 s 1080p; lượt qua vạch: pvd_goc_a 194 (chỉ chiều ra), pvd_goc_b
169, pvd_goc_c 254, hầm chui 1/2/3: 69 / 76 / 98 (chưa có nhãn — không phải số đánh giá).

Hai lỗi thấy khi xem lại (đưa vào mục video demo của chương 4, hình `figures/demo_tow_truck.jpg`):
- Xe cứu hộ chở ô tô (`ham_chui_dem_1`, giây 47.5): cabin bị gọi buýt, box chỉ ôm cabin; ô tô trên sàn nhận riêng là
  car. Đáy box (chỉ ôm cabin) qua vạch ở giây 47.5 nhưng bánh xe thật còn dưới vạch tới chỗ cắt cảnh 48.3 s → đếm
  thừa 1 lượt, lại sai lớp (buýt thay vì tải); ô tô trên sàn không bị đếm thêm.
- Video gốc hầm chui có đoạn lặp (phát hiện 11/10/2026): `ham_chui_dem_1` frame 0–1387 = `ham_chui_dem_2` frame
  412–1799; clip 1 cắt cảnh ở frame 1448. Ba clip demo đêm thực ra chỉ gồm hai đoạn giao thông khác nhau; tập test
  `vn_extra` đêm (clip 2 và 3) không bị trùng.
- Xe buýt #103 (`ham_chui_dem_3`): đếm đúng là buýt ở frame 236; khi đi xa chỉ thấy đuôi, mô hình gọi chắc "truck"
  (141 lần, conf TB 0.74 so với 55 lần buýt, 0.52) → nhãn hiển thị đổi từ frame ~326. Đã khoá lớp hiển thị sau khi đếm.

### 4.15 Ba seed của YOLO26n (v5) và YOLO26s (độ dao động giữa các lần huấn luyện)

v5: Kaggle 12/10/2026 (`notebooks/kaggle_train`, `SEEDS = [1, 2]`; seed 1 dừng sớm epoch 89, tốt nhất 69, 2.68 giờ T4;
seed 2 đủ 100 epoch, tốt nhất 93, 3.02 giờ). Chấm: `scripts/rerun_seeds.sh` -> thư mục `seeds`, `seeds_roboflow`,
`seeds_dev`, `seeds_test` trong `outputs/eval_*`. Trọng số `weights/yolo26n_vn_v5_seed{1,2}_best.pt`.

YOLO26s: Kaggle 12–13/10/2026 (cùng notebook, batch 8; seed 1 đủ 100 epoch, tốt nhất 81, 5.36 giờ; seed 2 dừng sớm epoch
80, tốt nhất 60, 4.31 giờ; val ultralytics mAP50 / 50-95 0.875 / 0.644 và 0.873 / 0.647; test Roboflow ultralytics
0.879 / 0.594 và 0.887 / 0.603). Chấm: `scripts/rerun_seeds_yolo26s.sh` -> thư mục `seeds_yolo26s*`. Trọng số
`weights/yolo26s_vn_v5_seed{1,2}_best.pt`.

| Tập, chỉ số | YOLO26n (v5) seed 42 / 1 / 2 | YOLO26s seed 42 / 1 / 2 | v3 |
|---|---|---|---|
| Val Hà Nội (dev) | 0.919 / 0.915 / 0.914 | 0.915 / 0.911 / 0.919 | 0.917 |
| Test Hà Nội | 0.923 / 0.918 / 0.918 | 0.914 / 0.918 / 0.919 | 0.921 |
| vn_extra ngày | 0.745 / 0.757 / 0.732 | 0.798 / 0.828 / 0.809 | 0.757 |
| vn_extra đêm | 0.628 / 0.638 / 0.510 | 0.713 / 0.725 / 0.821 | 0.437 |
| 20 frame PVD | 0.892 / 0.905 / 0.898 | 0.903 / 0.903 / 0.904 | 0.891 |
| Val UVH-26 | 0.850 / 0.847 / 0.838 | 0.859 / 0.866 / 0.871 | 0.335 |
| UA-DETRAC đêm | 0.845 / 0.846 / 0.833 | 0.876 / 0.864 / 0.837 | 0.722 |
| UA-DETRAC ngày | 0.812 / 0.831 / 0.792 | 0.885 / 0.885 / 0.884 | 0.709 |
| Test Roboflow (4 lớp, đường chấm dự án) | 0.834 / 0.844 / 0.837 | 0.862 / 0.865 / 0.862 | — |
| VnTraffic dev (không GMC): IDR | 0.827 / 0.836 / 0.861 | 0.845 / 0.826 / 0.854 | 0.731 |
| VnTraffic dev: AssA | 0.709 / 0.711 / 0.722 | 0.711 / 0.703 / 0.708 | 0.653 |
| VnTraffic dev: đổi ID | 61 / 97 / 95 | 65 / 71 / 62 | 130 |
| Clip PVD test: HOTA | 0.718 / 0.728 / 0.714 | 0.758 / 0.763 / 0.767 | — |
| Clip PVD test: MOTA | 0.813 / 0.793 / 0.813 | 0.841 / 0.864 / 0.857 | — |
| Clip PVD test: IDF1 | 0.880 / 0.870 / 0.885 | 0.904 / 0.908 / 0.905 | — |
| Clip PVD test: đổi ID | 26 / 37 / 23 | 31 / 19 / 30 | — |

Kết luận:
1. Các ưu thế lớn của v5 so với v3 (UVH, UA-DETRAC, IDR dev, vn_extra đêm) đứng vững ở cả 3 seed; ngày và Hà Nội hoà.
2. YOLO26s hơn YOLO26n ở MỌI seed trên vn_extra ngày (kém nhất 0.798 > tốt nhất 0.757), vn_extra đêm (0.713 > 0.638),
   UA-DETRAC ngày, UVH, test Roboflow, HOTA test PVD. Hà Nội, 20 frame PVD, UA-DETRAC đêm, IDR dev: xen kẽ (hoà).
3. RT-DETRv2-S (1 lần) so với khoảng của YOLO26s: ngày 0.886 > 0.798–0.828, UVH 0.894 > 0.859–0.871 -> vượt dao động;
   đêm 0.816 nằm trong 0.713–0.821 -> chưa phân biệt được. Cảnh đêm: cùng một mô hình dao động 0.11–0.13 giữa các seed.
4. Đổi ID dev 61 của v5 seed 42 là thuận lợi (seed khác 95–97).
Báo cáo: `tab:seeds` trong phần thảo luận chương 4 (viết lại 13/10/2026), kết luận 1 của mục so sánh kiến trúc, đoạn
"Trong họ YOLO", đoạn khoảng tin cậy theo khối, kết luận/hạn chế/hướng phát triển chương 4, chương 2, 3, mở đầu.

### 4.14 Khoảng tin cậy theo khối (`outputs/eval_hanoi/ci_block/`, `outputs/eval_ua_detrac/ci_block/`)

Khối 10 s liên tiếp trong mỗi clip (90% xe ở trong khung hình < 8.2 s trên nhãn PVD); UA-DETRAC mỗi video một cụm.
Test Hà Nội 6 khối, `vn_extra` mỗi tập 10 khối, UA-DETRAC 21 (đêm) / 39 (ngày) video. Clip PVD (2 khối) và UVH-26
(không biết camera) không tính.

| Tập | Mô hình | mAP50 | CI theo ảnh | CI theo khối |
|---|---|---|---|---|
| Test Hà Nội | v2 / v3 / v5 | 0.890 / 0.921 / 0.923 | [0.877–0.901] / [0.911–0.933] / [0.913–0.932] | [0.871–0.905] / [0.905–0.938] / [0.906–0.938] |
| `vn_extra` ngày | v5 | 0.745 | [0.699–0.787] | [0.697–0.790] |
| | YOLO26s / @1280 | 0.798 / 0.875 | [0.765–0.828] / [0.844–0.902] | [0.757–0.832] / [0.828–0.914] |
| | RT-DETRv2-S | 0.886 | [0.855–0.914] | [0.849–0.914] |
| `vn_extra` đêm | v3 / v5 | 0.437 / 0.628 | [0.384–0.489] / [0.575–0.681] | [0.369–0.496] / [0.555–0.684] |
| | YOLO26s / @1280 | 0.713 / 0.800 | [0.685–0.744] / [0.765–0.832] | [0.685–0.744] / [0.762–0.838] |
| | RT-DETRv2-S | 0.816 | [0.777–0.854] | [0.766–0.858] |
| UA-DETRAC đêm | COCO / v3 / v5 | 0.689 / 0.722 / 0.845 | [0.654–0.739] / [0.668–0.774] / [0.797–0.890] | [0.647–0.726] / [0.646–0.776] / [0.785–0.886] |
| UA-DETRAC ngày | COCO / v3 / v5 | 0.690 / 0.709 / 0.812 | [0.661–0.753] / [0.649–0.764] / [0.755–0.870] | [0.653–0.744] / [0.601–0.836] / [0.738–0.893] |

v6 (thêm 13/10/2026): Hà Nội 0.921 [0.909–0.932] / khối [0.902–0.937]; `vn_extra` ngày 0.760 [0.724–0.795] / [0.718–0.798];
`vn_extra` đêm 0.755 [0.708–0.795] / [0.687–0.802] (không chồng v5).

### 4.16 Mô hình v6 = v5 + 288 frame đêm — mặc định 13–15/10/2026, nay là bản nhẹ (mô hình chính: mục 4.17)

- Dữ liệu: 288 frame đêm từ 2 camera TP.HCM (Midnight Soundscape `-J2JUaNU0BY`, `3fUYYb509YM`; video ghép lặp, lấy 4 s/frame
  trên 10 đoạn 2 phút, bỏ frame gần trùng (chênh điểm ảnh < 2 ở 160×90) -> bỏ 12, còn 288). Dataset v6: train 5.002
  ảnh (frame đêm: 6.256 xe máy / 3.811 ô tô / 162 buýt / 148 tải), val/test như v5. `configs/label_frames_night.yaml`.
- Tập dev đêm (mới): `night_extra/val`, 32 frame nút giao Công viên Gia Định (Vietnam Naturally `P0qqLcfxQiw`, phút 40/50/58/64,
  15 s/frame), nhãn sơ bộ YOLO26x, người kiểm 13/10/2026: 359 xe máy / 162 ô tô / 3 buýt + 20 ignore; 200/524 xe < 32 px.
- Nhãn tự động frame đêm (`configs/auto_label_night_sahi.yaml`): thầy RT-DETRv2-S@960, YOLO26x@1280, v5@960, v5@1280,
  v5@960 cắt 2×2; TTA lật; WBF; keep_conf 0.20 (F1 cao nhất trên dev đêm). Đo trên dev đêm: mAP50 của nhãn tự động 0.800
  (không cắt ảnh 0.778; RT-DETRv2 đứng riêng 0.702), P 0.81 / R 0.63 / F1 0.712; xe ≥ 64 px giữ ~95%, < 32 px ~40%.
- Train (Kaggle kernel riêng `vn-vehicles-yolo26-train-v6`): seed 42 tốt nhất epoch 73/93, val 0.868 / 0.629, Roboflow
  0.868 / 0.584, 3.04 giờ; seed 1 85/100, 0.862 / 0.626, 0.858 / 0.587, 3.27 giờ.
- Tiêu chí nhận (chốt 12/10/2026, trước khi train) và kết quả: (1) dev đêm cả hai seed > 0.499 (v5 tốt nhất): 0.599 /
  0.618 ĐẠT; (2) val Hà Nội ≥ 0.909: 0.914 / 0.914 ĐẠT; (3) IDR VnTraffic ≥ 0.822: 0.841 / 0.824 ĐẠT.
  v6 = seed 42 (mặc định tới 15/10/2026, nay là bản nhẹ).
- Test (chấm một lần, `scripts/rerun_v6.sh`, seed 42 / seed 1): xem bảng seeds trong báo cáo (`tab:seeds`). Đêm 0.755 /
  0.747 (v5 0.510–0.638), mAP50-95 đêm 0.495 (v5 0.404); recall đêm ≥ 64 px 0.91 (v5 0.74). Ngày ngang v5.
- Tracking test PVD với tracker MẶC ĐỊNH HIỆN TẠI (BoT-SORT đã chỉnh, không GMC, lọc box trùng IoU 0.6, từ 15/10/2026;
  `outputs/eval_tracking/nms06_v6_test/`): HOTA 0.744, DetA 0.701, AssA 0.806, MOTA 0.847, IDF1 0.905, IDR 0.859, IDP 0.955,
  đổi ID 26 (= v5), FN 1.704, FP 341; lọc track < 3 frame 0.734 / đổi ID 21; seed 1 HOTA 0.728, IDR 0.852, đổi ID 34.
  Chưa lọc box trùng (mặc định cũ, `v6_test`): HOTA 0.732, MOTA 0.835, IDF1 0.884, IDR 0.845, đổi ID 49 (v5 26), FN 1.689
  (v5 2.157), FP 505; seed 1 0.722 / đổi ID 38. Dev VnTraffic chưa lọc: IDR 0.841, AssA 0.698, đổi ID 93 (v5 61; seeds v5
  61–97), FN 717 (v5 1.058).
- Đếm xe (mặc định hiện tại): ngày 71/71 lượt ra, sai số 2 (2.7%) — bật/tắt bộ lọc như nhau; đêm sót 3, thừa 2, sai số 3
  (1.7%) (`outputs/eval_manual_counts/night_human_nms06/`; chưa lọc: thừa 3, sai số 4 = 2.3%). Seed 1: 8 = 4.5% (chưa lọc
  7 = 4.0%); v5 15 = 8.5%; YOLO26s 8 = 4.5%.
- OpenVINO v6 (demo): FP32 GPU 544×960 val 0.914, dev đêm 0.609, IDR 0.825 (PyTorch 0.841 -> đánh đổi); INT8 CPU 0.918 /
  0.597 / 0.825. Tốc độ như v5 (cùng kiến trúc, không đo lại).
- Giảm đổi ID cho v6 — ba lần thử (báo cáo mục `sec:giam-doi-id`, bảng `tab:giam-doi-id`):
  (a) 14/10: ngưỡng mở track 0.6 + ghép lần một 0.4 (`configs/botsort_v6.yaml`): dev VnTraffic đổi ID 93.5 -> 62.5, IDR
  0.833 -> 0.836, FN 714 -> 833; test đổi ID 49 -> 7 nhưng FN 1.689 -> 3.291, HOTA 0.732 -> 0.687, IDF1 0.884 -> 0.853,
  đếm đêm 2.3% -> 3.4% -> loại DỰA VÀO TEST (VnTraffic thiếu nhãn xe nhỏ ở xa); giữ làm tuỳ chọn "ít đổi ID" trong app.
  (b) 14/10: bộ lọc box trùng không phân lớp trước tracker (`vtrack/box_filter.py`, YOLO26 end-to-end không có NMS):
  IoU 0.6 dev đổi ID 63, FN +1.4% nhưng mAP50 dev đêm −0.006 (điều kiện thay thế, giới hạn 0.005) -> không đạt
  (`configs/tracker_tuning_v6_nms.yaml`), không chấm test.
  (c) 15/10 — ĐƯỢC NHẬN: chọn lại trên HAI tập dev (VnTraffic + clip dev mới `pvd_goc_c_dev`, nhãn đầy đủ xe nhỏ ở xa),
  16 ứng viên (12 tổ hợp ngưỡng + lọc box trùng IoU 0.7/0.6/0.5 + mốc), tiêu chí commit trước khi chạy (`bce7ea7`,
  `configs/tracker_tuning_v6_dev2.yaml`): so với mốc, trung bình 2 seed, FN dev mới ≤ +2%, HOTA dev mới ≥ −0.005, IDR
  VnTraffic ≥ −0.005; chọn tổng đổi ID ít nhất, đổi chỉ khi giảm ≥ 10%. Bỏ điều kiện mAP thay thế của (b) vì bỏ sót giờ
  đo trực tiếp (quyết định khi đã biết kết quả mAP của (b) -> ghi rõ trong báo cáo). Kết quả
  (`outputs/eval_tracking/tuning_v6_dev2/selection.csv`): ngưỡng mở track 0.6 -> FN +39–43% (dev mới thấy cái giá test
  từng thấy); ngưỡng 0.4 -> +9–16%; IoU 0.5 -> +2.5%; IoU 0.7 trượt IDR; ĐẠT: mốc, match 0.9 (đổi ID −1%), IoU 0.6 (tổng
  đổi ID 130 -> 90.5 = −30%, FN +0.9%, HOTA dev mới 0.689 -> 0.695, IDR VnT 0.833 -> 0.841) -> mặc định mới
  (`vtrack.tracking.DEFAULT_AGNOSTIC_NMS_IOU = 0.6`). Test chấm một lần (lần thứ ba test dùng cho tracker v6,
  `scripts/rerun_tracker_v6_dev2_test.sh`): xem dòng tracking/đếm ở trên. Cái giá phát hiện: mAP50 dev đêm −0.006, val
  Hà Nội không đổi; các bảng mAP trong báo cáo là mô hình KHÔNG có bộ lọc. OpenVINO demo + bộ lọc: VnTraffic IDR 0.825 ->
  0.840, đổi ID 85 -> 72; dev mới HOTA 0.694 -> 0.698, đổi ID 30 -> 27.
- Clip dev tracking `pvd_goc_c_dev` (14–15/10/2026): PVD góc C 1:48:00–1:48:20 (cùng góc với nửa `vn_extra` ngày, khác
  góc clip test), 600 frame, 96 track (80 ngoài vùng ignore y < 637), 11.071 box xe máy / 5.649 ô tô. Nhãn sơ bộ YOLO26s
  v5; người gán trên CVAT; trợ lý AI soát/sửa qua API CVAT (lấp chỗ hở, gộp/tách track, khôi phục 2 track bị xoá nhầm, tách
  box hai xe chạy song song bằng mắt, 8 xe đường bên trái từ box YOLO26x COCO đã duyệt) — nhật ký
  `data/tracking_gt/pvd_goc_c_dev/cvat_backup/edit_log.json`. Báo cáo mô tả là "kiểm tra lại bằng script đọc nhãn từ
  CVAT, xem ảnh từng chỗ trước khi sửa; YOLO26x COCO dò chỗ sót và đề xuất box cho 8 xe, duyệt bằng mắt" (người làm đồ án
  chọn cách diễn đạt này 15/10/2026).
- Lưu ý khi bảo vệ: hướng làm v6 và việc chọn mô hình thầy xuất phát từ kết quả trên test đêm `vn_extra`; quyết định nhận
  dựa trên dev đêm ở camera thứ ba, test chấm sau. Đã ghi trong mục "Những lần tập test được dùng lại".

### 4.17 YOLO26s-v6 — MÔ HÌNH CHÍNH từ 15/10/2026 (v6 thành bản nhẹ)

- Khung trình bày (người làm đồ án chốt 15/10/2026): so sánh ba kiến trúc trả lời "mô hình nào hợp bối cảnh đề tài" —
  camera giao thông đưa hình về máy chủ có GPU — chứ không phải "mô hình nào chạy được trên laptop". Trên T4, cả pipeline
  YOLO26s 43 frame/s ≈ YOLO26n 42; YOLO26s chính xác hơn ở mọi seed trên camera mới; RT-DETRv2-S 16 frame/s riêng detector,
  chỉ hơn YOLO26s cùng số điểm ảnh ở độ khít box. Nhưng YOLO26s của thí nghiệm kiến trúc chỉ học dữ liệu v5 -> huấn luyện
  YOLO26s trên dữ liệu v6. v6 (YOLO26n) giữ vai trò bản nhẹ cho laptop / thiết bị tại camera.
- Train (Kaggle, `notebooks/kaggle_train/train.ipynb`, MODEL `yolo26s.pt`, batch 8, cấu hình y hệt YOLO26s v5): seed 42
  dừng sớm epoch 97, tốt nhất 77, val ultralytics 0.871 / 0.645, Roboflow 0.890 / 0.601, 5.16 giờ T4; seed 1 dừng 78, tốt
  nhất 58, 0.885 / 0.648, 0.880 / 0.592, 3.99 giờ. Trọng số `weights/yolo26s_vn_v6_best.pt`, `_seed1_best.pt`; OpenVINO
  `yolo26s_vn_v6_best_544x960_openvino_model` (GPU tích hợp) chấm trên dev (`scripts/rerun_openvino_yolo26s_v6.sh`, `outputs/eval_hanoi/openvino_yolo26s_v6{,_nightdev}/`, `outputs/eval_tracking/openvino_yolo26s_v6_*`): mAP50 val HN 0.912 vs PyTorch 0.911, dev đêm 0.710 vs 0.711 (hoà); VnTraffic không lọc IDR 0.875 vs 0.866, đổi ID 57 vs 63; có lọc 0.6 (mặc định) IDR 0.860 vs 0.870, đổi ID 66 vs 62 (kém 0.010 = đánh đổi, như v6); dev PVD góc C có lọc HOTA 0.754 vs 0.760, đổi ID 15 vs 18; báo cáo: dòng YOLO26s-v6 trong `tab:openvino` + đoạn sau bảng.
- Tiêu chí nhận (commit trước khi train, `configs/accept_yolo26s_v6.yaml`; chấm `scripts/rerun_yolo26s_v6_dev.sh`, kiểm
  `scripts/check_accept_yolo26s_v6.py`), mỗi điều kiện vượt cả YOLO26s v5 lẫn v6: (1) dev đêm cả hai seed > 0.641 (YOLO26s
  v5 tốt nhất; v6 tốt nhất 0.618): 0.711 / 0.712 ĐẠT; (2) val Hà Nội ≥ 0.909: 0.911 / 0.916 ĐẠT; (3) IDR VnTraffic ≥ 0.821
  (YOLO26s v5 kém nhất − 0.005): 0.866 / 0.870 ĐẠT. Mô hình chính = seed 42.
- Bộ lọc box trùng chọn lại theo đúng luật của mục 4.16 (c), ứng viên không lọc / IoU 0.7 / 0.6 / 0.5: vẫn IoU 0.6 (tổng
  đổi ID hai dev 88.5 -> 70.5 = −20%, FN dev mới +1.2%, HOTA dev mới 0.748 -> 0.751); IoU 0.5 loại vì IDR VnTraffic.
- Dev VnTraffic (chưa lọc, `tab:detector-tracking`): HOTA 0.565, AssA 0.714, IDR 0.866, đổi ID 63, FN 632, 327 track.
- Test (chấm một lần, `scripts/rerun_yolo26s_v6_test.sh`, thư mục `yolo26s_v6*`), mAP50 chính seed 42 / 1: Hà Nội 0.920
  [0.911–0.931] / 0.927; `vn_extra` ngày 0.830 [0.801–0.858] / 0.841 (mAP50-95 0.545), đêm 0.887 [0.861–0.913] / 0.847
  (mAP50-95 0.614; khối [0.851–0.915]); 20 frame PVD 0.901 / 0.914; UVH 0.871 / 0.875; UA-DETRAC đêm 0.867 / 0.872, ngày
  0.881 / 0.881; Roboflow 0.857 / 0.848. Recall theo chiều cao — ngày <20 / 20–32 / 32–64 / ≥64 px: 0.51 / 0.68 / 0.89 /
  0.93; đêm 20–32 / 32–64 / ≥64 px: 0.62 / 0.79 / 0.94.
- Tracking test PVD (`outputs/eval_tracking/yolo26s_v6_test/`, BoT-SORT đã chỉnh, không GMC, lọc box trùng 0.6): HOTA
  0.755, DetA 0.717, AssA 0.811, MOTA 0.874, IDF1 0.916, IDR 0.882, IDP 0.953, đổi ID 22, FN 1.353, FP 340; lọc track < 3
  frame: 0.745 / đổi ID 17; seed 1 HOTA 0.769.
- Đếm: ngày (`outputs/eval_counting/yolo26s_v6_*`) 70 lượt ra / 71 thật, sai số theo hướng 3 (4.1%; seed 1: 2 = 2.7%) —
  1 xe máy sót + Transit thành ô tô (2 lượt "vào" của nhãn là hiện tượng tán cây như mục 4.11); đêm
  (`outputs/eval_manual_counts/night_human_yolo26s_v6/`) clip 2 đúng 79/79, clip 3 sai số 3 -> 3 = 1.7% (seed 1 1.1%).
- Tốc độ (đo lại trên laptop 16/10/2026, `scripts/benchmark_speed.py --agnostic-nms-ious none 0.6`, máy cắm sạc,
  `KMP_BLOCKTIME=0 OMP_WAIT_POLICY=PASSIVE`, nhóm CPU và GPU tích hợp chạy riêng; `outputs/benchmark_speed/e2_v6_cpu/`,
  `e2_v6_igpu/`) — phát hiện ms / pipeline ms không lọc / có lọc 0.6:
  YOLO26s-v6 PyTorch CPU 338.0 / 309.0 / 318.2; YOLO26s học v5 (đối chiếu, cùng lần) 302.8 / 290.0 / 325.4 (trung vị
  từng đợt 283–425 ms -> hai bản cùng tốc độ trong mức nhiễu); v6 PyTorch CPU 130.0 / 151.4 / 149.7; v6 INT8 CPU 544×960
  39.9 / 54.9 / 55.6; v6 OV FP32 GPU tích hợp 544×960 34.7 / 51.3 / 57.4 (19.5 / 17.4 frame/s); YOLO26s-v6 OV GPU tích hợp
  80.1 / 101.8 / 105.7 (9.8 / 9.5 frame/s). Bộ lọc box trùng: +6.1 / +3.9 ms trên GPU tích hợp, trên CPU không nhất quán
  (−1.7 đến +35 ms, trong nhiễu); đo riêng hàm lọc trên box thật (~45 box/frame) khoảng 4 ms/frame (kiểm tạm, không đưa vào
  báo cáo). Phần phát hiện đo xen kẽ, pipeline đo liền một mạch nên pipeline của YOLO26s-v6 CPU thấp hơn phát hiện.
  T4: chưa đo lại (cùng kiến trúc với YOLO26s v5: pipeline 43 frame/s).
- Video demo (15/10/2026, hết quota GPU Kaggle tuần đó): xử lý sẵn trên máy bằng `scripts/make_demo_videos.sh` ->
  `outputs/demo_precomputed/` (YOLO26s-v6 PyTorch CPU, bộ lọc box trùng 0.6, lọc tuổi track 3, vạch theo
  `configs/demo_clips.yaml`), 2.57–2.80 frame/s cả quy trình (~11–12 phút/clip). Lượt qua vạch 60 s: pvd_goc_a 194 (1 vào /
  193 ra), pvd_goc_b 180, pvd_goc_c 247, hầm chui 1/2/3: 70 / 79 / 96 (clip 2, 3 trùng số đã chấm với người đếm tay).
  Xe cứu hộ (hầm chui 1, frame 1429 = 47.6 s): VẪN lỗi như bản cũ — box chỉ ôm cabin, gọi "bus", đếm thừa 1; ô tô trên
  sàn (#243) không bị đếm. Xe buýt hầm chui 3 (track 212, hiển thị #72, đếm "bus" ở frame 236): YOLO26s-v6 gọi buýt 405 /
  427 lần (conf TB 0.85), tải 10, ô tô 12 -> lớp bỏ phiếu không đổi (kiểm bằng script tạm chạy lại 700 frame đầu, đúng cấu
  hình demo). App chế độ "Kết quả xử lý sẵn"; danh sách mô hình = YOLO26s-v6 (OpenVINO GPU tích hợp / PyTorch) + bản nhẹ
  v6 + COCO gốc; bỏ chọn tracker. Ảnh chụp app (`app_gpu_demo.jpg`, `app_counts.jpg`) người làm đồ án đã chụp lại
  15/10/2026; ảnh frame demo (`demo_pvd_goc_a.jpg`, `demo_ham_chui_dem_1.jpg`, `demo_tow_truck.jpg`) tạo lại từ video mới
  (`report/make_figures.py`, `DEMO_DIR = outputs/demo_precomputed`).
- So với RT-DETRv2-S (học dữ liệu v5): mAP50 `vn_extra` ngày thấp hơn (0.830 vs 0.886), đêm cao hơn (0.887 vs 0.816; khác
  dữ liệu nên không kết luận về kiến trúc). mAP50-95 trên tập nhãn độc lập RT-DETRv2-S vẫn khít nhất: UVH 0.757 vs 0.729,
  UA-DETRAC đêm 0.588 vs 0.556, ngày 0.599 vs 0.573 (`outputs/eval_uvh26/yolo26s_v6/`, `outputs/eval_ua_detrac/yolo26s_v6/`).
  So với YOLO26s học v5: ngang ở hầu hết tập, test Roboflow thấp hơn một chút (0.848–0.857 vs 0.862–0.865).
- Đếm đêm: 3 lượt sót = 2 xe máy mà v6 và seed 1 cũng sót (giây 24.43, 39.4 clip 3) + 1 xe máy ở frame cuối clip (59.97 s).
- Hạn chế thêm: bối cảnh máy chủ GPU mới đo một luồng video trên T4 (FP32, chưa TensorRT/FP16), chưa đo nhiều camera cùng lúc.
- Lưu ý khi bảo vệ: hướng làm YOLO26s-v6 có tham khảo kết quả test của YOLO26s v5 (tốt ban ngày) và v6 (tốt ban đêm);
  quyết định nhận dựa trên dev, test chấm một lần sau đó (đã ghi trong mục "Những lần tập test được dùng lại"). Chỉ 2 seed;
  đêm chênh 0.040 giữa hai lần.
- Báo cáo: mở đầu, chương 1 ("Thứ sáu, yêu cầu thời gian thực và giới hạn tài nguyên"), chương 3 (`tab:lich-su-train`,
  `sec:tieu-chi-chon`, web demo), chương 4 (mọi bảng chính thêm cột/dòng YOLO26s-v6, `tab:seeds`, thảo luận, kết luận),
  sơ đồ pipeline.

## 5. Danh mục hình

| Mã đề xuất | Nội dung | File | Trạng thái |
|---|---|---|---|
| H1.1 | Frame demo có box, ID, vạch đếm | trích từ `outputs/kaggle/vtrack-demo-gpu/demo/pvd_goc_a_tracked.mp4` | **cần trích** |
| H1.2 | Ví dụ khó khăn: cụm xe máy dày, xe nhỏ ở xa, ban đêm | `outputs/error_analysis/2_…`, `1_…`, `3_…` (dùng lại) hoặc frame gốc | có sẵn |
| H2.1–H2.8 | Hình gốc từ bài báo, ghi nguồn trong chú thích (chú thích ngắn cho danh mục hình không có `\cite` để không đảo thứ tự tài liệu IEEE): lưới S×S của YOLO (arXiv 1506.02640), kiến trúc / các khối / độ chính xác–độ trễ T4 của YOLO26 (2606.03748, CC BY 4.0), DETR (2005.12872, CC0), RT-DETR (2304.08069), ByteTrack (2110.06864), BoT-SORT (2206.14651); các bài không ghi CC dùng giấy phép mặc định arXiv → trích dẫn cho học tập, nghiên cứu | `report/overleaf/figures/paper_*`, tạo bằng `report/fetch_paper_figures.py` (cần `pypdfium2`; mã nguồn arXiv lưu ở `outputs/paper_sources/`) | có (15/10/2026) |
| H3.1 | Sơ đồ pipeline hệ thống | — | **cần vẽ** |
| H3.2 | Ví dụ ảnh từ mỗi nguồn dữ liệu | `data/raw/...`, `data/labeling/...` | **cần chọn** |
| H3.3 | Giao diện gán nhãn X-AnyLabeling / CVAT | — | **cần chụp** |
| H3.4 | Đường học v5 | `outputs/kaggle/vn-vehicles-yolo26-train/runs/yolo26n_v5_e100_960/results.png` | có sẵn |
| H3.5 | Đường học RT-DETRv2 | từ `rtdetrv2_r18_v5_e100_960_epochs.csv` | **cần vẽ** |
| H4.1 | Ma trận nhầm lẫn YOLO26s-v6 trên test Roboflow (thay bản v5, 16/10/2026) | `outputs/kaggle/vn-vehicles-yolo26s-train-v6/runs/yolo26s_v6_e100_960/test/confusion_matrix_normalized.png` | đã đưa vào |
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
| Mô hình triển khai: v5 OpenVINO 544×960 trên GPU Intel tích hợp (INT8 trên CPU nếu không có GPU Intel); YOLO26s nếu có GPU rời | Tiêu chí chốt trước (hoà → nhanh hơn) | RT-DETRv2 chậm 15× so với v5 trên CPU; trên GPU T4 n và s nhanh như nhau |
| Bootstrap thêm theo khối 10 s / theo video | Block bootstrap (Künsch 1989), cluster bootstrap (Field & Welsh 2007) | Frame cùng clip tương quan; 90% xe ở trong khung hình < 8.2 s |
| Ảnh hiệu chỉnh INT8 lấy từ train, không từ val | Khuyến nghị NNCF ~300 ảnh đại diện | Val Hà Nội là tập dev dùng để quyết định |

## 7. Danh mục từ viết tắt

Bản đầy đủ (45 mục) ở `report/overleaf/frontmatter/viettat.tex`; 16/10/2026 thêm AIFI, BCE, CCFF, COCO, CVAT, DFL, dHash,
FP16, GIoU, KITTI, SPPF, STAL, VFL (tên đầy đủ lấy từ bài báo gốc). Bảng dưới là bản tổng hợp ban đầu.

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
29a. Intel, "Neural Network Compression Framework (NNCF), phiên bản 3.4.0." https://github.com/openvinotoolkit/nncf
30. Roboflow Universe datasets (CC BY 4.0): "vietnamese-vehicle" (car-classification, v3); "vehicle-vietnam-cantho"
    (vehicle, v19); "vehicle-detection" (skripsi-1qzlz, v2). URL ở mục 9.
31. P. Skalos, "UA-DETRAC fix masks two-wheelers," Kaggle dataset, CC BY-NC-SA 3.0. **[cần kiểm tra tên đầy đủ tác
    giả và năm]**
32. B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*. New York: Chapman & Hall, 1993.
33. H. R. Künsch, "The jackknife and the bootstrap for general stationary observations," *Ann. Statist.*, vol. 17,
    no. 3, pp. 1217–1241, 1989, doi: 10.1214/aos/1176347265.
34. C. A. Field and A. H. Welsh, "Bootstrapping clustered data," *J. R. Stat. Soc. B*, vol. 69, no. 3, pp. 369–390,
    2007, doi: 10.1111/j.1467-9868.2007.00593.x.

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

- Chỉ v5 và YOLO26s train 3 seed (mục 4.15); các cấu hình khác 1 seed (RT-DETRv2 ban đêm chưa so được với YOLO26s); bootstrap chỉ đo dao động do mẫu ảnh test; bootstrap theo khối giả định khối
  10 s độc lập; test Hà Nội chỉ 6 khối; PVD và UVH-26 không tính được khoảng theo khối.
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
- Trên laptop phát hiện + theo dõi ~20 frame/s (chưa bằng 30 frame/s của video), cả quy trình tính vẽ và ghi video
  ~7 frame/s; RT-DETRv2 chưa export OpenVINO; GPU T4 mới đo FP32, chưa TensorRT.
- Tracker chỉnh với v3 rồi dùng cho v5.

## 11. Lệch giữa tài liệu nội bộ và file kết quả (đã kiểm khi tổng hợp)

- ghi chú dự án ghi số box `vn_extra` đêm là 465 / 159 / 7 / 8 — đó là **trước** khi sửa nhãn 8/10/2026; file kết quả hiện
  tại là **468 / 157 / 6 / 9**. Báo cáo dùng số sau.
- Số "pipeline OpenVINO 10.8 / 10.9 frame/s" (README, ghi chú dự án, bản báo cáo trước) là đo trên GPU tích hợp, không
  phải CPU; đã thay bằng bảng OpenVINO theo thiết bị (mục 4.8).
- Bảng phân tích recall theo kích thước xe (mục 4.2) giờ có script và file kết quả (`outputs/size_recall/`).
- ghi chú dự án ghi tốc độ RT-DETRv2-S trên CPU là "1.561 [1.442–1.719] ms" — đúng ra là **giây** (1561 ms/frame).
