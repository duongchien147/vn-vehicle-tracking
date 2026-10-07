# Giao thức so sánh kiến trúc: YOLO26 (CNN) và RT-DETRv2 (Transformer)

**Câu hỏi:** họ Transformer có phát hiện xe Việt Nam tốt hơn họ CNN không, và phải trả giá bao nhiêu về tốc độ CPU?

**Nguyên tắc:** đồng nhất mọi thứ đồng nhất được **mà không làm bên nào bị train sai cách**. Thứ gì gắn liền với
kiến trúc hoặc với công thức tối ưu của từng họ thì để mỗi bên theo bản chính thức, và ghi rõ ở mục 3.
- YOLO26: mặc định của ultralytics 8.4.163, file `notebooks/kaggle_train/train.ipynb`.
- RT-DETRv2-S: cấu hình chính thức `rtdetrv2_r18vd_120e_coco.yml` của tác giả (github.com/lyuwenyu/RT-DETR), file
  `notebooks/kaggle_train_rtdetr/train_rtdetr.py`, mô hình qua transformers 5.18 (`PekingU/rtdetr_v2_r18vd`).

Các khẳng định về hành vi thư viện dưới đây đều đã đối chiếu mã nguồn (ghi file:dòng của ultralytics 8.4.163).

## 1. Đồng nhất tuyệt đối

| Yếu tố | Cách giữ đồng nhất |
|---|---|
| Ảnh train/val/test | Cùng phiên bản dataset Kaggle `chienduong147/vn-vehicles-yolo` (v5: 4.714 train, 320 val, 308 test), cùng thư mục. |
| Nhãn | Cùng file nhãn YOLO. Thứ tự lớp đọc từ chính `data.yaml` (không viết cứng). Ảnh nền (2 ảnh train không có xe) giữ ở cả hai. |
| Pretrain | Cả hai khởi tạo từ trọng số pretrained trên COCO; đầu phân lớp khởi tạo lại cho 4 lớp. |
| imgsz | 960 (cách đưa ảnh về 960 khác nhau: mục 3) |
| Số epoch tối đa, dừng sớm | 100 epoch; dừng khi mAP50-95 val không cải thiện sau 20 epoch. RT-DETRv2 cần >1 phiên Kaggle (~70 epoch/11 giờ) nên lưu checkpoint mỗi epoch và train tiếp đúng chỗ dừng (cả trạng thái optimizer, EMA, AMP, ngẫu nhiên). |
| Tập val | Cùng một tập val dùng cho cả dừng sớm lẫn chọn checkpoint, ở cả hai (như mặc định ultralytics). Chấp nhận được vì không dùng val để chỉnh siêu tham số. |
| Chọn checkpoint | Trọng số **EMA** có **mAP50-95 val** cao nhất. Ultralytics: fitness = mAP50-95 (`utils/metrics.py:1011`). |
| EMA | **Đã đối chiếu mã nguồn, cùng công thức:** RT-DETRv2 chính thức (`rtdetrv2_pytorch/src/optim/ema.py`) và ultralytics (`utils/torch_utils.py:857`) đều cập nhật `v = d·v + (1−d)·w` cho tensor số thực, với `d = 0.9999·(1 − exp(−updates/2000))`. Script RT-DETRv2 chép đúng công thức này. |
| Họ optimizer | Cả hai là **AdamW**. Với dataset này `optimizer=auto` của ultralytics chọn AdamW vì số vòng lặp `ceil(4714/max(batch, nbs=64))·100 = 7.400 < 10.000` (`engine/trainer.py:315, 1156`). |
| Độ chính xác số | Mixed precision FP16 (AMP) |
| Seed, chế độ tất định | Seed 42; `use_deterministic_algorithms(True, warn_only=True)`, `cudnn.deterministic`, `CUBLAS_WORKSPACE_CONFIG`, `PYTHONHASHSEED`, giống hệt `init_seeds` của ultralytics (`utils/torch_utils.py:795`). Cả hai **không** tất định tuyệt đối: một số phép GPU không có bản tất định nên chỉ cảnh báo. Với RT-DETRv2 là backward của `grid_sample` trong deformable attention (log Kaggle có cảnh báo). Ghép cặp Hungarian chạy trên CPU (scipy) nên vẫn tất định. |
| DataLoader | 4 worker ở cả hai. |
| GPU train | T4 (Kaggle) |
| Số phát hiện tối đa/ảnh | 300 (`max_det` của ultralytics = top-300 của RT-DETR). Cả hai đều không dùng NMS. |
| TTA | Tắt ở cả hai, cả khi chấm lẫn khi đo tốc độ. |
| Bộ chấm | Chấm lại trên máy local bằng **một đường chấm duy nhất** (`vtrack.detectors.load_detector` → `detection_benchmark.score_model`): pycocotools, conf 0.001, tối đa 500 phát hiện/ảnh, cùng vùng ignore, cùng lớp chính, cùng bootstrap. Báo **mAP50, mAP75, mAP50-95**. |
| Tập chấm | Như nhau cho mọi mô hình: test Roboflow, test Hà Nội, `vn_extra` ngày/đêm, 20 frame PVD, val UVH-26, UA-DETRAC đêm/ngày. |
| Đo tốc độ | `scripts/benchmark_speed.py` (chi tiết ở mục 4), cùng CPU, cùng frame, đo qua cùng giao diện `Detector` (tiền xử lý + suy luận + hậu xử lý). |
| Đo độ phức tạp | Một công cụ cho cả hai: `torch.utils.flop_counter` của PyTorch (`vtrack/complexity.py`), batch 1, một lần truyền xuôi (backbone + neck + head, không gồm tiền/hậu xử lý), ở **đúng đầu vào thực tế**: YOLO 544×960 (letterbox video 1080p), RT-DETRv2 960×960. Quy ước 1 nhân-cộng = 2 FLOPs. Đối chiếu: YOLO26n đo được 7.4 GFLOPs ở 544×960, khớp 5.9 GFLOPs ở 640×640 mà ultralytics tự báo (×1.275). |

## 2. Đồng nhất có điều chỉnh: lệch bản chính thức RT-DETRv2 cho giống YOLO

| Yếu tố | Bản chính thức RT-DETRv2 | Dùng ở đây | Lý do và cái giá |
|---|---|---|---|
| imgsz | 640 | **960** | Cùng độ phân giải với YOLO26 v5. Kích thước 960 không làm sai cách train: anchor của RT-DETRv2 sinh động theo kích thước đầu vào (`anchor_image_size: null`). |
| Warmup lr | 2000 bước tối ưu | **3 epoch** (= `warmup_epochs` của ultralytics; tối thiểu 100 bước) | 2000 bước được đặt cho COCO, nơi nó chỉ là ~0.27 epoch (118k ảnh / batch 16 = 7.400 bước/epoch). Trên 4.714 ảnh (295 bước/epoch), 2000 bước lại thành **~6.8 epoch**. Đặt 3 epoch để cả hai dành cùng một phần quá trình train cho warmup. Đây là chỗ lệch bản chính thức. |
| Tắt augmentation mạnh ở cuối | 3 epoch cuối | **10 epoch cuối** (= `close_mosaic`) | Cùng cơ chế. Cả hai chỉ xảy ra nếu chưa bị dừng sớm. |
| Số epoch | 120 (từ ImageNet) | 100 + dừng sớm | Ngân sách giống YOLO. Đây là fine-tune từ COCO. |

## 3. Không thể đồng nhất (gắn với kiến trúc hoặc công thức tối ưu), ghi rõ trong báo cáo

| Yếu tố | YOLO26 | RT-DETRv2-S | Vì sao không đồng nhất |
|---|---|---|---|
| LR và lịch LR | lr 0.00125 (`lr_fit = 0.002·5/(4+nc)`, chỉ phụ thuộc số lớp, `trainer.py:1155`), giảm tuyến tính về 1% | lr 1e-4 cố định sau warmup, cắt gradient 0.1 | Transformer cần lr nhỏ và cắt gradient; lr của YOLO dễ làm DETR phân kỳ, còn lr 1e-4 quá nhỏ với YOLO. |
| Weight decay | 5e-4, nhân hệ số `batch·accumulate/nbs` (= 1 ở đây, `trainer.py:313`) | 1e-4; lớp norm = 0 | Theo công thức của mỗi họ. |
| **Batch hiệu dụng** | **64**: batch thật 16 (n) / 8 (s), cộng dồn gradient tới `nbs=64` (`trainer.py:312`) | **16**: 8 × cộng dồn 2 | Batch gắn chặt với LR của từng công thức. Ép YOLO về `nbs=16` thì số vòng lặp thành 29.500 > 10.000 và `optimizer=auto` **đổi sang MuSGD lr 0.01** (`trainer.py:1156`): YOLO đổi cả optimizer, số bước cập nhật (×4) và weight decay, tức là train sai công thức. Ngược lại, nâng RT-DETRv2 lên 64 mà giữ lr 1e-4 cũng lệch công thức đã kiểm chứng. Lưu ý khi đọc tài liệu khác: `batch=16` trong ultralytics vẫn có batch hiệu dụng 64. |
| Augmentation | Mosaic, HSV, lật, scale/translate | PhotometricDistort, ZoomOut, IoUCrop, lật | Mosaic là thành phần chính của công thức YOLO; công thức DETR chính thức không dùng mosaic. |
| Đưa ảnh về imgsz | Letterbox: giữ tỉ lệ, đệm viền; video 1080p → 544×960 | Kéo giãn thẳng về 960×960 | Quy ước của họ RT-DETR; ngay cả ultralytics khi chạy RT-DETR cũng kéo giãn (`models/rtdetr/predict.py:82`, `scale_fill=True`). |
| Loss phân lớp, mất cân bằng lớp | BCE; **không** dùng class weights hay focal (mặc định `cls_pw=0` tắt trọng số lớp, `models/yolo/detect/train.py:173`) | Varifocal loss (VFL) | Là một phần của kiến trúc/đầu ra; không họ nào dùng trọng số lớp. |
| Gán nhãn và loss box | Gán nhãn kiểu YOLO26 end-to-end | Ghép cặp Hungarian (từng ảnh), L1 + GIoU, denoising | Chính là kiến trúc đang được so sánh. |
| Bộ tính mAP val (chọn checkpoint) | Bộ chấm của ultralytics | pycocotools | Chỉ ảnh hưởng việc chọn epoch nào; mọi số trong báo cáo đều chấm lại bằng cùng một bộ chấm (mục 1). |

**Hệ quả phải nêu khi đọc kết quả:**
- **Số điểm ảnh.** Với video 16:9, YOLO nhìn 960×540 điểm ảnh có nội dung, RT-DETRv2 nhìn 960×960, tức nhiều hơn
  ~1.8 lần. Điều này có lợi cho xe nhỏ và chịu thiệt về tốc độ; bảng tốc độ đo trên đúng đầu vào này.
- **Cỡ mô hình.** RT-DETRv2-S là bản nhỏ nhất của họ nhưng lớn hơn YOLO26n/s. Vì vậy trình bày **đồ thị mAP –
  ms/frame**, kèm số tham số và GFLOPs, như bài báo RT-DETR làm. Không so một cặp số mAP rời rạc.
- **IoU chặt (mAP75, mAP50-95) có thể thiên vị họ CNN trên tập nhãn của dự án.** Nhãn tay của ta đều được sửa từ
  nhãn sơ bộ do **mô hình YOLO** vẽ (test Hà Nội từ v2, PVD từ v3, `vn_extra` từ YOLO26x). Box giữ nguyên mang phong
  cách khoanh box của họ YOLO. Vì vậy:
  - **so hai kiến trúc ở IoU chặt chỉ dùng UA-DETRAC và val UVH-26**: nhãn do người khác gán, không qua mô hình của dự án;
  - trên tập nhãn của dự án, chỉ dùng mAP50.

## 4. Đo tốc độ CPU (i5-8250U, laptop dòng U, dễ bị hạ xung khi nóng)

Cách đo (`vtrack/speed.py`):
- **5 đợt × 30 frame** 1080p, nghỉ **10 giây** giữa các đợt.
- Trong mỗi đợt, các mô hình chạy lần lượt trên **cùng 30 frame**, nghỉ **2 giây** khi chuyển mô hình. Thứ tự đảo chiều
  mỗi đợt (A B C, C B A, ...).
- **Khởi động** ít nhất 10 frame **và** 15 giây cho mỗi mô hình, cộng thêm 3 frame ở đầu mỗi lượt.
- Ghi thời gian **từng frame** (`timings.csv`) và trung vị từng đợt, để phát hiện xu hướng chậm dần.
- Báo **trung vị + IQR** (phân vị 25–75) và phân vị 90.
- Máy cắm sạc, chế độ nguồn Windows "Balanced", không chạy tiến trình nặng khác.

Vì sao cần từng chi tiết (đo thử 7/10/2026 với v5 PyTorch và OpenVINO):
- Khởi động cố định 10 frame là không đủ: OpenVINO còn chậm 265–782 ms ở ~8 frame tiếp theo rồi mới ổn định ở ~64 ms.
- Không nghỉ khi chuyển mô hình thì PyTorch chạy ngay sau OpenVINO bị chậm 120–163 ms/frame, so với ~90 ms khi có nghỉ,
  vì luồng của engine trước còn quay chờ và tranh CPU.
- Với cách đo mới: OpenVINO 64.3 ms (IQR 0.8), PyTorch 90.5 ms (IQR 20).
- **Số E2 cũ (v5 PyTorch 136 ms/frame) đo bằng cách cũ (liền một mạch, khởi động 10 frame) nên bị đội lên.** Phải đo
  lại toàn bộ bảng E2 bằng cách mới.

Pipeline tracking (đọc video + detector + tracker) cần frame liên tiếp nên vẫn đo một mạch, có nghỉ trước khi đo;
chỉ chạy được với YOLO.

## 5. Giới hạn

- Mỗi cấu hình train **1 lần (1 seed)**. Bootstrap chỉ đo dao động do mẫu ảnh test, không đo dao động giữa các lần
  train. Chênh lệch nằm trong khoảng tin cậy coi là hoà.
- Nếu RT-DETRv2 không train đủ tới điều kiện dừng (hết quota), `run_config.json` ghi lại và phải nêu cùng kết quả.
- Tracking (HOTA/IDF1) chỉ chạy được với YOLO. So sánh kiến trúc chỉ ở mức detector (mAP + tốc độ).

## 6. Thứ tự thực hiện

0. **Kiểm chứng đường chấm** trước khi chấm chính thức. Chạy `score_model` trên toàn bộ 320 ảnh val với từng mô
   hình, so với mAP val mà quá trình train ghi lại:
   - RT-DETRv2: cùng pycocotools, chỉ khác FP16 trên GPU và FP32 trên CPU → phải lệch < 0.5 điểm;
   - YOLO: so với `yolo val`; bộ chấm của ultralytics khác pycocotools nên lệch ~1–2 điểm là bình thường.

   Lệch nhiều hơn nghĩa là khâu chuyển đầu ra sai (thứ tự lớp, xyxy/xywh, tỉ lệ toạ độ), phải sửa trước.

   **Kết quả YOLO26s (7/10/2026, 320 ảnh val, CPU):**

   | Cách sinh dự đoán | mAP50 | mAP50-95 |
   |---|---|---|
   | `yolo val` mặc định (log train: 0.873 / 0.639) | 0.872 | 0.640 |
   | Dự đoán của chính `yolo val` đó (`predictions.json`), chấm bằng `vtrack.coco_eval` | 0.870 | 0.641 |
   | `yolo val rect=False` | 0.855 | 0.621 |
   | Đường chấm của dự án (`predict`, như khi chạy video) | 0.847 | 0.617 |

   → **Bộ chấm đúng** (0.870 so với 0.872). Chênh lệch 2.5 điểm với số ultralytics in ra là do **quy ước đệm ảnh khi
   val**: `rect=True` đệm thêm nửa stride, ảnh 640×640 thành đầu vào 992×992. Bỏ phần đệm đó thì chỉ còn lệch 0.4–0.8
   điểm, do `predict` letterbox tối thiểu (1280×720 → 960×544) còn val vuông hoá thành 960×960. Đường chấm của dự án
   đo đúng cách mô hình chạy khi triển khai, và áp dụng như nhau cho mọi mô hình. **Hệ quả:** không so trực tiếp số mAP
   do ultralytics in khi train với số trong báo cáo.
   **Kết quả RT-DETRv2-S:** đường chấm dự án (CPU, FP32) 0.8785 / 0.6506, log train (GPU, FP16) 0.878 / 0.6507 →
   lệch < 0.001. Khâu chuyển đầu ra đúng.
1. Chạy thử RT-DETRv2 (2 epoch, 5% dữ liệu): thời gian/epoch, bộ nhớ GPU. Đã xong 7/10/2026: 7.4 GB, ~9–10 phút/epoch.

**Diễn biến train thực tế (để đọc kết quả):**

| Mô hình | Số epoch đã chạy | Epoch tốt nhất (mAP50-95 val) | Pha tắt augmentation mạnh ở 10 epoch cuối | Giờ T4 |
|---|---|---|---|---|
| YOLO26n (v5) | 100 (không dừng sớm) | 90 | có | 3.0 |
| YOLO26s | 100 (không dừng sớm) | 87 | có | 4.9 |
| RT-DETRv2-S | 81 (dừng sớm) | 61 | không tới | 10.7 |

Epoch đánh số từ 1 (file `_epochs.csv` của RT-DETRv2 đánh từ 0: tốt nhất là dòng 60). Cả ba dùng cùng luật dừng và
cùng tiêu chí chọn checkpoint; khác biệt trong bảng là kết quả của luật đó. RT-DETRv2 hội tụ sớm hơn (val gần như phẳng
từ epoch ~55), bị dừng sớm trước pha tắt augmentation, trong khi hai bản YOLO đều trải qua pha này. Đây là một khác
biệt do luật dừng chung tạo ra, cần nêu khi so sánh. RT-DETRv2 train vừa một phiên Kaggle, không cần train tiếp.
2. Train thật YOLO26s và RT-DETRv2-S; chọn checkpoint chỉ bằng val, không nhìn tập test nào.
3. Chấm mọi mô hình (COCO gốc, v5 = YOLO26n, YOLO26s, RT-DETRv2-S) một lần trên mọi tập test.
4. Đo tốc độ (mục 4) và độ phức tạp; vẽ đồ thị mAP – ms/frame.
