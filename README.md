# Phát hiện, phân loại và theo dõi phương tiện giao thông Việt Nam từ video

Đồ án môn học. Hệ thống nhận video từ camera giao thông, **phát hiện và phân loại** 4 lớp phương tiện (`motorbike`,
`car`, `bus`, `truck`), **theo dõi** từng xe bằng một ID không đổi giữa các frame, rồi đếm số xe đi qua vạch theo lớp và
hướng. Kết quả hiển thị trên web demo Streamlit. Đếm xe và web demo là ứng dụng minh hoạ của phần phát hiện và theo dõi.

**Phạm vi:** camera cố định nhìn chéo từ trên cao (kiểu camera giao thông), giao thông Việt Nam với mật độ xe máy cao,
cả ban ngày lẫn ban đêm. Bối cảnh triển khai là máy chủ có GPU xử lý hình ảnh camera tập trung; máy phát triển và chạy
demo là laptop không có GPU rời; việc huấn luyện chạy trên GPU T4 của Kaggle.

## Kết quả chính

Mọi tập test dưới đây đều **không** được dùng khi train hoặc khi chọn tham số. Các lựa chọn (mô hình, tracker,
kích thước ảnh) được chốt trên tập dev theo tiêu chí đặt trước.

**Phát hiện** (mAP50 trung bình xe máy + ô tô):

| Tập test | COCO gốc (chưa fine-tune) | v5 (YOLO26n) | v6 (YOLO26n, bản nhẹ) | **YOLO26s-v6 (mô hình chính)** |
|---|---|---|---|---|
| Clip TP.HCM (20 frame clip PVD) — camera mới | 0.664 | 0.892 | 0.907 | **0.901** |
| Cảnh mới, ban ngày, nhiều xe nhỏ ở xa (`vn_extra` ngày) — camera mới | 0.458 | 0.745 | 0.760 | **0.830** |
| Cảnh đêm Việt Nam (`vn_extra` đêm) — camera mới | 0.437 | 0.628 | 0.755 | **0.887** |
| Test Hà Nội (30 frame gán tay) — **cùng camera** với frame huấn luyện, khác thời điểm | 0.686 | 0.923 | 0.921 | **0.920** |

Test Hà Nội đo độ chính xác khi mô hình được tinh chỉnh bằng video của chính camera đó; khả năng tổng quát sang camera
mới đọc ở ba dòng đầu. v6 = v5 + 288 frame đêm (hai camera TP.HCM khác hầm chui, nhãn tự động). YOLO26s-v6 = YOLO26s
học cùng dữ liệu v6: trên GPU nó nhanh ngang YOLO26n mà chính xác hơn, nên là mô hình chính cho bối cảnh máy chủ có
GPU; v6 là bản nhẹ cho máy không có GPU rời. Cả hai được nhận theo tiêu chí viết trước khi train, chấm trên tập dev
(trong đó có một tập dev đêm riêng, camera thứ ba, nhãn người).

**Theo dõi** (clip test PVD 20 s, nhãn tracking gán tay, chấm theo chuẩn MOTChallenge bằng TrackEval):
YOLO26s-v6 + bộ lọc box trùng: HOTA **0.755**, MOTA **0.874**, IDF1 **0.916**, 22 lần đổi ID trên 183 xe; bản nhẹ v6:
0.744 / 0.847 / 0.905, 26 lần (v5: 0.718 / 0.813 / 0.880, 26 lần). Bộ lọc box trùng (NMS không phân lớp, IoU 0.6, trước
tracker) được chọn trên hai tập dev — VnTraffic và một clip PVD khác gán nhãn tay có đủ xe nhỏ ở xa; không có nó, v6
đổi ID 49 lần.

**Đếm xe qua vạch** (sai số tính riêng từng hướng): ban ngày (cùng clip, 73 lượt thật) sai số 2.7% với v6, 4.1% với
YOLO26s-v6 (lần huấn luyện thứ hai 2.7%), v5 4.1%; lỗi thật là 1 xe Ford Transit (nhãn buýt) bị đếm thành ô tô và, với
YOLO26s-v6, 1 xe máy bị sót. Ban đêm (hai clip hầm chui 60 giây, 176 lượt người đếm tay): sai số **1.7%** với YOLO26s-v6
và v6, 8.5% với v5, 4.5% với YOLO26s học dữ liệu v5; lỗi gần như toàn bộ ở xe máy.

**Tốc độ** (video 1080p, cả pipeline phát hiện + theo dõi):

| Phần cứng | YOLO26n (v5, v6) | YOLO26s (v5, YOLO26s-v6) |
|---|---|---|
| GPU Tesla T4 (Kaggle), PyTorch | ~42 frame/s | ~43 frame/s |
| Laptop i5-8250U, OpenVINO đầu vào 544×960 trên GPU Intel UHD 620 tích hợp | ~19.8 frame/s | ~9.4 frame/s |
| Laptop i5-8250U, chỉ CPU, OpenVINO INT8 544×960 | ~18.8 frame/s | ~8.2 frame/s |
| Laptop i5-8250U, chỉ CPU, PyTorch | ~7.5 frame/s | ~3.3 frame/s |

Đo với các bản học dữ liệu v5. Đo lại trên laptop với v6 và YOLO26s-v6 (`outputs/benchmark_speed/e2_v6_cpu|e2_v6_igpu/`)
cho kết quả tương đương: bản nhẹ OpenVINO trên GPU tích hợp 19.5 frame/s, INT8 trên CPU 18.2, YOLO26s-v6 trên GPU tích
hợp 9.8. Bộ lọc box trùng chỉ tốn thêm vài ms mỗi frame.

Tính cả vẽ kết quả và ghi video 1080p, web demo trên laptop chạy khoảng 7 frame/s với bản nhẹ; video demo của mô hình
chính được xử lý sẵn (`scripts/make_demo_videos.sh`) để trình chiếu mượt.

**So sánh kiến trúc** (cùng dữ liệu, cùng giao thức): YOLO26n, YOLO26s (CNN) và RT-DETRv2-S (Transformer). Khi được
nhìn cùng số điểm ảnh, YOLO26s và RT-DETRv2-S hoà nhau trên dữ liệu Việt Nam. RT-DETRv2-S cho box khít hơn
(mAP50-95 cao hơn ~0.03 trên các tập nhãn độc lập) nhưng chậm hơn YOLO26s 5.8 lần trên CPU và ~4 lần trên GPU T4.
Vì vậy mô hình chính là YOLO26s (học thêm dữ liệu đêm: YOLO26s-v6) chạy trên GPU; YOLO26n v6 là bản nhẹ cho laptop.
Giao thức và kết quả đầy đủ: [docs/giao_thuc_so_sanh_kien_truc.md](docs/giao_thuc_so_sanh_kien_truc.md).

## Pipeline

```
Video ──► YOLO26s-v6 (phát hiện 4 lớp, imgsz 960) ──► lọc box trùng ──► BoT-SORT (gán ID theo dõi)
      ──► bỏ phiếu lớp theo track ──► đếm qua vạch (supervision LineZone)
      ──► video có vẽ box/ID + CSV sự kiện đếm
```

- **Phát hiện:** YOLO26s fine-tune từ trọng số COCO trên dữ liệu xe Việt Nam, camera giao thông và cảnh đêm
  (`weights/yolo26s_vn_v6_best.pt`); bản nhẹ YOLO26n cùng dữ liệu (`weights/yolo26n_vn_v6_best.pt`) cho máy không có GPU.
- **Theo dõi:** BoT-SORT (ultralytics) đã chỉnh trên tập dev, tắt bù chuyển động camera vì camera đứng yên
  (`configs/botsort_tuned_nogmc.yaml`); trước tracker bỏ box trùng bất kể lớp (NMS IoU 0.6, `vtrack/box_filter.py`) vì
  YOLO26 là mô hình end-to-end, không chạy NMS.
- **Bỏ phiếu lớp:** một xe có thể bị gọi lúc "car" lúc "truck" giữa các frame; lớp cuối cùng của mỗi track được chọn
  bằng bỏ phiếu theo độ tin cậy, trước khi đếm.

## Cài đặt

Yêu cầu Python 3.10. Trên máy chỉ có CPU, cài PyTorch bản CPU trước:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -e .
```

**Trọng số mô hình và dữ liệu không nằm trong repo** (quá lớn, và một số video có giấy phép không cho đăng lại).
Trọng số được tạo bằng các notebook huấn luyện trong `notebooks/` và đặt vào `weights/` (mặc định
`weights/yolo26s_vn_v6_best.pt`, bản nhẹ `weights/yolo26n_vn_v6_best.pt`; bản OpenVINO cho laptop tạo bằng
`scripts\export_openvino.py --weights <trọng số> --input-hw 544 960`).

## Chạy thử

```powershell
# Xử lý một video -> outputs/<tên>_tracked.mp4 + _events.csv + _run_config.json
.venv\Scripts\python.exe scripts\infer_video.py --source data\videos\nga_tu_hanoi_a.mp4
# Bản nhẹ OpenVINO trên GPU Intel tích hợp (#gpu; máy không có GPU Intel thì dùng bản INT8 với #cpu)
# và vạch đếm tuỳ chọn (toạ độ tương đối 0–1)
.venv\Scripts\python.exe scripts\infer_video.py --source <video> --weights "weights\yolo26n_vn_v6_best_544x960_openvino_model#gpu" --line 0 0.6 1 0.6

# Xử lý sẵn 6 clip demo bằng mô hình chính (vạch đếm theo configs/demo_clips.yaml) -> outputs/demo_precomputed/
bash scripts/make_demo_videos.sh

# Web demo: chế độ "Kết quả xử lý sẵn" nạp lại video + bảng đếm ở outputs/demo_precomputed/ (không xử lý lại);
# chế độ xử lý trực tiếp: chọn clip hoặc tải video lên, chọn mô hình / vạch đếm -> video kết quả, bảng đếm, biểu đồ, CSV.
.venv\Scripts\streamlit.exe run app\streamlit_app.py
```

## Đánh giá

| Lệnh | Đo gì | Kết quả |
|---|---|---|
| `scripts/eval_hanoi_test.py` | mAP kiểu COCO (có vùng bỏ qua) trên tập ảnh gán tay | `outputs/eval_hanoi/` |
| `scripts/eval_clip_detection.py` | mAP trên frame của clip có nhãn tracking | `outputs/eval_clip_detection/` |
| `scripts/eval_uvh26.py`, `scripts/eval_ua_detrac.py` | mAP trên bộ dữ liệu ngoài (UVH-26; UA-DETRAC đêm/ngày) | `outputs/eval_uvh26/`, `outputs/eval_ua_detrac/` |
| `scripts/eval_tracking.py` | HOTA, MOTA, IDF1, số lần đổi ID (TrackEval) | `outputs/eval_tracking/` |
| `scripts/eval_counting.py` | Sai số đếm theo lớp và hướng (số thật từ nhãn quỹ đạo) | `outputs/eval_counting/` |
| `scripts/count_by_hand.py`, `scripts/eval_manual_counts.py` | Sai số đếm so với người đếm tay (clip đêm) | `data/count_gt/`, `outputs/eval_manual_counts/` |
| `scripts/analyze_size_recall.py` | Recall theo chiều cao xe | `outputs/size_recall/` |
| `scripts/benchmark_speed.py` | Tốc độ CPU/GPU, số tham số, GFLOPs | `outputs/benchmark_speed/` |
| `scripts/render_error_examples.py` | Ảnh minh hoạ các loại lỗi | `outputs/error_analysis/` |

Mỗi lần chạy ghi kết quả vào thư mục riêng, kèm cấu hình đã dùng (`run_config.json`). Kiểm thử:
`.venv\Scripts\python.exe -m pytest -q`.

## Cấu trúc thư mục

| Thư mục | Nội dung |
|---|---|
| `src/vtrack/` | Toàn bộ logic: nạp mô hình, phát hiện, theo dõi, đếm, chấm điểm, đo tốc độ |
| `scripts/` | Lớp gọi mỏng: chuẩn bị dữ liệu, gán nhãn, suy luận, đánh giá, đưa lên Kaggle |
| `app/` | Web demo Streamlit |
| `configs/` | Cấu hình tracker, dữ liệu, tập đánh giá, ảnh minh hoạ lỗi |
| `notebooks/` | Huấn luyện trên Kaggle (`kaggle_train` cho YOLO26, `kaggle_train_rtdetr` cho RT-DETRv2), đo tốc độ GPU |
| `docs/` | Giao thức so sánh kiến trúc, hướng dẫn gán nhãn tracking |
| `tests/` | Kiểm thử logic thuần (đếm, bỏ phiếu lớp, chuyển định dạng, chấm điểm) |
| `data/`, `weights/`, `outputs/` | Dữ liệu, trọng số, kết quả (không đưa lên git) |

## Dữ liệu

**Huấn luyện** (gộp bằng `scripts/prepare_data.py`):
- Roboflow Universe: [vietnamese-vehicle](https://universe.roboflow.com/car-classification/vietnamese-vehicle),
  [vehicle-vietnam-cantho](https://universe.roboflow.com/vehicle/vehicle-vietnam-cantho-2gxc8),
  [vehicle-detection](https://universe.roboflow.com/skripsi-1qzlz/vehicle-detection-ckrxi) (CC BY 4.0).
- [VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset) (chỉ dùng cho nghiên cứu).
- [UVH-26](https://huggingface.co/datasets/iisc-aim/UVH-26) (IISc, CC BY 4.0): 2.000 ảnh camera giao thông.
- Frame từ video giao thông Hà Nội, gán nhãn tự động bằng ghép nhiều mô hình.

**Đánh giá:** frame gán tay từ video giao thông Hà Nội và TP.HCM (ngày và đêm); nhãn tracking gán tay bằng CVAT;
[VnTraffic](https://doi.org/10.5281/zenodo.18195750) (CC BY 4.0) làm tập dev cho tracking; val UVH-26; UA-DETRAC
(bản gán lại có xe máy) cho so sánh đêm/ngày. Nguồn và giấy phép từng video: `data/videos/SOURCES.md`. Video lấy từ
YouTube chỉ dùng cho học tập và không được đăng lại.

## Hạn chế

- Tập test nhỏ (mỗi tập 20–30 frame; một clip tracking 20 s). Có khoảng tin cậy bootstrap, nhưng mỗi mô hình chỉ train
  một lần.
- Xe rất nhỏ ở xa (cao < 20 px) và xe bị che ban đêm vẫn hay bị sót. Ban đêm, mô hình đôi khi chỉ khoanh phần đèn hậu
  của xe máy.
- Lớp "van" không có trong 4 lớp: xe cỡ Ford Transit được quy ước là buýt, và mô hình đôi khi gọi là ô tô.
- Trên CPU, demo chưa đạt thời gian thực (~10.9 frame/s với video 1080p); cần GPU để đạt ~42 frame/s.
