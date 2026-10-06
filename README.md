# Phát hiện, phân loại và theo dõi phương tiện giao thông Việt Nam từ video

Đồ án môn học. Hệ thống nhận video giao thông, phát hiện 4 lớp phương tiện (`motorbike`, `car`, `bus`, `truck`), gán ID theo dõi cho từng xe, đếm số xe đi qua vạch theo từng lớp và hiển thị kết quả trên web demo Streamlit.

## Pipeline

```
Video → YOLO26 (phát hiện 4 lớp) → ByteTrack / BoT-SORT (gán ID)
      → bỏ phiếu lớp theo track → đếm qua vạch (supervision LineZone)
      → video có annotate + CSV số lượng
```

## Cài đặt

Yêu cầu Python 3.10. Môi trường local chỉ chạy CPU, việc huấn luyện thực hiện trên Kaggle (GPU T4).

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Cấu trúc thư mục

| Thư mục | Nội dung |
|---|---|
| `configs/` | `data.yaml`, cấu hình tracker, cấu hình app |
| `src/vtrack/` | Mã nguồn pipeline (phát hiện, theo dõi, đếm, đọc/ghi video) |
| `scripts/` | Chuẩn bị dữ liệu, suy luận video, đánh giá, export OpenVINO, đo tốc độ |
| `app/` | Web demo Streamlit |
| `notebooks/` | EDA, notebook huấn luyện trên Kaggle |
| `tests/` | Kiểm thử logic đếm và đọc/ghi video |
| `data/`, `weights/`, `outputs/` | Dữ liệu, trọng số, kết quả (không đưa lên git) |

## Kết quả

_Sẽ cập nhật sau khi chạy thí nghiệm._
