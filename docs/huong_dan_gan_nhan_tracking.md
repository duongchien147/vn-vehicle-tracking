# Hướng dẫn gán nhãn tracking bằng CVAT (clip test `pvd_goc_a`)

Mục đích: tạo nhãn tracking **đầy đủ** cho 1 clip 20 s (600 frame, 30 fps) để làm **tập test** của tracker. Tập dev là
VnTraffic (nhãn thiếu, chỉ dùng để chỉnh tham số). Clip test chỉ dùng để chấm cấu hình đã chọn, **một lần**.

## 1. Chuẩn bị (máy đã làm sẵn)

```powershell
.venv\Scripts\python.exe scripts\prelabel_tracking.py --source data\videos\pvd_goc_a.mp4 --start 20 --duration 20 `
    --tracker <cấu hình chọn trên dev> --out data\tracking_gt\pvd_goc_a
```

Trong `data\tracking_gt\pvd_goc_a\`: `clip.mp4` (video để gán), `prelabel_cvat.xml` (nhãn sơ bộ từ tracker, keyframe mỗi
10 frame), `labels.txt`.

## 2. Tạo task trên CVAT (app.cvat.ai, tài khoản miễn phí)

1. **Projects → +** tạo project, khai báo đúng 5 nhãn, đúng thứ tự trong `labels.txt`:
   `motorbike`, `car`, `bus`, `truck`, `ignore` (kiểu Rectangle).
2. **Tasks → +** trong project đó, chọn file `clip.mp4`. Mục *Advanced*: **Image quality 100**, *Frame step* để trống
   (phải giữ đủ 600 frame). Task để riêng tư, **không chia sẻ** (video YouTube chỉ dùng cho học tập).
3. Mở job → **Menu → Upload annotations → "CVAT for video 1.1"** → chọn `prelabel_cvat.xml`.

## 3. Quy tắc gán (giống quy tắc của tập test Hà Nội)

- **Mỗi xe một track, giữ nguyên một ID suốt clip** — kể cả khi bị che rồi hiện lại. Đây là thứ đang được chấm.
- Box ôm sát xe; **xe máy gồm cả người ngồi trên**. Xe đỗ/dừng cũng gán.
- Gán xe khi nhìn frame đó chỉ ra được là xe gì và nằm từ đâu đến đâu. Bị che **hẳn** → bấm **O** (outside) ở frame
  bắt đầu bị che; khi hiện lại, bấm **O** lần nữa trên **chính track đó** (không tạo track mới).
- **Vùng xa quá nhỏ** (cuối đại lộ, xe chỉ vài pixel): vẽ **một** box nhãn `ignore` phủ vùng đó ở frame 0, để nguyên
  đến hết clip. Trong vùng ignore không gán xe; tracker cũng không bị tính đúng/sai ở đó.

## 4. Cách sửa nhãn sơ bộ (nhanh nhất)

Nhãn sơ bộ là đầu ra của tracker nên có đủ loại lỗi đang cần đo — **đừng tin, kiểm từng track**:

| Lỗi trong nhãn sơ bộ | Cách sửa |
|---|---|
| Một xe bị tách thành 2–3 track (đổi ID) | Chọn các track → **M** (merge) |
| Một track nhảy từ xe này sang xe khác | Tới frame nhảy → **Alt+M** (split), rồi merge phần sau vào đúng xe |
| Box lệch ở giữa hai keyframe | Kéo box ở frame lệch nhiều nhất (tự thành keyframe), CVAT nội suy lại |
| Track rác (không phải xe) | Chọn → Delete |
| Xe không có track | **N** vẽ box ở frame xuất hiện, các frame sau chỉnh vài keyframe, **O** khi xe rời cảnh |
| Sai lớp (car ↔ truck) | Đổi nhãn của track |

Mẹo: bật *Show all interpolation tracks*, phát ở tốc độ chậm và theo dõi từng xe một từ lúc vào đến lúc ra cảnh.
Thời gian dự kiến: 2–4 giờ.

## 5. Xuất kết quả

**Menu → Export job dataset → "MOT 1.1"** (không kèm ảnh) → giải nén vào `data\tracking_gt\pvd_goc_a\` để có
`gt\gt.txt` và `gt\labels.txt`. Sau đó báo để chấm tracker (script chuyển nhãn và vùng ignore tự động).
