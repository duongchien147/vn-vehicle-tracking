# Hướng dẫn đếm tay số xe qua vạch

Dùng làm nhãn chuẩn cho **sai số đếm** trên các clip chưa có nhãn tracking: hai clip đêm `ham_chui_dem_2` và
`ham_chui_dem_3` (mỗi clip 60 giây, khoảng 80–100 lượt). Không đếm `ham_chui_dem_1`: clip này trùng 46 giây với clip 2
và có chỗ cắt cảnh ở giây 48.3 (xem `data/videos/SOURCES.md`). Đếm tay theo lớp và hướng là cách cuộc thi AI City
Challenge (bài đếm xe) tạo nhãn chuẩn.

Đếm **độc lập**: đừng mở bộ đếm do AI làm trong `data/count_gt/ai_review_2026-10-11/` trước khi đếm xong. Bộ đó được
giữ để so sau (mức đồng thuận người–AI).

**Đừng xem video kết quả của mô hình (`*_tracked.mp4`, app Streamlit) trước khi đếm clip đó**: biết mô hình đếm bao
nhiêu dễ làm mình đếm theo.

## 1. Chạy công cụ

```powershell
.venv\Scripts\python.exe scripts\count_by_hand.py --source data\videos\ham_chui_dem_2.mp4
.venv\Scripts\python.exe scripts\count_by_hand.py --source data\videos\ham_chui_dem_3.mp4
```

Hai clip đêm dùng vạch mặc định (ngang, y = 0.6), trùng vạch của video demo GPU nên không cần thêm `--line`. Mỗi lần
bấm được lưu ngay vào `data\count_gt\<clip>\manual_counts.csv`: tắt giữa chừng thì chạy lại lệnh là làm tiếp.

| Phím | Việc |
|---|---|
| `m` `c` `b` `t` | xe máy / ô tô / buýt / tải đi theo chiều **RA** (mũi tên cam) |
| `Shift` + `M` `C` `B` `T` | cùng các lớp đó đi theo chiều **VÀO** (mũi tên xanh) |
| `Space` | dừng / chạy tiếp (mở lên đang dừng) |
| `,` `.` | lùi / tới 1 frame (khi dừng) |
| `a` `d` | lùi / tới 2 giây |
| `-` `=` | chậm hơn / nhanh hơn (0.1x – 1x, mặc định 0.25x) |
| `u` | xoá lượt gần frame đang xem nhất (trong 2 giây) |
| `q` hoặc `Esc` | lưu và thoát |

Tắt **Caps Lock** (nếu không, chữ thường thành chữ hoa và chiều bị đảo). Cửa sổ kéo giãn hoặc phóng to được.

## 2. Quy tắc đếm

- **Thời điểm bấm:** khi **điểm giữa đáy xe** (chỗ thấp nhất của xe trên ảnh, gần chỗ bánh chạm đường) đi qua vạch vàng.
  Bộ đếm của pipeline dùng đúng điểm này, nên đếm theo đầu xe sẽ lệch. Bấm lúc đã dừng video cũng được: frame đang hiện
  là frame được ghi.
- **Mỗi lần xe đi hẳn qua vạch là một lượt.** Xe qua rồi quay lại (hiếm) thì tính hai lượt, ngược chiều nhau.
- **Lớp xe**, giống quy tắc nhãn của đề tài: xe máy tính cả người ngồi trên; van nhỏ và xe 7 chỗ là **ô tô**; van cỡ
  Ford Transit mui cao (16 chỗ hoặc chở hàng) là **buýt**; xe có thùng hàng riêng là **tải**.
- **Xe bị che lúc qua vạch:** vẫn đếm nếu chắc chắn có một xe đi qua (thấy trước và sau chỗ che). Lớp lấy theo lúc nhìn
  rõ nhất, lùi/tới vài giây để xem.
- **Cụm xe đông:** dừng lại, bước từng frame (`.`) và bấm cho từng xe. Dòng "gan day" trên màn hình liệt kê các lượt
  đã bấm trong ±3 giây, để khi tua lại không bấm trùng.
- **Không chắc là xe hay không** (vệt đèn, bóng): không đếm. Ghi lại giây đó (cột ghi chú bên dưới) nếu muốn hỏi lại.

## 3. Sau khi đếm xong

Nhắn cho trợ lý AI. trợ lý AI sẽ chạy pipeline v5 trên CPU với đúng vạch đó, rồi chấm cả v5 lẫn YOLO26s (video demo GPU):

```powershell
.venv\Scripts\python.exe scripts\eval_manual_counts.py --runs outputs\kaggle\vtrack-demo-gpu\demo outputs\manual_count_runs\v5
```

`mistakes.csv` liệt kê từng lượt bị sót, đếm thừa hoặc sai lớp kèm giây trong video, để mở lại xem lỗi do mô hình hay
do mình đếm nhầm. Nếu là mình nhầm thì sửa `manual_counts.csv` và ghi lý do vào `data\count_gt\<clip>\notes.txt`,
**trước** khi so lần cuối (giống cách sửa nhãn test `vn_extra`).
