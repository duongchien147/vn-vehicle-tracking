# Báo cáo LaTeX: Ứng dụng học sâu trong phát hiện, phân loại và theo dõi phương tiện giao thông

## Đưa lên Overleaf

1. Overleaf → **New Project** → **Upload Project** → chọn file `bao_cao_overleaf.zip`.
2. **Menu** → **Compiler**: **XeLaTeX**; **Main document**: `main.tex`; **TeX Live version**: bản mới nhất.
3. Bấm **Recompile**.

## Nếu Overleaf báo "Your compile timed out"

Gói Overleaf miễn phí chỉ cho **10 giây** mỗi lần biên dịch (từ 8/9/2025). Biên dịch mới cả báo cáo cần 3 lượt
XeLaTeX cộng BibTeX, đo trên máy local khoảng 8 giây, nên trên máy chủ Overleaf có thể sát hoặc vượt giới hạn. Khi đó
các cảnh báo "Citation ... undefined" chỉ là hệ quả: BibTeX chưa kịp chạy.

Cách xử lý: biên dịch theo bước. Overleaf giữ lại file `.aux` giữa các lần biên dịch, nên các phần đã dựng không phải
dựng lại. Trong `main.tex` có sẵn 3 dòng `\includeonly`:

1. Bỏ dấu `%` ở dòng **bước 1** → Recompile → đặt lại dấu `%`.
2. Làm tương tự với **bước 2**, rồi **bước 3**.
3. Để cả ba dòng có dấu `%` → Recompile: biên dịch đủ báo cáo, giờ chỉ còn 1–2 lượt.

Khi sửa một chương, chỉ bật `\includeonly{chapters/chuongN}` cho chương đó. Số chương, hình, trang của các phần khác vẫn
giữ đúng. Có thể chọn thêm **Recompile ▸ Compile mode: Fast [draft]** khi đang soạn (bỏ qua hình, nhanh hơn).

Các biện pháp đã áp dụng để giảm thời gian biên dịch:
- font toán dùng `newtxmath` thay cho `unicode-math`;
- không nạp TikZ: sơ đồ vẽ sẵn thành PDF, khung bìa vẽ bằng `eso-pic`;
- biểu đồ dạng PDF, ảnh dạng JPEG (XeLaTeX nhúng PNG lớn rất chậm);
- độ rộng số trong mục lục đặt cố định (bớt một lượt biên dịch);
- không dùng `hyperref`.

## Cấu trúc thư mục

```
main.tex                 file chính: thứ tự các phần, các dòng \includeonly để biên dịch theo bước
settings/preamble.tex    định dạng theo mẫu của trường (A4, lề, Times 13 pt, giãn dòng 1.5, header, chú thích)
frontmatter/             bìa (2 trang), nhận xét, lời cam đoan, lời cảm ơn, mục lục + từ viết tắt + danh sách bảng/hình
chapters/                modau.tex và chuong1.tex ... chuong4.tex (đúng 4 chương)
figures/                 toàn bộ hình (logo, ảnh dữ liệu, biểu đồ PDF, ảnh phân tích lỗi, ảnh giao diện, sơ đồ)
references.bib           tài liệu tham khảo, kiểu IEEE (IEEEtran)
```

## Định dạng

- Font chữ: TeX Gyre Termes (bản Times có sẵn trên Overleaf, đủ dấu tiếng Việt); font toán: newtxmath (kiểu Times).
- Header ghi tên đề tài; số trang ở giữa chân trang, đánh liên tục từ bìa như mẫu Word.
- Hình: chú thích dưới hình; bảng: chú thích trên bảng; công thức đánh số theo chương, ví dụ (2.1).

## Việc cần tự điền hoặc kiểm tra trước khi nộp

- `frontmatter/loicamdoan.tex`: ngày ký lời cam đoan (đang để trống: "ngày ... tháng ... năm 2026").
- `references.bib`: số trang của ByteTrack và OC-SORT, tên đầy đủ tác giả bộ nhãn UA-DETRAC trên Kaggle đang để trống
  vì chưa xác minh được; nên đối chiếu lại toàn bộ danh mục với Google Scholar.
- Một số hình là frame của video YouTube (clip Phạm Văn Đồng, hầm chui, Hà Nội). Các video này chỉ được dùng cho học
  tập, có ghi nguồn; không đăng báo cáo công khai kèm các hình này.

## Tạo lại hình

Hình được tạo từ kết quả của dự án (chạy trong repo, không chạy trên Overleaf):
- `report/make_figures.py`: biểu đồ, frame demo, ảnh dữ liệu, ảnh phân tích lỗi, ma trận nhầm lẫn;
- `report/figures_src/pipeline.tex`: sơ đồ luồng xử lý, biên dịch riêng thành `figures/pipeline.pdf`.
