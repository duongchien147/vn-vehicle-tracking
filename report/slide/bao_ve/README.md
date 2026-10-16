# Mẫu Slide Thuyết Trình Beamer (UTEHY / HYUTE Minimal)

Bộ khung (template) Slide Beamer giao diện đơn sắc tối giản, tách thành ba lớp độc lập: **nội dung (`main.tex`)**, **cấu hình (`config/`)** và **engine (`theme/`)**.

## Cấu trúc thư mục

```
Mau_Slide/
├── main.tex                  # CHỈ chứa nội dung slide (bộ khung mẫu)
├── references.bib            # Tài liệu tham khảo (BibTeX, kiểu IEEE)
├── README.md                 # Hướng dẫn sử dụng và tùy chỉnh
├── config/                   # MỌI tinh chỉnh tập trung tại đây
│   ├── fonts.tex             #   họ font, font toán, cỡ chữ
│   └── settings.tex          #   khổ slide, logo, thư mục ảnh, footer, nhãn
├── theme/                    # "Engine" giao diện, tái sử dụng được
│   └── beamerthemeMLWNet.sty
├── fonts/                    # Be Vietnam Pro (SIL Open Font License, xem OFL.txt)
│   ├── BeVietnamPro-Regular.ttf
│   ├── BeVietnamPro-Bold.ttf
│   ├── BeVietnamPro-Italic.ttf
│   ├── BeVietnamPro-BoldItalic.ttf
│   └── OFL.txt
└── images/
    ├── logo-utehy.png        # Logo trường (hiển thị ở trang bìa và góc phải tiêu đề)
    └── dwt_haar_demo.png     # Ảnh mẫu trong thư mục images/
```

## Biên dịch

**Bắt buộc dùng XeLaTeX** (font `.ttf` cần gói `fontspec`).

- **Overleaf:** Menu → Compiler → **XeLaTeX** → Recompile. BibTeX chạy tự động.
- **Máy cá nhân:**
  ```bash
  xelatex main && bibtex main && xelatex main && xelatex main
  # hoặc
  latexmk -xelatex main.tex
  ```

Nếu lỡ biên dịch bằng pdfLaTeX, theme in cảnh báo và chuyển sang Latin Modern, font trong `config/` bị bỏ qua.

## Tinh chỉnh nhanh

| Muốn đổi | Sửa ở | Macro |
|---|---|---|
| Font chữ khác (file) | `config/fonts.tex` | `\mlwFontName`, `\mlwFontExt` (bỏ file vào `fonts/`) |
| Font cài sẵn trong máy | `config/fonts.tex` | `\mlwFontPath{}` để trống, `\mlwFontName{Arial}` |
| Font toán | `config/fonts.tex` | `\mlwMathFont`: `cm`, `sans` hoặc tên font OpenType Math |
| Cỡ chữ toàn bài | `config/fonts.tex` | `\mlwBaseSize` (8pt...20pt) |
| Cỡ tiêu đề, footer, nguồn | `config/fonts.tex` | `\mlwSizeFrameTitle`, `\mlwSizeFooter`, `\mlwSizeSource`... |
| Khổ slide | `config/settings.tex` | `\mlwAspectRatio` (169, 1610, 43) |
| Logo, kích thước logo | `config/settings.tex` | `\mlwLogoFile`, `\mlwLogoHeaderHeight`, `\mlwLogoOnEveryFrame` |
| Thư mục ảnh | `config/settings.tex` | `\mlwImageDir` |
| Dòng footer | `config/settings.tex` | `\mlwFooterText` (để trống thì lấy `\title[ngắn]`) |
| Tắt slide chuyển mục | `config/settings.tex` | `\mlwSectionDivider{false}` |
| Kiểu trích dẫn | `config/settings.tex` | `\mlwBibStyle` (thiếu `IEEEtran.bst` thì tự dùng `ieeetr`) |

## Quy ước soạn slide trong `main.tex`

- Không gõ số vào tiêu đề: dùng `\subsection{Tên}` rồi `\begin{frame}{\stitle}` để có "x.y. Tên".
- Trích dẫn bằng khoá BibTeX: `\cite{key}`; sau tên phương pháp dùng `\ct{key}` (số [n] màu xám).
- Hình/bảng lấy từ nguồn: thêm `\src{Nguồn: \cite{key}, Bảng x.}` ngay bên dưới.
- Chèn hình: `\figorbox{ten.png}{width=...,height=...,keepaspectratio}{chiều cao khung dự phòng}`, chỉ ghi tên file.
- Dòng cần nhấn mạnh trong bảng: `\rowcolor{HiRow}`.
- Slide cuối: `\bibslide` (tự chia trang) và `\qaslide{lời cảm ơn}`.
