"""Tạo hình cho báo cáo LaTeX (report/overleaf/figures/) từ kết quả có sẵn trong outputs/ và data/:
biểu đồ độ chính xác – tốc độ, đường học, frame video demo, ảnh minh hoạ phần mở đầu, ảnh ví dụ từng nguồn dữ liệu,
ảnh phân tích lỗi.
Số liệu biểu đồ đọc từ CSV kết quả (không gõ tay) để hình khớp bảng trong báo cáo. Biểu đồ xuất PDF, ảnh xuất JPEG:
XeLaTeX nhúng hai loại này gần như không tốn thời gian, còn PNG lớn làm chậm biên dịch (Overleaf miễn phí giới hạn 10 s).

Chạy: .venv\\Scripts\\python.exe report\\make_figures.py
"""

import csv
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

OUT = Path("report/overleaf/figures")
KAGGLE_YOLO = Path("outputs/kaggle/vn-vehicles-yolo26-train")
KAGGLE_YOLO_MAIN = Path("outputs/kaggle/vn-vehicles-yolo26s-train-v6")  # lần train mô hình chính YOLO26s-v6 (seed 42)
RTDETR_EPOCHS = Path("outputs/kaggle/vn-vehicles-rtdetrv2-train/rtdetrv2_r18_v5_e100_960_epochs.csv")
DEMO_DIR = Path("outputs/demo_precomputed")  # video demo của mô hình chính, tạo bằng scripts/make_demo_videos.sh

# Bảng màu danh mục (3 ô đầu đạt kiểm tra mù màu theo mọi cặp) + marker/nét khác nhau để in đen trắng vẫn phân biệt
SERIES = [
    {"color": "#2a78d6", "marker": "o", "linestyle": "-"},
    {"color": "#eb6834", "marker": "s", "linestyle": "--"},
    {"color": "#1baf7a", "marker": "^", "linestyle": ":"},
]
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
FRAME_TIME_S = 30  # lấy frame giữa clip demo 60 s
# Ảnh phân tích lỗi (outputs/error_analysis/<số>_*.jpg) có trong báo cáo: xe nhỏ ở xa, cụm xe máy, xe đêm chỉ thấy
# đèn, đổi ID. Các loại lỗi còn lại chỉ mô tả bằng lời trong báo cáo.
ERROR_CASES_IN_REPORT = ("1", "2", "3", "7")
# Ảnh gốc (chưa qua mô hình) minh hoạ lý do chọn đề tài: tên file -> (video, giây), chọn bằng mắt.
# Ban ngày: ngã tư Hà Nội lúc xe máy đi dày xen ô tô; ban đêm: đường phố TP.HCM nhiều xe, chỉ có đèn đường và đèn xe.
INTRO_FRAMES = {
    "intro_day.jpg": ("data/videos/nga_tu_hanoi_a.mp4", 35),
    "intro_night.jpg": ("data/raw/night_extra/dem_duong_buyt_1h35.mp4", 1),
}
JPEG_QUALITY = 85
MAX_SIDE = 1600  # thu nhỏ ảnh để gói Overleaf gọn, vẫn đủ nét khi in
SPEED_TICKS_MS = [100, 200, 500, 1000, 2000]
# Không ghi ngày tạo vào PDF: chạy lại script không sinh thay đổi giả trong git
PDF_METADATA = {"CreationDate": None}


def style_axes(ax: Axes) -> None:
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=TEXT)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_score(path: Path, model: str, imgsz: str, column: str = "main_mAP50") -> float:
    for row in read_rows(path):
        if row["model"] == model and row["imgsz"] == imgsz:
            return float(row[column])
    raise KeyError(f"{model}@{imgsz} không có trong {path}")


def cpu_ms(model_key: str) -> float:
    for row in read_rows(Path("outputs/benchmark_speed/e2_arch_nogmc/results.csv")):
        if row["model"] == model_key:
            return float(row["detector_ms_per_frame"])
    raise KeyError(model_key)


def accuracy_vs_speed() -> None:
    """Hai bảng con cùng trục x: mAP50 chính trên vn_extra đêm và val UVH-26 theo ms/frame CPU (thang log)."""
    models = [
        ("YOLO26n (v5)", "yolo26n_vn_v5_best", "960", "weights/yolo26n_vn_v5_best.pt"),
        ("YOLO26s", "yolo26s_vn_v5_best", "960", "weights/yolo26s_vn_v5_best.pt"),
        ("YOLO26s @1280", "yolo26s_vn_v5_best", "1280", "weights/yolo26s_vn_v5_best.pt@1280"),
        ("RT-DETRv2-S", "rtdetrv2_r18_vn_v5_best", "960", "weights/rtdetrv2_r18_vn_v5_best"),
    ]
    panels = [
        ("(a) vn_extra đêm", {"960": Path("outputs/eval_hanoi/arch/night/results.csv"),
                               "1280": Path("outputs/eval_hanoi/arch_pixel_control/night/results.csv")}),
        ("(b) val UVH-26", {"960": Path("outputs/eval_uvh26/arch/results.csv"),
                            "1280": Path("outputs/eval_uvh26/arch/results.csv")}),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True)
    for ax, (title, sources) in zip(axes, panels):
        xs = [cpu_ms(key) for _, _, _, key in models]
        ys = [find_score(sources[imgsz], name, imgsz) for _, name, imgsz, _ in models]
        # Chỉ vẽ điểm, không nối: các mô hình là lựa chọn rời rạc, không có giá trị ở giữa
        ax.plot(xs, ys, linestyle="none", color=SERIES[0]["color"], marker="o", markersize=8,
                markeredgecolor="white", markeredgewidth=1.5)
        for index, ((label, *_), x, y) in enumerate(zip(models, xs, ys)):
            # Điểm đầu sát trục tung nên ghi nhãn bên phải, các điểm còn lại ghi bên trái để không đè nhau
            right = index == 0
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(8 if right else -8, 8),
                        ha="left" if right else "right", fontsize=9, color=TEXT)
        ax.set_xscale("log")
        ax.set_xticks(SPEED_TICKS_MS, [str(t) for t in SPEED_TICKS_MS])
        ax.minorticks_off()
        ax.margins(y=0.15)
        ax.set_title(title, fontsize=10, color=TEXT, loc="left")
        ax.set_xlabel("Thời gian phát hiện trên CPU (ms/frame, thang log)", fontsize=9, color=TEXT)
        style_axes(ax)
    axes[0].set_ylabel("mAP50 chính", fontsize=9, color=TEXT)
    axes[0].set_xlim(60, 2000)
    fig.tight_layout()
    fig.savefig(OUT / "accuracy_vs_speed.pdf", metadata=PDF_METADATA)
    plt.close(fig)


def learning_curves() -> None:
    """mAP50-95 trên val theo epoch của 3 mô hình (số do bộ chấm lúc train in ra)."""
    curves = []
    for label, run in [("YOLO26n (v5)", "yolo26n_v5_e100_960"), ("YOLO26s", "yolo26s_v5_e100_960")]:
        rows = read_rows(KAGGLE_YOLO / "runs" / run / "results.csv")
        curves.append((label, [int(r["epoch"]) for r in rows], [float(r["metrics/mAP50-95(B)"]) for r in rows]))
    rows = read_rows(RTDETR_EPOCHS)
    # _epochs.csv đánh epoch từ 0, đổi sang đánh từ 1 như ultralytics
    curves.append(("RT-DETRv2-S", [int(r["epoch"]) + 1 for r in rows], [float(r["val_mAP50-95"]) for r in rows]))

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for (label, epochs, values), spec in zip(curves, SERIES):
        ax.plot(epochs, values, color=spec["color"], linestyle=spec["linestyle"], linewidth=1.8, label=label)
        best = max(range(len(values)), key=values.__getitem__)
        ax.plot(epochs[best], values[best], marker=spec["marker"], markersize=8, color=spec["color"],
                markeredgecolor="white", markeredgewidth=1.5)
    ax.set_xlabel("Epoch", fontsize=9, color=TEXT)
    ax.set_ylabel("mAP50-95 trên tập val", fontsize=9, color=TEXT)
    ax.set_ylim(0.25, 0.7)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT / "learning_curves.pdf", metadata=PDF_METADATA)
    plt.close(fig)


def save_jpeg(image, path: Path) -> None:
    height, width = image.shape[:2]
    scale = min(1.0, MAX_SIDE / max(height, width))
    if scale < 1.0:
        image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])


def demo_frames() -> None:
    for clip in ("pvd_goc_a",):
        capture = cv2.VideoCapture(str(DEMO_DIR / f"{clip}_tracked.mp4"))
        capture.set(cv2.CAP_PROP_POS_MSEC, FRAME_TIME_S * 1000)
        ok, frame = capture.read()
        capture.release()
        if not ok:
            raise RuntimeError(f"Không đọc được frame của {clip}")
        save_jpeg(frame, OUT / f"demo_{clip}.jpg")


def intro_frames() -> None:
    for name, (video, second) in INTRO_FRAMES.items():
        capture = cv2.VideoCapture(video)
        capture.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
        ok, frame = capture.read()
        capture.release()
        if not ok:
            raise RuntimeError(f"Không đọc được giây {second} của {video}")
        save_jpeg(frame, OUT / name)


def data_source_examples() -> None:
    """Một ảnh mỗi nguồn train: ảnh thứ `index` theo thứ tự tên (chọn bằng mắt cho rõ cảnh)."""
    picks = {"vnvehicle": 5, "cantho": 0, "indo": 0, "visdrone": 250, "uvh26": 0, "hanoi": 0}
    images = sorted(Path("data/processed/vn_vehicles/train/images").iterdir())
    for source, index in picks.items():
        candidates = [p for p in images if p.name.startswith(source + "_")]
        image = cv2.imread(str(candidates[index]))
        save_jpeg(image, OUT / f"data_{source}.jpg")


def copy_existing() -> None:
    for path in sorted(Path("outputs/error_analysis").glob("*.jpg")):
        case = path.name.split("_")[0]
        if case in ERROR_CASES_IN_REPORT:
            save_jpeg(cv2.imread(str(path)), OUT / f"error_{case}.jpg")
    confusion = cv2.imread(str(KAGGLE_YOLO_MAIN / "runs/yolo26s_v6_e100_960/test/confusion_matrix_normalized.png"))
    save_jpeg(confusion, OUT / "confusion_matrix.jpg")
    # Ảnh chụp giao diện web demo (chụp bằng trình duyệt, lưu PNG vào OUT) -> JPEG
    for screenshot in sorted(OUT.glob("app_*.png")):
        save_jpeg(cv2.imread(str(screenshot)), screenshot.with_suffix(".jpg"))
        screenshot.unlink()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    accuracy_vs_speed()
    learning_curves()
    demo_frames()
    intro_frames()
    data_source_examples()
    copy_existing()
    print("\n".join(sorted(p.name for p in OUT.iterdir())))


if __name__ == "__main__":
    main()
