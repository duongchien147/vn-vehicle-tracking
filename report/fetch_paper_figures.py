"""Lấy hình kiến trúc từ mã nguồn arXiv của các bài báo được trích dẫn ở chương 2 -> report/overleaf/figures/paper_*.

Mỗi hình ghi rõ bài báo, file gốc và giấy phép để truy ngược. Hình dùng lại có ghi nguồn trong chú thích (trích dẫn cho
mục đích học tập, nghiên cứu). Cần pypdfium2 để chuyển hình PDF sang ảnh (không có trong requirements.txt vì chỉ dùng
cho báo cáo):  pip install pypdfium2  rồi chạy  python report/fetch_paper_figures.py
"""
import gzip
import io
import tarfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import pypdfium2 as pdfium  # pyright: ignore[reportMissingImports]  (chỉ cần khi tạo hình cho báo cáo)

CACHE = Path("outputs/paper_sources")  # mã nguồn arXiv tải về (dữ liệu ngoài, không đưa vào git)
OUT = Path("report/overleaf/figures")
MAX_SIDE = 2000  # đủ nét khi in rộng ~16 cm (khoảng 300 dpi)
JPEG_QUALITY = 88


@dataclass
class PaperFigure:
    arxiv_id: str
    source_file: str  # đường dẫn trong gói mã nguồn arXiv
    output: str  # tên file trong report/overleaf/figures; .png cho sơ đồ, .jpg cho hình có ảnh chụp
    license: str
    keep_top: float = 1.0  # chỉ giữ phần trên (tỉ lệ chiều cao), bỏ phần hình không liên quan tới đề tài


FIGURES = [
    PaperFigure("1506.02640", "model.pdf", "paper_yolo_grid.jpg", "arXiv non-exclusive (Redmon et al., CVPR 2016)"),
    PaperFigure("2606.03748", "fig/YOLO26-architecture.pdf", "paper_yolo26_arch.png", "CC BY 4.0"),
    PaperFigure("2606.03748", "fig/YOLO26-blocks.pdf", "paper_yolo26_blocks.png", "CC BY 4.0"),
    PaperFigure("2606.03748", "fig/teaser.pdf", "paper_yolo26_speed.png", "CC BY 4.0"),
    PaperFigure("2005.12872", "figures/DETR_detail_seagulls.pdf", "paper_detr.png", "CC0 1.0"),
    PaperFigure("2304.08069", "figs/rtdetr_overview.pdf", "paper_rtdetr.png", "arXiv non-exclusive (Zhao et al., CVPR 2024)"),
    PaperFigure("2110.06864", "image/teasing.pdf", "paper_bytetrack.jpg", "arXiv non-exclusive (Zhang et al., ECCV 2022)"),
    # Bỏ khối "Post-Processing (optional)" ở dưới (nội suy quỹ đạo sau khi chạy xong, đề tài không dùng)
    PaperFigure("2206.14651", "figures/BoT-SORT-Flow.png", "paper_botsort.png", "arXiv non-exclusive (Aharon et al., 2022)",
                keep_top=0.55),
]


def download_source(arxiv_id: str) -> Path:
    """Tải gói mã nguồn arXiv một lần rồi giải nén vào CACHE/<id>/ (bỏ liên kết và đường dẫn thoát ra ngoài)."""
    folder = CACHE / arxiv_id
    if folder.exists():
        return folder
    request = urllib.request.Request(f"https://arxiv.org/e-print/{arxiv_id}", headers={"User-Agent": "Mozilla/5.0"})
    data = gzip.decompress(urllib.request.urlopen(request, timeout=120).read())
    time.sleep(2)  # tránh gửi dồn yêu cầu tới arXiv
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for member in archive.getmembers():
            name = PurePosixPath(member.name)
            if not member.isfile() or name.is_absolute() or ".." in name.parts:
                continue
            content = archive.extractfile(member)
            if content is None:
                continue
            target = folder.joinpath(*name.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content.read())
    return folder


def load_image(path: Path) -> np.ndarray:
    """Đọc hình gốc thành ảnh BGR; hình PDF được dựng trang đầu với cạnh dài MAX_SIDE điểm ảnh."""
    if path.suffix.lower() != ".pdf":
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Không đọc được ảnh {path}")
        return image
    page = pdfium.PdfDocument(str(path))[0]
    scale = MAX_SIDE / max(page.get_size())
    rgb = np.asarray(page.render(scale=scale, fill_color=(255, 255, 255, 255)).to_pil().convert("RGB"))
    return np.ascontiguousarray(rgb[:, :, ::-1])  # RGB -> BGR cho OpenCV


def save(image: np.ndarray, path: Path) -> None:
    height, width = image.shape[:2]
    scale = min(1.0, MAX_SIDE / max(height, width))
    if scale < 1.0:
        image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY] if path.suffix == ".jpg" else [cv2.IMWRITE_PNG_COMPRESSION, 9]
    cv2.imwrite(str(path), image, params)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for figure in FIGURES:
        image = load_image(download_source(figure.arxiv_id) / figure.source_file)
        image = image[: round(image.shape[0] * figure.keep_top)]
        save(image, OUT / figure.output)
        print(f"{figure.output:28s} <- arXiv:{figure.arxiv_id}/{figure.source_file} ({figure.license})")


if __name__ == "__main__":
    main()
