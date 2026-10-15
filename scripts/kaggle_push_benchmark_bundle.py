"""Đóng gói những gì cần để chạy trên GPU Kaggle (mã nguồn vtrack, script, cấu hình tracker, trọng số, video) rồi đưa
lên Kaggle dưới dạng dataset PRIVATE; dùng bởi kernel notebooks/kaggle_benchmark_gpu (đo tốc độ) và
notebooks/kaggle_demo_gpu (xử lý sẵn video demo).

Video đo tốc độ là VnTraffic (CC BY 4.0, được phép phân phối lại). Clip demo PVD / hầm chui lấy từ YouTube: chỉ được
dùng cho học tập, KHÔNG đăng lại -> dataset phải luôn để private (người làm đồ án đồng ý đưa lên 8/10/2026).

Ví dụ:
    python scripts/kaggle_push_benchmark_bundle.py
    python scripts/kaggle_push_benchmark_bundle.py --message "Thêm mô hình mới"
"""

import argparse
import json
import shutil
from pathlib import Path

from vtrack.kaggle_cli import kaggle_username, push_dataset

BUNDLE_DIR = Path("outputs/kaggle_gpu_bundle")
DATASET_SLUG = "vtrack-gpu-benchmark"
VIDEO = Path("data/raw/vntraffic/VNTraffic/VNTraffic_Original-video.mp4")
VIDEO_NAME = "vntraffic_1080p.mp4"  # tên trong gói; kernel truyền đường dẫn này cho benchmark_speed.py
WEIGHTS = ["weights/yolo26n_vn_v5_best.pt", "weights/yolo26s_vn_v5_best.pt", "weights/rtdetrv2_r18_vn_v5_best",
           "weights/yolo26s_vn_v6_best.pt"]  # YOLO26s-v6: mô hình chính, dùng cho video demo GPU (từ 15/10/2026)
FILES = ["scripts/benchmark_speed.py", "scripts/infer_video.py", "configs/botsort_tuned_nogmc.yaml", "configs/demo_clips.yaml"]
DEMO_CLIPS = [f"data/videos/pvd_goc_{c}.mp4" for c in "abc"] + [f"data/videos/ham_chui_dem_{i}.mp4" for i in (1, 2, 3)]


def copy_into_bundle(source: Path, relative: Path) -> None:
    target = BUNDLE_DIR / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))
    else:
        shutil.copy2(source, target)


def build_bundle() -> None:
    shutil.rmtree(BUNDLE_DIR, ignore_errors=True)
    copy_into_bundle(Path("src/vtrack"), Path("src/vtrack"))
    for path in [*FILES, *WEIGHTS, *DEMO_CLIPS]:
        copy_into_bundle(Path(path), Path(path))
    copy_into_bundle(VIDEO, Path("data") / VIDEO_NAME)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--message", default="Cập nhật gói đo tốc độ GPU", help="Ghi chú cho phiên bản mới")
    args = parser.parse_args()

    build_bundle()
    dataset_ref = f"{kaggle_username()}/{DATASET_SLUG}"
    metadata = {"title": DATASET_SLUG, "id": dataset_ref, "licenses": [{"name": "other"}]}
    (BUNDLE_DIR / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    push_dataset(BUNDLE_DIR, dataset_ref, args.message)


if __name__ == "__main__":
    main()
