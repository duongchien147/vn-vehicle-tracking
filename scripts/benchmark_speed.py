"""Thí nghiệm E2: tốc độ trên CPU của các mô hình (PyTorch và OpenVINO) — riêng detector và cả pipeline tracking.
Chạy khi máy rảnh (không có tiến trình nặng khác), vì kết quả phụ thuộc tải CPU.
Kết quả: outputs/benchmark_speed/results.csv + run_config.json (kèm thông tin CPU, phiên bản thư viện).

Ví dụ:
    python scripts/benchmark_speed.py
"""

import argparse
import csv
import json
import platform
from dataclasses import asdict
from pathlib import Path

import openvino
import torch
import ultralytics

from vtrack.detection_benchmark import ModelSpec
from vtrack.models import FALLBACK_IMGSZ, load_model, training_imgsz
from vtrack.speed import measure_detector, measure_pipeline
from vtrack.tracking import DEFAULT_DETECTION_CONF, DEFAULT_TRACKER
from vtrack.video_io import read_frame

DEFAULT_MODELS = [
    "weights/yolo26n_vn_v5_best.pt",
    "weights/yolo26n_vn_v5_best_openvino_model",
    "weights/yolo26n_vn_v4_best.pt",
    "weights/yolo26n_vn_v5_best.pt@640",  # cùng mô hình ở ảnh nhỏ hơn: đánh đổi tốc độ / độ chính xác
]
DEFAULT_VIDEO = "data/tracking_gt/pvd_goc_a/clip.mp4"  # 1080p, mật độ xe vừa phải


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="trọng số[@imgsz]")
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--frames", type=int, default=150, help="Số frame đo (cộng thêm frame khởi động)")
    parser.add_argument("--tracker", default=DEFAULT_TRACKER)
    args = parser.parse_args()

    frames = [read_frame(args.video, index) for index in range(args.frames + 10)]
    rows = []
    for spec_text in args.models:
        spec = ModelSpec.parse(spec_text)
        model = load_model(spec.weights)
        imgsz = spec.imgsz or training_imgsz(model) or FALLBACK_IMGSZ
        detector = measure_detector(model, frames, imgsz, DEFAULT_DETECTION_CONF)
        pipeline = measure_pipeline(model, args.video, args.tracker, imgsz, DEFAULT_DETECTION_CONF, len(frames))
        row = {
            "model": spec_text, "imgsz": imgsz,
            **{f"detector_{k}": round(v, 2) for k, v in asdict(detector).items()},
            **{f"pipeline_{k}": round(v, 2) for k, v in asdict(pipeline).items()},
        }
        rows.append(row)
        print(f"  {spec_text:50s} @{imgsz:<5d} detector {detector.ms_per_frame:6.1f} ms ({detector.fps:4.1f} frame/s) | "
              f"pipeline {pipeline.ms_per_frame:6.1f} ms ({pipeline.fps:4.1f} frame/s, p90 {pipeline.ms_p90:.0f} ms)",
              flush=True)

    out_dir = Path("outputs/benchmark_speed")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    environment = {"cpu": platform.processor(), "python": platform.python_version(), "torch": torch.__version__,
                   "torch_threads": torch.get_num_threads(), "openvino": openvino.__version__,
                   "ultralytics": ultralytics.__version__}
    run_config = {**vars(args), "environment": environment}
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nĐã ghi {out_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
