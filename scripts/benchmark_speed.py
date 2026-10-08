"""Thí nghiệm E2: tốc độ trên CPU của các mô hình (YOLO PyTorch/OpenVINO, RT-DETRv2) — riêng detector và cả pipeline
tracking — kèm số tham số và GFLOPs. Cách đo chống thermal throttling của CPU laptop: xem docstring vtrack/speed.py.
Chạy khi máy rảnh và đang cắm sạc (không có tiến trình nặng khác), vì kết quả phụ thuộc tải và chế độ nguồn của CPU.
Kết quả: outputs/benchmark_speed/<--name>/results.csv (mỗi mô hình một dòng), timings.csv (từng frame, để kiểm tra
xu hướng chậm dần), run_config.json (thông tin CPU, phiên bản thư viện).

Ví dụ:
    python scripts/benchmark_speed.py --name e2_architectures --models weights/yolo26n_vn_v5_best.pt \
        weights/yolo26s_vn_v5_best.pt weights/rtdetrv2_r18_v5_best
"""

import argparse
import csv
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import openvino
import torch
import transformers
import ultralytics

from vtrack.detection_benchmark import ModelSpec
from vtrack.detectors import is_rtdetr_dir, load_detector
from vtrack.models import load_model
from vtrack.speed import COOLDOWN_S, ROUND_FRAMES, ROUNDS, WARMUP_FRAMES, measure_detectors, measure_pipeline, summarize
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
    parser.add_argument("--tracker", default=DEFAULT_TRACKER)
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--round-frames", type=int, default=ROUND_FRAMES)
    parser.add_argument("--cooldown", type=float, default=COOLDOWN_S, help="Giây nghỉ giữa các đợt")
    parser.add_argument("--no-pipeline", action="store_true", help="Chỉ đo detector")
    parser.add_argument("--name", default="default", help="Tên thí nghiệm -> outputs/benchmark_speed/<tên>")
    args = parser.parse_args()

    timed_frames = args.rounds * args.round_frames
    frames = [read_frame(args.video, index) for index in range(timed_frames)]
    specs = {spec_text: ModelSpec.parse(spec_text) for spec_text in args.models}
    detectors = {spec_text: load_detector(spec.weights, spec.imgsz) for spec_text, spec in specs.items()}

    print(f"Đo detector: {args.rounds} đợt x {args.round_frames} frame, nghỉ {args.cooldown:.0f} s giữa đợt, "
          f"thứ tự mô hình đảo chiều mỗi đợt", flush=True)
    timings = measure_detectors(detectors, frames, DEFAULT_DETECTION_CONF, args.rounds, args.round_frames, args.cooldown)

    rows = []
    for spec_text, detector in detectors.items():
        own = [t for t in timings if t.model == spec_text]
        speed = summarize([t.ms for t in own])
        # Trung vị từng đợt: tăng dần qua các đợt = máy nóng lên, kết quả cần đọc thận trọng
        round_medians = [float(np.median([t.ms for t in own if t.round_index == r])) for r in range(args.rounds)]
        complexity = detector.complexity() if detector.complexity else None
        if args.no_pipeline or is_rtdetr_dir(specs[spec_text].weights):
            # Pipeline tracking dùng model.track của ultralytics -> chỉ chạy được với YOLO
            pipeline = None
        else:
            time.sleep(args.cooldown)
            pipeline = measure_pipeline(load_model(specs[spec_text].weights), args.video, args.tracker, detector.imgsz,
                                        DEFAULT_DETECTION_CONF, timed_frames + WARMUP_FRAMES)
        row = {
            "model": spec_text, "imgsz": detector.imgsz,
            "input_hw": "x".join(map(str, complexity.input_hw)) if complexity else "",
            "params_m": round(complexity.params_m, 2) if complexity else float("nan"),
            "gflops": round(complexity.gflops, 1) if complexity else float("nan"),
            **{f"detector_{k}": round(v, 2) for k, v in asdict(speed).items()},
            "detector_round_medians_ms": " ".join(f"{m:.1f}" for m in round_medians),
            **({f"pipeline_{k}": round(v, 2) for k, v in asdict(pipeline).items()} if pipeline else {}),
        }
        rows.append(row)
        iqr = speed.ms_q3 - speed.ms_q1
        pipeline_text = f"pipeline {pipeline.ms_per_frame:6.1f} ms ({pipeline.fps:4.1f} frame/s)" if pipeline else "pipeline -"
        complexity_text = f"{complexity.params_m:5.1f} M tham số, {complexity.gflops:6.1f} GFLOPs" if complexity else "-"
        print(f"  {spec_text:50s} @{detector.imgsz:<5d} detector {speed.ms_per_frame:6.1f} ms (IQR {iqr:4.1f}, "
              f"{speed.fps:4.1f} frame/s) | theo đợt {row['detector_round_medians_ms']} | {pipeline_text} | {complexity_text}",
              flush=True)

    out_dir = Path("outputs/benchmark_speed") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))  # mô hình không có pipeline thiếu vài cột
    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "timings.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "round_index", "frame_index", "ms"])
        writer.writeheader()
        writer.writerows(asdict(t) for t in timings)
    environment = {"cpu": platform.processor(), "python": platform.python_version(), "torch": torch.__version__,
                   "torch_threads": torch.get_num_threads(), "openvino": openvino.__version__,
                   "ultralytics": ultralytics.__version__, "transformers": transformers.__version__,
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    run_config = {**vars(args), "environment": environment}
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nĐã ghi {out_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
