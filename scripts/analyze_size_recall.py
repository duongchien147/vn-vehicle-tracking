"""Recall theo chiều cao xe trên các tập frame có nhãn người (mặc định vn_extra ngày + đêm), cho nhiều mô hình:
xe nhỏ ở xa bị sót bao nhiêu, xe lớn bị sót bao nhiêu, xe máy bị gọi nhầm thành ô tô bao nhiêu lần.
Ngưỡng điểm 0.25 (ngưỡng chạy thật), ghép ở IoU 0.5 và 0.3 (IoU 0.3: tách "sót thật" khỏi "box lệch").
Kết quả: outputs/size_recall/<--name>/results.csv (mỗi dòng = tập x mô hình x khoảng chiều cao) + run_config.json.

Ví dụ:
    python scripts/analyze_size_recall.py
    python scripts/analyze_size_recall.py --split-dirs data/labeling/hanoi/test --models weights/yolo26n_vn_v5_best.pt
"""

import argparse
import csv
import json
from pathlib import Path

from vtrack.detection_benchmark import ModelSpec, load_split
from vtrack.detectors import load_detector
from vtrack.size_recall import ANALYSIS_CONF, HEIGHT_EDGES_PX, recall_by_height

DEFAULT_SPLITS = ["data/labeling/vn_extra/day", "data/labeling/vn_extra/night"]
DEFAULT_MODELS = [
    "weights/yolo26n_vn_v3_best.pt",
    "weights/yolo26n_vn_v4_best.pt",
    "weights/yolo26n_vn_v5_best.pt",
    "weights/yolo26n_vn_v5_best.pt@1280",
    "weights/yolo26s_vn_v5_best.pt",
    "weights/yolo26s_vn_v5_best.pt@1280",
    "weights/rtdetrv2_r18_vn_v5_best",
]
LOOSE_IOU = 0.3  # nới ngưỡng ghép: recall tăng nhiều = box lệch; tăng ít = sót thật


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split-dirs", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="trọng số[@imgsz]")
    parser.add_argument("--conf", type=float, default=ANALYSIS_CONF)
    parser.add_argument("--name", default="default", help="Tên thí nghiệm -> outputs/size_recall/<tên>")
    args = parser.parse_args()
    out_dir = Path("outputs/size_recall") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {Path(d).name: load_split(Path(d)) for d in args.split_dirs}
    rows = []
    for spec in args.models:
        model = ModelSpec.parse(spec)
        detector = load_detector(model.weights, model.imgsz)
        for split, loaded in splits.items():
            images = [image for image, _ in loaded]
            for image, frame in loaded:
                image.predictions = detector.predict(frame, args.conf)
            strict = recall_by_height(images, min_iou=0.5, conf=args.conf)
            loose = recall_by_height(images, min_iou=LOOSE_IOU, conf=args.conf)
            for size_bin, loose_bin in zip(strict, loose):
                rows.append({
                    "split": split, "model": detector.name, "imgsz": detector.imgsz, "height_px": size_bin.height_bin,
                    "n_truth": size_bin.truth_count, "recall": round(size_bin.recall, 4),
                    "recall_any_class": round(size_bin.recall_any_class, 4),
                    f"recall_iou{int(LOOSE_IOU * 100)}": round(loose_bin.recall, 4),
                    "motorbike_as_car": size_bin.motorbike_as_car,
                })
            summary = " | ".join(f"{b.height_bin} {b.recall:.2f} (n={b.truth_count})" for b in strict)
            print(f"  {split:6s} {detector.name:40s} @{detector.imgsz}: {summary}", flush=True)

    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    run_config = {**vars(args), "height_edges_px": HEIGHT_EDGES_PX, "loose_iou": LOOSE_IOU}
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nĐã ghi {out_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
