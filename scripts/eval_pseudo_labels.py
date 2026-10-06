"""Đo độ chính xác của nhãn tự động (labels_auto_raw, có điểm) so với nhãn người đã kiểm (labels) trên cùng tập,
ở nhiều ngưỡng điểm -> chọn keep_conf cho nhãn giả của tập train. Chỉ chạy trên VAL, không bao giờ trên test
(chọn ngưỡng bằng test làm tập test bị "nhiễm").

Ví dụ:
    python scripts/eval_pseudo_labels.py data/labeling/hanoi/val
"""

import argparse
import csv
from pathlib import Path

from vtrack import CLASS_NAMES, IGNORE_CLASS_ID
from vtrack.detection_eval import MatchCounts, match_detections
from vtrack.label_review import read_scored_yolo_file, read_yolo_file

THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("split_dir")
    parser.add_argument("--raw-subdir", default="labels_auto_raw")
    args = parser.parse_args()
    split_dir = Path(args.split_dir)
    if split_dir.name == "test":
        raise SystemExit("Từ chối: không chọn ngưỡng bằng tập test.")

    frames = []
    for label_path in sorted((split_dir / "labels").glob("*.txt")):
        human = read_yolo_file(label_path)
        truth = [b for b in human if b.class_id != IGNORE_CLASS_ID]
        ignore = [b for b in human if b.class_id == IGNORE_CLASS_ID]
        frames.append((truth, ignore, read_scored_yolo_file(split_dir / args.raw_subdir / label_path.name)))

    rows = []
    for threshold in THRESHOLDS:
        overall, per_class = MatchCounts(), {name: MatchCounts() for name in CLASS_NAMES}
        for truth, ignore, predicted in frames:
            kept = [b for b in predicted if b.confidence >= threshold]
            overall.add(match_detections(kept, truth, ignore))
            for class_id, name in enumerate(CLASS_NAMES):
                per_class[name].add(match_detections(
                    [b for b in kept if b.class_id == class_id], [b for b in truth if b.class_id == class_id], ignore
                ))
        motorbike = per_class["motorbike"]
        rows.append({
            "threshold": threshold, "precision": round(overall.precision, 3), "recall": round(overall.recall, 3),
            "f1": round(overall.f1, 3), "motorbike_precision": round(motorbike.precision, 3),
            "motorbike_recall": round(motorbike.recall, 3), "car_recall": round(per_class["car"].recall, 3),
            "false_positives": overall.false_positives, "false_negatives": overall.false_negatives,
        })

    out_dir = Path("outputs/pseudo_label_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split_dir.parent.name}_{split_dir.name}_{args.raw_subdir}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'ngưỡng':>6s} {'P':>6s} {'R':>6s} {'F1':>6s} {'P xe máy':>9s} {'R xe máy':>9s} {'R ô tô':>7s} {'sai':>5s} {'sót':>5s}")
    for r in rows:
        print(f"{r['threshold']:6.2f} {r['precision']:6.3f} {r['recall']:6.3f} {r['f1']:6.3f} {r['motorbike_precision']:9.3f} "
              f"{r['motorbike_recall']:9.3f} {r['car_recall']:7.3f} {r['false_positives']:5d} {r['false_negatives']:5d}")
    print(f"\nĐã ghi {out_path}")


if __name__ == "__main__":
    main()
