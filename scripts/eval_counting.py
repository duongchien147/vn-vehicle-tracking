"""Sai số đếm xe qua vạch trên clip có nhãn tracking gán tay (configs/tracking_sequences.yaml, định dạng cvat_mot):
chạy pipeline (đúng thứ demo chạy) rồi so số xe đếm được với số xe thật (track nhãn tay đi qua cùng bộ đếm),
theo từng lớp và hướng. Kết quả: outputs/eval_counting/<--name>/<sequence>/{results.csv, events.csv, run_config.json,
tracked.mp4}.

Vạch đếm mặc định là vạch mặc định của pipeline, KHÔNG chọn theo kết quả.

Ví dụ:
    python scripts/eval_counting.py --sequence pvd_goc_a
"""

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import yaml

from vtrack.count_eval import compare_counts, count_ground_truth
from vtrack.models import DEFAULT_WEIGHTS
from vtrack.mot_format import read_cvat_mot_labeled
from vtrack.pipeline import process_video
from vtrack.tracking import DEFAULT_AGNOSTIC_NMS_IOU, DEFAULT_MIN_TRACK_AGE, DEFAULT_TRACKER

DEFAULT_LINE = (0.0, 0.6, 1.0, 0.6)  # = mặc định của process_video


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/tracking_sequences.yaml")
    parser.add_argument("--sequence", default="pvd_goc_a")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--tracker", default=DEFAULT_TRACKER)
    parser.add_argument("--agnostic-nms-iou", type=float, default=DEFAULT_AGNOSTIC_NMS_IOU,
                        help="Ngưỡng IoU của bộ lọc box trùng trước tracker (vtrack.box_filter); mặc định như pipeline")
    parser.add_argument("--line", type=float, nargs=4, default=DEFAULT_LINE, metavar=("X1", "Y1", "X2", "Y2"),
                        help="Vạch đếm, toạ độ tương đối 0–1")
    parser.add_argument("--name", default="default", help="Tên thí nghiệm -> outputs/eval_counting/<tên>/<sequence>")
    args = parser.parse_args()

    sequence = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["sequences"][args.sequence]
    if sequence.get("format") != "cvat_mot":
        raise SystemExit("Cần nhãn có lớp xe (định dạng cvat_mot) để đếm theo lớp")
    label_names = Path(sequence["labels"]).read_text(encoding="utf-8").split()
    labeled = read_cvat_mot_labeled(sequence["ground_truth"], label_names)
    capture = cv2.VideoCapture(sequence["video"])
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    line = tuple(args.line)

    out_dir = Path("outputs/eval_counting") / args.name / args.sequence
    out_dir.mkdir(parents=True, exist_ok=True)
    truth = count_ground_truth(labeled, line, width, height)
    result = process_video(sequence["video"], out_dir / "tracked.mp4", args.weights, tracker=args.tracker, line=line,
                           agnostic_nms_iou=args.agnostic_nms_iou)
    rows = compare_counts(truth, result.counts())
    result.write_events_csv(out_dir / "events.csv")

    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "truth_in", "truth_out", "predicted_in", "predicted_out", "truth_total",
                         "predicted_total", "absolute_error", "relative_error"])
        for r in rows:
            writer.writerow([r.class_name, r.truth_in, r.truth_out, r.predicted_in, r.predicted_out, r.truth_total,
                             r.predicted_total, r.absolute_error, round(r.relative_error, 4)])
    run_config = {**vars(args), "line": line, "min_track_age": DEFAULT_MIN_TRACK_AGE, "frames": result.frames,
                  "imgsz": result.imgsz, "rows": [asdict(r) for r in rows]}
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{'lớp':10s} {'thật (vào/ra)':>14s} {'đếm (vào/ra)':>14s} {'sai số':>7s} {'tương đối':>9s}")
    for r in rows:
        print(f"{r.class_name:10s} {r.truth_in:>6d}/{r.truth_out:<7d} {r.predicted_in:>6d}/{r.predicted_out:<7d} "
              f"{r.absolute_error:>7d} {r.relative_error:>9.1%}")
    print(f"\nĐã ghi {out_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
