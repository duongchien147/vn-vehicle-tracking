"""Chấm mAP kiểu COCO cho nhiều mô hình trên frame lấy từ một clip có nhãn tracking gán tay (CVAT, có lớp xe),
vd. clip test PVD (TP.HCM) — tập test detector thứ hai, khác cảnh với test Hà Nội. Lấy 1 frame mỗi `--every-s` giây
để tránh các frame gần trùng. Kết quả: outputs/eval_clip_detection/[<--name>/]<sequence>/results.csv + run_config.json.

Ví dụ:
    python scripts/eval_clip_detection.py --sequence pvd_goc_a
"""

import argparse
import csv
import json
from pathlib import Path

import supervision as sv
import yaml

from vtrack import IGNORE_LABEL
from vtrack.coco_eval import BOOTSTRAP_ROUNDS, instance_counts, well_represented_classes
from vtrack.detection_benchmark import EVAL_CONF, ModelSpec, describe, frames_to_eval_images, score_model
from vtrack.mot_format import read_cvat_mot_labeled
from vtrack.video_io import read_frame

DEFAULT_MODELS = [
    "weights/yolo26n.pt@1280",
    "weights/yolo26n_vn_v2_best.pt",
    "weights/yolo26n_vn_v3_best.pt",
    "weights/yolo26n_vn_v4_best.pt",
    "weights/yolo26n_vn_v5_best.pt",
]
MIN_INSTANCES_FOR_MAIN_MAP = 30  # cùng ngưỡng với test Hà Nội


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/tracking_sequences.yaml")
    parser.add_argument("--sequence", default="pvd_goc_a")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="trọng số[@imgsz]")
    parser.add_argument("--every-s", type=float, default=1.0, help="Lấy 1 frame mỗi bấy nhiêu giây")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_ROUNDS)
    parser.add_argument("--name", default="", help="Tên thí nghiệm (thư mục con), để không ghi đè bảng cũ")
    args = parser.parse_args()

    spec = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["sequences"][args.sequence]
    if spec.get("format") != "cvat_mot":
        raise SystemExit(f"{args.sequence}: cần nhãn CVAT có lớp xe (format: cvat_mot)")
    labeled = read_cvat_mot_labeled(spec["ground_truth"], Path(spec["labels"]).read_text(encoding="utf-8").split())
    info = sv.VideoInfo.from_video_path(spec["video"])
    last_frame = max(box.frame for _, box in labeled)
    step = max(1, round(info.fps * args.every_s))
    frames = list(range(1, last_frame + 1, step))  # frame đánh từ 1 như MOTChallenge

    images = frames_to_eval_images(labeled, frames, info.width, info.height, IGNORE_LABEL)
    pixels = [read_frame(spec["video"], frame - 1) for frame in frames]  # ít frame -> đọc một lần, dùng chung
    counts = instance_counts(images)
    main_classes = well_represented_classes(images, MIN_INSTANCES_FOR_MAIN_MAP)
    print(f"{args.sequence}: {len(frames)} frame (mỗi {args.every_s:g} s), "
          f"{sum(len(i.ignore) for i in images)} vùng ignore, số box: {counts}; mAP chính trên {main_classes}")

    rows = []
    for model_spec in args.models:
        score = score_model(ModelSpec.parse(model_spec), images, lambda i: pixels[i], main_classes, args.bootstrap)
        rows.append(score.to_row(counts))
        print(describe(score), flush=True)

    out_dir = Path("outputs/eval_clip_detection") / args.name / args.sequence
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    run_config = {**vars(args), "sequence_spec": spec, "frames": frames, "eval_conf": EVAL_CONF}
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nĐã ghi {out_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
