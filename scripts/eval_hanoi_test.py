"""Chấm mAP kiểu COCO (có vùng ignore) cho nhiều mô hình trên một tập frame Hà Nội có nhãn người (mặc định: test).
Mỗi mô hình chạy ở imgsz lúc train (hoặc chỉ định bằng `@imgsz`). Mô hình COCO được gộp box người lái vào box
xe máy trước khi chấm, để cùng cách vẽ với nhãn chuẩn (nếu không sẽ bị chấm thấp oan vì khác quy ước).
Kết quả: outputs/eval_hanoi/[<--name>/]<tập>/results.csv + run_config.json.

Ví dụ:
    python scripts/eval_hanoi_test.py
    python scripts/eval_hanoi_test.py --models weights/yolo26n.pt@1280 weights/yolo26n_vn_v2_best.pt --bootstrap 200
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from vtrack import IGNORE_CLASS_ID
from vtrack.coco_eval import EvalImage, instance_counts, well_represented_classes
from vtrack.detection_benchmark import EVAL_CONF, ModelSpec, describe, score_model
from vtrack.label_review import read_yolo_file

DEFAULT_MODELS = [
    "weights/yolo26n.pt@640",
    "weights/yolo26n.pt@1280",
    "weights/yolo26n_vn_best.pt",
    "weights/yolo26n_vn_v2_best.pt",
    "weights/yolo26n_vn_v3_best.pt",
]
# Lớp có ít hơn số box này trong tập test cho AP quá dao động (test Hà Nội: bus 5, truck 2 box)
MIN_INSTANCES_FOR_MAIN_MAP = 30


def load_split(split_dir: Path) -> list[tuple[EvalImage, np.ndarray]]:
    """Ảnh (đọc một lần, dùng chung cho mọi mô hình) kèm nhãn người, tách box xe và vùng ignore."""
    loaded = []
    for image_path in sorted((split_dir / "images").glob("*.jpg")):
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise SystemExit(f"Không đọc được ảnh: {image_path}")
        labels = read_yolo_file(split_dir / "labels" / f"{image_path.stem}.txt")
        image = EvalImage(
            file_name=image_path.name, width=frame.shape[1], height=frame.shape[0],
            truth=[b for b in labels if b.class_id != IGNORE_CLASS_ID],
            ignore=[b for b in labels if b.class_id == IGNORE_CLASS_ID],
        )
        loaded.append((image, frame))
    return loaded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split-dir", default="data/labeling/hanoi/test")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="trọng số[@imgsz]")
    # Mỗi lần chấm 30 ảnh mất ~15–25s trên CPU (pycocotools viết bằng Python) -> 50 lần ~15–20 phút mỗi mô hình
    parser.add_argument("--bootstrap", type=int, default=50, help="Số lần lấy mẫu lại cho khoảng tin cậy (0 = bỏ qua)")
    parser.add_argument("--name", help="Tên thí nghiệm -> outputs/eval_hanoi/<tên>/<tập> (không ghi đè bảng cũ)")
    parser.add_argument("--min-instances", type=int, default=MIN_INSTANCES_FOR_MAIN_MAP,
                        help="mAP chính chỉ tính trên lớp có ít nhất bấy nhiêu box nhãn chuẩn")
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    loaded = load_split(split_dir)
    images = [image for image, _ in loaded]
    out_dir = Path("outputs/eval_hanoi") / (args.name or "") / split_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = instance_counts(images)
    main_classes = well_represented_classes(images, args.min_instances)
    print(f"{len(images)} ảnh, {sum(len(i.ignore) for i in images)} vùng ignore, số box: {counts}")
    print(f"mAP chính tính trên các lớp có >= {args.min_instances} box: {main_classes}")

    rows = []
    for spec in args.models:
        score = score_model(ModelSpec.parse(spec), images, lambda i: loaded[i][1], main_classes, args.bootstrap)
        rows.append(score.to_row(counts))
        print(describe(score), flush=True)

    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    run_config = {**vars(args), "eval_conf": EVAL_CONF, "n_images": len(images)}
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nĐã ghi {out_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
