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

from vtrack.coco_eval import (
    BLOCK_SECONDS, BOOTSTRAP_ROUNDS, frame_name_blocks, instance_counts, well_represented_classes,
)
from vtrack.detection_benchmark import EVAL_CONF, ModelSpec, describe, load_split, score_model

DEFAULT_MODELS = [
    "weights/yolo26n.pt@640",
    "weights/yolo26n.pt@1280",
    "weights/yolo26n_vn_best.pt",
    "weights/yolo26n_vn_v2_best.pt",
    "weights/yolo26n_vn_v3_best.pt",
]
# Lớp có ít hơn số box này trong tập test cho AP quá dao động (test Hà Nội: bus 5, truck 2 box)
MIN_INSTANCES_FOR_MAIN_MAP = 30


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split-dir", default="data/labeling/hanoi/test")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="trọng số[@imgsz]")
    # Bootstrap chỉ gộp lại kết quả đã ghép của từng ảnh (vtrack.coco_eval.iou50_matches) nên 1.000 lần vẫn nhanh
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_ROUNDS,
                        help="Số lần lấy mẫu lại cho khoảng tin cậy (0 = bỏ qua)")
    parser.add_argument("--name", help="Tên thí nghiệm -> outputs/eval_hanoi/<tên>/<tập> (không ghi đè bảng cũ)")
    parser.add_argument("--min-instances", type=int, default=MIN_INSTANCES_FOR_MAIN_MAP,
                        help="mAP chính chỉ tính trên lớp có ít nhất bấy nhiêu box nhãn chuẩn")
    parser.add_argument("--agnostic-nms-iou", type=float,
                        help="Bật bộ lọc box trùng bất kể lớp với ngưỡng IoU này (vtrack.box_filter); mặc định tắt")
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
    # Frame cắt từ video: thêm khoảng tin cậy bootstrap theo khối BLOCK_SECONDS giây (frame gần nhau chung nhiều xe)
    blocks = frame_name_blocks([image.file_name for image in images])
    if blocks is not None:
        print(f"Bootstrap theo khối {BLOCK_SECONDS:g} s: {len(set(blocks))} khối")

    rows = []
    for spec in args.models:
        score = score_model(ModelSpec.parse(spec), images, lambda i: loaded[i][1], main_classes, args.bootstrap,
                            groups=blocks, agnostic_nms_iou=args.agnostic_nms_iou)
        rows.append(score.to_row(counts))
        print(describe(score), flush=True)

    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    run_config = {**vars(args), "eval_conf": EVAL_CONF, "n_images": len(images),
                  "block_seconds": BLOCK_SECONDS if blocks else None, "n_blocks": len(set(blocks)) if blocks else None}
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nĐã ghi {out_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
