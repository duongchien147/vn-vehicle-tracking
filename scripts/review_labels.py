"""Rà soát nhãn người gán của một tập (vd. data/labeling/hanoi/test): box trùng, box bất thường,
thay đổi so với nhãn sơ bộ, và chỗ đáng ngờ do mô hình chỉ ra (xe có thể bị sót / box có thể thừa).
Kết quả: outputs/label_review/<tập>/review.csv + ảnh đánh dấu chỗ đáng ngờ để người gán mở đúng chỗ mà sửa.

Ví dụ:
    python scripts/review_labels.py data/labeling/hanoi/test
"""

import argparse
import csv
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from vtrack import CLASS_NAMES, IGNORE_CLASS_ID, IGNORE_LABEL
from vtrack.label_review import (
    YoloBox,
    best_iou,
    deduplicate,
    find_duplicates,
    invalid_boxes,
    match_boxes,
    overlap_fraction,
    read_yolo_file,
)
from vtrack.models import class_mapping

# (trọng số, imgsz) của các mô hình dùng để dò lỗi: dùng hai mô hình khác nhau để bớt phụ thuộc điểm mù của một mô hình
REVIEW_MODELS = [("weights/yolo26n_vn_v2_best.pt", 960), ("weights/yolo26n.pt", 1280)]
PREDICT_CONF = 0.1  # lấy cả dự đoán điểm thấp: box người gán không khớp dự đoán nào kể cả ở mức này mới đáng ngờ
MISSED_CONF = 0.5  # chỉ coi là "có thể bị sót" khi mô hình khá chắc chắn
MATCH_IOU = 0.3  # ngưỡng lỏng: box người gán và mô hình vẽ lệch nhau một chút vẫn tính là cùng một xe
# Dự đoán nằm >= 50% diện tích trong vùng ignore thì coi như thuộc vùng đó (không báo "sót")
IGNORE_OVERLAP = 0.5
SUSPECT_COLOR = {"sót?": (0, 0, 255), "thừa?": (255, 0, 255), "trùng": (0, 165, 255), "lỗi box": (0, 255, 255)}
LABEL_NAMES = [*CLASS_NAMES, IGNORE_LABEL]  # tên theo id trong nhãn gán tay (có thêm lớp ignore)


def predict(models: list[tuple[YOLO, int, dict[int, int]]], image: np.ndarray) -> list[YoloBox]:
    boxes = []
    for model, imgsz, mapping in models:
        result = cast(Results, list(model.predict(image, imgsz=imgsz, conf=PREDICT_CONF, classes=list(mapping), verbose=False))[0])
        detected = result.boxes
        assert detected is not None  # mô hình phát hiện đối tượng luôn trả về boxes
        for cls, conf, (x, y, w, h) in zip(detected.cls.tolist(), detected.conf.tolist(), detected.xywhn.tolist()):
            boxes.append(YoloBox(mapping[int(cls)], x, y, w, h, conf))
    return boxes


def in_ignore_region(box: YoloBox, ignore_regions: list[YoloBox]) -> bool:
    """Box nằm phần lớn (>= IGNORE_OVERLAP) trong một vùng ignore do người gán đánh dấu."""
    return any(overlap_fraction(box, region) >= IGNORE_OVERLAP for region in ignore_regions)


def draw(
    image: np.ndarray, human: list[YoloBox], ignore_regions: list[YoloBox], suspects: list[tuple[str, YoloBox]]
) -> np.ndarray:
    """Box người gán: xanh mảnh; vùng ignore: xám; chỗ đáng ngờ: khung dày có số thứ tự và màu theo loại."""
    height, width = image.shape[:2]
    canvas = image.copy()

    def corners(b: YoloBox) -> tuple[tuple[int, int], tuple[int, int]]:
        return ((int((b.center_x - b.width / 2) * width), int((b.center_y - b.height / 2) * height)),
                (int((b.center_x + b.width / 2) * width), int((b.center_y + b.height / 2) * height)))

    for b in human:
        cv2.rectangle(canvas, *corners(b), (0, 200, 0), 1)
    for b in ignore_regions:
        cv2.rectangle(canvas, *corners(b), (160, 160, 160), 2)
    for number, (kind, b) in enumerate(suspects, start=1):
        top_left, bottom_right = corners(b)
        cv2.rectangle(canvas, (top_left[0] - 4, top_left[1] - 4), (bottom_right[0] + 4, bottom_right[1] + 4), SUSPECT_COLOR[kind], 3)
        cv2.putText(canvas, f"{number}:{kind} {LABEL_NAMES[b.class_id]}", (top_left[0], max(20, top_left[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, SUSPECT_COLOR[kind], 2)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("split_dir", help="Thư mục chứa images/ và labels/ (nhãn người gán)")
    parser.add_argument("--prelabel-subdir", default="labels_prelabel_v2", help="Thư mục nhãn sơ bộ để so sánh")
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    out_dir = Path("outputs/label_review") / split_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    models = [(m := YOLO(weights), imgsz, class_mapping(m.names)) for weights, imgsz in REVIEW_MODELS]

    rows = []
    for image_path in sorted((split_dir / "images").glob("*.jpg")):
        image = cv2.imread(str(image_path))
        if image is None:
            raise SystemExit(f"Không đọc được ảnh: {image_path}")
        height, width = image.shape[:2]
        all_labels = read_yolo_file(split_dir / "labels" / f"{image_path.stem}.txt")
        human = [b for b in all_labels if b.class_id != IGNORE_CLASS_ID]  # box xe người gán
        ignore_regions = [b for b in all_labels if b.class_id == IGNORE_CLASS_ID]
        prelabel = read_yolo_file(split_dir / args.prelabel_subdir / f"{image_path.stem}.txt")
        predictions = predict(models, image)

        vs_prelabel = match_boxes(human, prelabel, min_iou=0.5)
        class_changed = sum(human[i].class_id != prelabel[j].class_id for i, j in vs_prelabel.pairs)
        # "có thể sót": mô hình khá chắc có xe nhưng không box người gán nào chồng lên chỗ đó,
        # và chỗ đó cũng không nằm trong vùng ignore (người gán đã chủ động đánh dấu là mơ hồ)
        missed = [
            p for p in predictions
            if p.confidence >= MISSED_CONF and best_iou(p, human) < MATCH_IOU and not in_ignore_region(p, ignore_regions)
        ]
        missed_unique = deduplicate(missed)  # hai mô hình có thể cùng chỉ ra một xe bị sót
        # "có thể thừa": box người gán mà không mô hình nào thấy gì ở đó, kể cả dự đoán điểm thấp
        extra = [h for h in human if best_iou(h, predictions) < MATCH_IOU]
        duplicates = find_duplicates(human)
        invalid = invalid_boxes(all_labels, width, height)

        suspects = ([("sót?", b) for b in missed_unique] + [("thừa?", b) for b in extra]
                    + [("trùng", human[j]) for _, j in duplicates] + [("lỗi box", all_labels[i]) for i in invalid])
        if suspects:
            annotated = draw(image, human, ignore_regions, suspects)
            cv2.imwrite(str(out_dir / f"{image_path.stem}.jpg"), annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])

        unchanged = not vs_prelabel.unmatched_reference and not vs_prelabel.unmatched_candidate and class_changed == 0
        rows.append({
            "image": image_path.name, "boxes": len(human), "ignore_regions": len(ignore_regions),
            "added_vs_prelabel": len(vs_prelabel.unmatched_reference),
            "removed_vs_prelabel": len(vs_prelabel.unmatched_candidate), "class_changed": class_changed,
            "unchanged_vs_prelabel": unchanged, "maybe_missed": len(missed_unique), "maybe_extra": len(extra),
            "duplicates": len(duplicates), "invalid": len(invalid),
        })

    with (out_dir / "review.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'ảnh':22s} {'box':>4s} {'ignore':>6s} {'+thêm':>5s} {'-xoá':>5s} {'đổi lớp':>7s} {'sót?':>5s} {'thừa?':>5s} {'trùng':>5s} {'lỗi':>4s}")
    for r in rows:
        flag = "  <- chưa sửa gì so với nhãn sơ bộ?" if r["unchanged_vs_prelabel"] else ""
        print(f"{r['image']:22s} {r['boxes']:4d} {r['ignore_regions']:6d} {r['added_vs_prelabel']:5d} {r['removed_vs_prelabel']:5d} {r['class_changed']:7d} "
              f"{r['maybe_missed']:5d} {r['maybe_extra']:5d} {r['duplicates']:5d} {r['invalid']:4d}{flag}")
    print(f"\nẢnh đánh dấu chỗ đáng ngờ: {out_dir}")


if __name__ == "__main__":
    main()
