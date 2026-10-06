"""Tạo nhãn sơ bộ (pre-label) cho ảnh cần gán nhãn bằng một mô hình có sẵn, để người gán nhãn chỉ phải sửa
thay vì vẽ từ đầu. Ghi nhãn YOLO (4 lớp dự án) vào thư mục labels/ cạnh images/.

Ví dụ:
    python scripts/prelabel.py data/labeling/hanoi --weights weights/yolo26n_vn_v2_best.pt --imgsz 960
"""

import argparse
from pathlib import Path
from typing import cast

from ultralytics import YOLO
from ultralytics.engine.results import Results

from vtrack import CLASS_NAMES, IGNORE_LABEL
from vtrack.label_review import HUMAN_VERIFIED_MARKER, is_human_verified
from vtrack.models import DEFAULT_WEIGHTS, FALLBACK_IMGSZ, class_mapping, training_imgsz


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", help="Thư mục chứa <split>/images/")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--imgsz", type=int, default=None, help="Mặc định: đúng imgsz lúc train/export của mô hình")
    # Ngưỡng thấp vừa phải: xoá một box thừa nhanh hơn nhiều so với vẽ thêm một box bị sót
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--splits", nargs="+", help="Chỉ pre-label các tập này (mặc định: tất cả). "
                        "Luôn chỉ định khi đã có tập được gán nhãn tay, nếu không nhãn tay sẽ bị ghi đè")
    args = parser.parse_args()

    model = YOLO(args.weights)
    imgsz = args.imgsz or training_imgsz(model) or FALLBACK_IMGSZ
    print(f"Pre-label bằng {args.weights}, imgsz={imgsz}")
    mapping = class_mapping(model.names)
    total_images, total_boxes = 0, 0
    for images_dir in sorted(Path(args.root).glob("*/images")):
        if args.splits and images_dir.parent.name not in args.splits:
            continue
        if is_human_verified(images_dir.parent):
            raise SystemExit(f"Từ chối: {images_dir.parent} có nhãn người đã kiểm ({HUMAN_VERIFIED_MARKER}), không ghi đè.")
        labels_dir = images_dir.parent / "labels"
        labels_dir.mkdir(exist_ok=True)
        for image_path in sorted(images_dir.glob("*.jpg")):
            results = list(model.predict(str(image_path), imgsz=imgsz, conf=args.conf, classes=list(mapping), verbose=False))
            boxes = cast(Results, results[0]).boxes
            assert boxes is not None  # mô hình phát hiện đối tượng luôn trả về boxes
            lines = [
                f"{mapping[int(cls)]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                for cls, (x, y, w, h) in zip(boxes.cls.tolist(), boxes.xywhn.tolist())
            ]
            (labels_dir / f"{image_path.stem}.txt").write_text("\n".join(lines))
            total_images += 1
            total_boxes += len(lines)
        print(f"{images_dir.parent.name:5s}: đã pre-label {len(list(images_dir.glob('*.jpg')))} ảnh")

    # Tên lớp theo thứ tự id, định dạng mà các công cụ gán nhãn (CVAT, labelImg, X-AnyLabeling...) đọc được
    # Có cả lớp ignore (id 4) — nếu thiếu, nhãn ignore người đã gán sẽ hỏng khi chuyển từ X-AnyLabeling sang YOLO
    (Path(args.root) / "classes.txt").write_text("\n".join([*CLASS_NAMES, IGNORE_LABEL]), encoding="utf-8")
    print(f"Tổng: {total_images} ảnh, {total_boxes} box (trung bình {total_boxes / max(total_images, 1):.0f} box/ảnh)")


if __name__ == "__main__":
    main()
