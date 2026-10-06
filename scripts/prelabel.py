"""Tạo nhãn sơ bộ (pre-label) cho ảnh cần gán nhãn bằng một mô hình có sẵn, để người gán nhãn chỉ phải sửa
thay vì vẽ từ đầu. Ghi nhãn YOLO (4 lớp dự án) vào thư mục labels/ cạnh images/.
Mô hình COCO (vd. yolo26x.pt) được gộp box người lái vào box xe máy cho đúng quy ước nhãn của dự án.

Ví dụ:
    python scripts/prelabel.py data/labeling/hanoi --weights weights/yolo26n_vn_v2_best.pt --imgsz 960
"""

import argparse
from pathlib import Path

import cv2

from vtrack import CLASS_NAMES, IGNORE_LABEL
from vtrack.label_review import HUMAN_VERIFIED_MARKER, drop_nested_duplicates, is_human_verified, write_yolo_file
from vtrack.models import DEFAULT_WEIGHTS, FALLBACK_IMGSZ, load_model, training_imgsz
from vtrack.pseudo_label import predict_teacher


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

    model = load_model(args.weights)
    imgsz = args.imgsz or training_imgsz(model) or FALLBACK_IMGSZ
    print(f"Pre-label bằng {args.weights}, imgsz={imgsz}")
    total_images, total_boxes = 0, 0
    for images_dir in sorted(Path(args.root).glob("*/images")):
        if args.splits and images_dir.parent.name not in args.splits:
            continue
        if is_human_verified(images_dir.parent):
            raise SystemExit(f"Từ chối: {images_dir.parent} có nhãn người đã kiểm ({HUMAN_VERIFIED_MARKER}), không ghi đè.")
        labels_dir = images_dir.parent / "labels"
        labels_dir.mkdir(exist_ok=True)
        for image_path in sorted(images_dir.glob("*.jpg")):
            image = cv2.imread(str(image_path))
            if image is None:
                raise SystemExit(f"Không đọc được ảnh: {image_path}")
            # Bỏ box "chỉ phần xe" nằm trong box "xe + người lái" cùng lớp (mô hình COCO gộp người lái chưa hết)
            boxes = drop_nested_duplicates(predict_teacher(model, image, imgsz, args.conf))
            write_yolo_file(labels_dir / f"{image_path.stem}.txt", boxes)
            total_images += 1
            total_boxes += len(boxes)
        print(f"{images_dir.parent.name:5s}: đã pre-label {len(list(images_dir.glob('*.jpg')))} ảnh")

    # Tên lớp theo thứ tự id, định dạng mà các công cụ gán nhãn (CVAT, labelImg, X-AnyLabeling...) đọc được
    # Có cả lớp ignore (id 4) — nếu thiếu, nhãn ignore người đã gán sẽ hỏng khi chuyển từ X-AnyLabeling sang YOLO
    (Path(args.root) / "classes.txt").write_text("\n".join([*CLASS_NAMES, IGNORE_LABEL]), encoding="utf-8")
    print(f"Tổng: {total_images} ảnh, {total_boxes} box (trung bình {total_boxes / max(total_images, 1):.0f} box/ảnh)")


if __name__ == "__main__":
    main()
