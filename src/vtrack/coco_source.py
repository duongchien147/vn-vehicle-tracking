"""Đọc nhãn định dạng COCO JSON (vd. UVH-26) và đổi sang box YOLO theo lớp của dự án."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import json

from vtrack import CLASS_NAMES
from vtrack.coco_eval import EvalImage
from vtrack.label_review import YoloBox


@dataclass
class CocoImage:
    file_name: str
    width: int
    height: int
    boxes: list[tuple[str, YoloBox]]  # (tên lớp gốc, box YOLO chuẩn hoá; class_id chưa có ý nghĩa)


def load_coco(path: str | Path) -> list[CocoImage]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    category_names = {c["id"]: c["name"] for c in data["categories"]}
    annotations = defaultdict(list)
    for annotation in data["annotations"]:
        annotations[annotation["image_id"]].append(annotation)
    images = []
    for image in data["images"]:
        width, height = image["width"], image["height"]
        boxes = []
        for annotation in annotations[image["id"]]:
            left, top, box_width, box_height = annotation["bbox"]
            if box_width <= 0 or box_height <= 0:
                continue
            boxes.append((category_names[annotation["category_id"]], YoloBox(
                -1, (left + box_width / 2) / width, (top + box_height / 2) / height,
                box_width / width, box_height / height,
            )))
        images.append(CocoImage(image["file_name"], width, height, boxes))
    return images


def to_project_classes(image: CocoImage, class_map: dict[str, str]) -> list[YoloBox]:
    """Giữ box có lớp nằm trong `class_map` (tên gốc -> tên lớp dự án), bỏ các lớp còn lại."""
    return [
        YoloBox(CLASS_NAMES.index(class_map[name]), box.center_x, box.center_y, box.width, box.height)
        for name, box in image.boxes if name in class_map
    ]


def to_eval_image(image: CocoImage, class_map: dict[str, str], ignore_classes: set[str]) -> EvalImage:
    """Ảnh chấm detector: box lớp trong `class_map` là nhãn chuẩn, box lớp trong `ignore_classes` là vùng ignore."""
    ignore = [YoloBox(-1, b.center_x, b.center_y, b.width, b.height) for name, b in image.boxes if name in ignore_classes]
    return EvalImage(image.file_name, image.width, image.height, to_project_classes(image, class_map), ignore)
