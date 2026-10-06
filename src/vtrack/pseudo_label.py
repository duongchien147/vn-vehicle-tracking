"""Gán nhãn tự động (pseudo-label) cho tập train: ghép dự đoán của nhiều mô hình "thầy" bằng Weighted Boxes Fusion
(WBF), kèm TTA lật ngang. Cách làm chuẩn của học bán giám sát (Noisy Student, knowledge distillation): nhãn giả
chỉ dùng cho train; val/test luôn phải có người kiểm."""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import cast

import cv2
import numpy as np
from ensemble_boxes import weighted_boxes_fusion
from ultralytics import YOLO
from ultralytics.engine.results import Results

from vtrack.dataset import merge_riders_into_motorbikes
from vtrack.label_review import YoloBox, deduplicate
from vtrack.models import class_mapping


@dataclass(frozen=True)
class Teacher:
    weights: str
    imgsz: int
    weight: float  # trọng số của mô hình khi gộp WBF (mô hình đáng tin hơn -> trọng số cao hơn)
    tiles: int = 1  # > 1: cắt ảnh thành lưới tiles x tiles ô (kiểu SAHI) để xe nhỏ/dày đặc to hơn khi vào mô hình


# Các ô cắt chồng lên nhau 20% kích thước ô (mặc định của SAHI) để xe nằm ở chỗ cắt vẫn trọn vẹn trong một ô
TILE_OVERLAP = 0.2
# Box cách mép cắt bên trong ít hơn số pixel này bị coi là bị cắt cụt -> bỏ (phiên bản đầy đủ nằm ở ô bên cạnh)
TILE_EDGE_MARGIN_PX = 2


@dataclass(frozen=True)
class TileWindow:
    left: int
    top: int
    right: int
    bottom: int


def tile_windows(width: int, height: int, tiles: int, overlap: float = TILE_OVERLAP) -> list[TileWindow]:
    """Chia ảnh thành lưới tiles x tiles ô chồng lấn nhau, phủ kín ảnh."""
    tile_w = int(width / (tiles - (tiles - 1) * overlap))
    tile_h = int(height / (tiles - (tiles - 1) * overlap))
    lefts = [round(i * (width - tile_w) / (tiles - 1)) for i in range(tiles)] if tiles > 1 else [0]
    tops = [round(i * (height - tile_h) / (tiles - 1)) for i in range(tiles)] if tiles > 1 else [0]
    return [TileWindow(x, y, x + tile_w, y + tile_h) for y in tops for x in lefts]


def tile_box_to_image(box: YoloBox, window: TileWindow, width: int, height: int) -> YoloBox:
    """Đổi box chuẩn hoá theo ô sang chuẩn hoá theo ảnh gốc."""
    tile_w, tile_h = window.right - window.left, window.bottom - window.top
    return replace(
        box,
        center_x=(window.left + box.center_x * tile_w) / width,
        center_y=(window.top + box.center_y * tile_h) / height,
        width=box.width * tile_w / width,
        height=box.height * tile_h / height,
    )


def touches_inner_edge(box: YoloBox, window: TileWindow, width: int, height: int) -> bool:
    """Box (chuẩn hoá theo ảnh gốc) chạm mép ô ở phía có ô khác bên cạnh -> nhiều khả năng bị cắt cụt."""
    x1, x2 = (box.center_x - box.width / 2) * width, (box.center_x + box.width / 2) * width
    y1, y2 = (box.center_y - box.height / 2) * height, (box.center_y + box.height / 2) * height
    margin = TILE_EDGE_MARGIN_PX
    return (
        (window.left > 0 and x1 <= window.left + margin)
        or (window.right < width and x2 >= window.right - margin)
        or (window.top > 0 and y1 <= window.top + margin)
        or (window.bottom < height and y2 >= window.bottom - margin)
    )


def flip_horizontal(boxes: Sequence[YoloBox]) -> list[YoloBox]:
    """Lật box theo chiều ngang (toạ độ chuẩn hoá): dùng để đưa dự đoán trên ảnh lật về ảnh gốc."""
    return [replace(box, center_x=1.0 - box.center_x) for box in boxes]


def predict_teacher(model: YOLO, image: np.ndarray, imgsz: int, conf: float) -> list[YoloBox]:
    """Dự đoán của một mô hình, đổi về lớp dự án.

    Mô hình COCO (có lớp person và motorcycle) vẽ người lái và xe thành hai box riêng, nên box người lái được gộp
    vào box xe máy trước — cho khớp cách vẽ của dữ liệu Việt Nam (một box bao cả người lẫn xe).
    """
    result = cast(Results, list(model.predict(image, imgsz=imgsz, conf=conf, verbose=False))[0])
    detected = result.boxes
    assert detected is not None  # mô hình phát hiện đối tượng luôn trả về boxes
    raw = [
        (int(cls), x1, y1, x2, y2)
        for cls, (x1, y1, x2, y2) in zip(detected.cls.tolist(), detected.xyxyn.tolist())
    ]
    scores = detected.conf.tolist()

    name_to_id = {name: model_id for model_id, name in model.names.items()}
    if "person" in name_to_id and "motorcycle" in name_to_id:
        raw = merge_riders_into_motorbikes(raw, motor_class=name_to_id["motorcycle"], rider_class=name_to_id["person"])

    mapping = class_mapping(model.names)
    return [
        YoloBox(mapping[cls], (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, score)
        for (cls, x1, y1, x2, y2), score in zip(raw, scores)
        if cls in mapping
    ]


def predict_teacher_tiled(model: YOLO, image: np.ndarray, imgsz: int, conf: float, tiles: int) -> list[YoloBox]:
    """Dự đoán kiểu SAHI: chạy mô hình trên từng ô, đưa box về ảnh gốc, bỏ box bị mép cắt làm cụt.

    Box của cùng một xe ở vùng chồng lấn giữa hai ô sẽ được WBF gộp ở bước sau.
    """
    height, width = image.shape[:2]
    boxes = []
    for window in tile_windows(width, height, tiles):
        crop = image[window.top:window.bottom, window.left:window.right]
        for box in predict_teacher(model, crop, imgsz, conf):
            full = tile_box_to_image(box, window, width, height)
            if not touches_inner_edge(full, window, width, height):
                boxes.append(full)
    return boxes


def fuse(
    predictions: Sequence[Sequence[YoloBox]], weights: Sequence[float], iou_thr: float, skip_box_thr: float
) -> list[YoloBox]:
    """Gộp dự đoán của nhiều nguồn bằng Weighted Boxes Fusion.

    Khác NMS (chỉ giữ box điểm cao nhất), WBF lấy trung bình có trọng số toạ độ các box chồng nhau -> box chính xác
    hơn. Với conf_type="avg", box chỉ một nguồn thấy bị giảm điểm theo tỉ lệ số nguồn -> bớt box sai.
    """
    boxes_list, scores_list, labels_list = [], [], []
    for source in predictions:
        corners = [
            [b.center_x - b.width / 2, b.center_y - b.height / 2, b.center_x + b.width / 2, b.center_y + b.height / 2]
            for b in source
        ]
        boxes_list.append(np.clip(np.array(corners, dtype=float).reshape(-1, 4), 0.0, 1.0))  # WBF yêu cầu toạ độ trong [0, 1]
        scores_list.append([b.confidence for b in source])
        labels_list.append([b.class_id for b in source])
    if not any(len(scores) for scores in scores_list):
        return []
    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list, weights=list(weights), iou_thr=iou_thr, skip_box_thr=skip_box_thr,
        conf_type="avg",
    )
    return [
        YoloBox(int(label), (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, float(score))
        for (x1, y1, x2, y2), score, label in zip(boxes, scores, labels)
    ]


def select_pseudo_labels(fused: Sequence[YoloBox], keep_conf: float, dedup_iou: float) -> list[YoloBox]:
    """Giữ box có điểm >= keep_conf, rồi bỏ box trùng chỗ khác lớp (giữ box điểm cao hơn).

    WBF gộp riêng từng lớp, nên khi các mô hình bất đồng lớp (một gọi xe máy, một gọi ô tô) sẽ còn hai box trùng
    vị trí; gộp trùng không phân biệt lớp sau WBF là bước hậu xử lý thường dùng.
    """
    kept = [box for box in fused if box.confidence >= keep_conf]
    return deduplicate(kept, min_iou=dedup_iou)


def pseudo_label_image(
    image: np.ndarray,
    teachers: Sequence[tuple[YOLO, Teacher]],
    flip_tta: bool,
    predict_conf: float,
    iou_thr: float,
    skip_box_thr: float,
) -> list[YoloBox]:
    """Nhãn giả cho một ảnh: mỗi mô hình chạy trên ảnh gốc (và ảnh lật nếu bật TTA), rồi gộp tất cả bằng WBF.

    TTA tự làm ở đây vì `augment=True` của ultralytics không hỗ trợ YOLO26 (mô hình end-to-end): nó chỉ cảnh báo
    rồi chạy một lần như thường.
    """
    def run(model: YOLO, teacher: Teacher, frame: np.ndarray) -> list[YoloBox]:
        if teacher.tiles > 1:
            return predict_teacher_tiled(model, frame, teacher.imgsz, predict_conf, teacher.tiles)
        return predict_teacher(model, frame, teacher.imgsz, predict_conf)

    predictions, weights = [], []
    for model, teacher in teachers:
        predictions.append(run(model, teacher, image))
        weights.append(teacher.weight)
        if flip_tta:
            predictions.append(flip_horizontal(run(model, teacher, cv2.flip(image, 1))))
            weights.append(teacher.weight)
    return fuse(predictions, weights, iou_thr, skip_box_thr)
