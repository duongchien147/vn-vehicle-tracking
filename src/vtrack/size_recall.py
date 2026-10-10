"""Recall theo chiều cao xe (điểm ảnh): tách xem mô hình bỏ sót vì xe nhỏ ở xa hay vì lý do khác (đêm, che khuất...).

Chia theo chiều cao box như KITTI (độ khó theo chiều cao: >= 40 px dễ, >= 25 px vừa) thay vì theo diện tích như COCO,
vì xe máy hẹp và cao: chiều cao phản ánh khoảng cách tới camera tốt hơn diện tích.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from vtrack import CLASS_NAMES
from vtrack.coco_eval import EvalImage
from vtrack.error_analysis import classify_detections

# Mốc chiều cao (px): < 20 = xe máy rất xa (vài chục điểm ảnh), 32 = mốc "vật nhỏ" của COCO (32x32), 64 = gấp đôi
HEIGHT_EDGES_PX = (20, 32, 64)
ANALYSIS_CONF = 0.25  # ngưỡng mặc định khi chạy predict của ultralytics: đếm sót/nhầm ở ngưỡng dùng thật, không phải 0.001


@dataclass
class SizeBinRecall:
    height_bin: str  # vd. "<20", "20-32", ">=64"
    truth_count: int
    found: int  # khớp IoU >= min_iou và đúng lớp
    found_any_class: int  # khớp vị trí, lớp nào cũng được (tách lỗi "sót" khỏi lỗi "nhầm lớp")
    motorbike_as_car: int  # xe máy bị gọi là ô tô (lỗi đèn pha ban đêm)

    @property
    def recall(self) -> float:
        return self.found / self.truth_count if self.truth_count else float("nan")

    @property
    def recall_any_class(self) -> float:
        return self.found_any_class / self.truth_count if self.truth_count else float("nan")


def height_bin(height_px: float, edges: Sequence[int] = HEIGHT_EDGES_PX) -> str:
    """Tên khoảng chiều cao chứa height_px: '<20', '20-32', '32-64', '>=64'."""
    if height_px < edges[0]:
        return f"<{edges[0]}"
    for low, high in zip(edges, edges[1:]):
        if height_px < high:
            return f"{low}-{high}"
    return f">={edges[-1]}"


def bin_names(edges: Sequence[int] = HEIGHT_EDGES_PX) -> list[str]:
    return [f"<{edges[0]}", *(f"{low}-{high}" for low, high in zip(edges, edges[1:])), f">={edges[-1]}"]


def recall_by_height(
    images: Sequence[EvalImage], min_iou: float = 0.5, conf: float = ANALYSIS_CONF,
    edges: Sequence[int] = HEIGHT_EDGES_PX,
) -> list[SizeBinRecall]:
    """Recall của các box nhãn (không tính vùng ignore) theo khoảng chiều cao, từ `images[i].predictions`."""
    bins = {name: SizeBinRecall(name, 0, 0, 0, 0) for name in bin_names(edges)}
    motorbike, car = CLASS_NAMES.index("motorbike"), CLASS_NAMES.index("car")
    for image in images:
        predictions = [p for p in image.predictions if p.confidence >= conf]
        outcome = classify_detections(image.truth, predictions, min_iou)
        for truth in image.truth:
            bins[height_bin(truth.height * image.height, edges)].truth_count += 1
        for truth, _ in outcome.correct:
            row = bins[height_bin(truth.height * image.height, edges)]
            row.found += 1
            row.found_any_class += 1
        for truth, prediction in outcome.wrong_class:
            row = bins[height_bin(truth.height * image.height, edges)]
            row.found_any_class += 1
            row.motorbike_as_car += truth.class_id == motorbike and prediction.class_id == car
    return list(bins.values())
