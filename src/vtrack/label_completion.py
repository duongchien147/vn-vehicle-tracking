"""Bổ sung nhãn còn thiếu trong dataset gán nhãn chưa đầy đủ bằng dự đoán của mô hình "thầy".

Dataset công khai (vd. UVH-26) thường bỏ sót xe ở xa. Train thẳng thì các xe không có nhãn bị coi là nền,
mô hình học cách KHÔNG phát hiện chúng — đúng điểm yếu cần sửa. Cách chuẩn với nhãn thiếu là thêm nhãn giả
có độ tin cậy cao ở chỗ chưa có nhãn nào, giữ nguyên nhãn người gán.
"""

from dataclasses import dataclass

from vtrack.label_review import YoloBox, best_iou, overlap_fraction

# Chỉ thêm dự đoán rất chắc: không có tập kiểm để chọn ngưỡng trên dataset này, nên ưu tiên ít nhãn sai
DEFAULT_MIN_CONFIDENCE = 0.5
# Dự đoán chồng lên một box nhãn (kể cả box của lớp bị bỏ, vd. xe ba bánh) quá mức này = vật đã có nhãn
DEFAULT_MAX_OVERLAP_IOU = 0.3
# ...hoặc nằm phần lớn bên trong một box nhãn: box xe máy có nhãn bao cả người lái nên to hơn box "chỉ phần xe"
# của mô hình thầy, IoU thấp dù là cùng một xe (gặp khi kiểm tra bằng mắt)
DEFAULT_MAX_INSIDE_FRACTION = 0.5


@dataclass
class CompletedLabels:
    labels: list[YoloBox]  # nhãn gốc đã đổi sang lớp dự án + nhãn bổ sung
    added: list[YoloBox]  # riêng phần bổ sung (để thống kê, xem lại)


def complete_labels(
    labels: list[YoloBox],
    occupied: list[YoloBox],
    predictions: list[YoloBox],
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    max_overlap_iou: float = DEFAULT_MAX_OVERLAP_IOU,
    max_inside_fraction: float = DEFAULT_MAX_INSIDE_FRACTION,
) -> CompletedLabels:
    """`labels`: nhãn gốc thuộc lớp dự án. `occupied`: MỌI box gốc, kể cả lớp không dùng — để không thêm
    nhầm nhãn cho vật đã được gán là lớp khác (xe ba bánh bị mô hình COCO gọi là ô tô...)."""
    added = [
        box for box in predictions
        if box.confidence >= min_confidence
        and best_iou(box, occupied) < max_overlap_iou
        and all(overlap_fraction(box, region) < max_inside_fraction for region in occupied)
    ]
    # Bỏ trùng giữa các nhãn bổ sung (mô hình đôi khi ra 2 box cho một xe)
    deduplicated: list[YoloBox] = []
    for box in sorted(added, key=lambda b: b.confidence, reverse=True):
        if best_iou(box, deduplicated) < max_overlap_iou:
            deduplicated.append(box)
    return CompletedLabels(labels=labels + deduplicated, added=deduplicated)
