"""Đối khớp dự đoán với nhãn chuẩn ở một ngưỡng điểm cố định -> đếm đúng/sai/sót, precision, recall, F1.

Cách đối khớp theo quy ước COCO: xét dự đoán theo điểm giảm dần, mỗi dự đoán ghép với box chuẩn cùng lớp chưa được
ghép có IoU lớn nhất (>= min_iou); dự đoán không ghép được mà nằm trong vùng ignore thì bỏ qua, không tính là sai.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from vtrack.label_review import YoloBox, iou, overlap_fraction


@dataclass
class MatchCounts:
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        """Trong các box dự đoán, bao nhiêu phần là đúng."""
        predicted = self.true_positives + self.false_positives
        return self.true_positives / predicted if predicted else 0.0

    @property
    def recall(self) -> float:
        """Trong các xe thật, bao nhiêu phần được phát hiện."""
        actual = self.true_positives + self.false_negatives
        return self.true_positives / actual if actual else 0.0

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0

    def add(self, other: "MatchCounts") -> None:
        self.true_positives += other.true_positives
        self.false_positives += other.false_positives
        self.false_negatives += other.false_negatives


def match_detections(
    predictions: Sequence[YoloBox],
    ground_truth: Sequence[YoloBox],
    ignore_regions: Sequence[YoloBox] = (),
    min_iou: float = 0.5,
    ignore_overlap: float = 0.5,
) -> MatchCounts:
    """Đếm đúng (TP) / sai (FP) / sót (FN) của `predictions` so với `ground_truth`.

    min_iou = 0.5 là ngưỡng của mAP50 (PASCAL VOC / COCO). Dự đoán nằm >= `ignore_overlap` diện tích trong một vùng
    ignore mà không khớp xe nào thì không bị tính là sai (giống cách COCO xử lý vùng iscrowd).
    """
    counts = MatchCounts()
    matched: set[int] = set()
    for prediction in sorted(predictions, key=lambda b: b.confidence, reverse=True):
        best_index, best_iou = None, min_iou
        for index, truth in enumerate(ground_truth):
            if index in matched or truth.class_id != prediction.class_id:
                continue
            score = iou(prediction, truth)
            if score >= best_iou:
                best_index, best_iou = index, score
        if best_index is not None:
            matched.add(best_index)
            counts.true_positives += 1
        elif not any(overlap_fraction(prediction, region) >= ignore_overlap for region in ignore_regions):
            counts.false_positives += 1
    counts.false_negatives = len(ground_truth) - len(matched)
    return counts
