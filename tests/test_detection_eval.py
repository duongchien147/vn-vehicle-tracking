import pytest

from vtrack.detection_eval import MatchCounts, match_detections
from vtrack.label_review import YoloBox

CAR, MOTORBIKE = 1, 0


def box(class_id: int, x: float, y: float, confidence: float = 1.0, size: float = 0.1) -> YoloBox:
    return YoloBox(class_id, x, y, size, size, confidence)


def test_perfect_match():
    truth = [box(CAR, 0.2, 0.2), box(MOTORBIKE, 0.6, 0.6)]
    counts = match_detections([box(CAR, 0.2, 0.2, 0.9), box(MOTORBIKE, 0.6, 0.6, 0.8)], truth)
    assert (counts.true_positives, counts.false_positives, counts.false_negatives) == (2, 0, 0)
    assert counts.precision == counts.recall == counts.f1 == 1.0


def test_wrong_class_counts_as_false_positive_and_miss():
    counts = match_detections([box(MOTORBIKE, 0.2, 0.2, 0.9)], [box(CAR, 0.2, 0.2)])
    assert (counts.true_positives, counts.false_positives, counts.false_negatives) == (0, 1, 1)


def test_each_truth_matched_once_higher_score_wins():
    duplicate_predictions = [box(CAR, 0.2, 0.2, 0.9), box(CAR, 0.201, 0.2, 0.5)]
    counts = match_detections(duplicate_predictions, [box(CAR, 0.2, 0.2)])
    assert (counts.true_positives, counts.false_positives) == (1, 1)  # dự đoán trùng thứ hai là sai


def test_prediction_inside_ignore_region_is_not_false_positive():
    region = YoloBox(4, 0.8, 0.8, 0.3, 0.3)
    counts = match_detections([box(CAR, 0.8, 0.8, 0.9)], [], ignore_regions=[region])
    assert (counts.true_positives, counts.false_positives, counts.false_negatives) == (0, 0, 0)


def test_counts_add_and_handle_empty():
    total = MatchCounts()
    total.add(MatchCounts(true_positives=3, false_positives=1, false_negatives=1))
    total.add(MatchCounts(true_positives=1, false_positives=1, false_negatives=3))
    assert total.precision == pytest.approx(4 / 6)
    assert total.recall == pytest.approx(4 / 8)
    assert MatchCounts().f1 == 0.0
