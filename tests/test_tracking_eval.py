"""Kiểm tra bộ chấm trên ví dụ nhỏ có đáp án biết trước (2 xe chạy song song trong 10 frame)."""

import pytest

from vtrack.mot_format import MOT_DISTRACTOR_CLASS, MotBox
from vtrack.tracking_eval import evaluate_trackers, write_ground_truth_sequence, write_tracker_sequence

NUM_FRAMES = 10
SEQUENCE = "toy"


def car(frame: int, track_id: int, row: int) -> MotBox:
    """Xe đi ngang 10 px mỗi frame; `row` tách 2 xe theo chiều dọc để không chồng lấn."""
    return MotBox(frame, track_id, left=10.0 * frame, top=100.0 * row, width=50, height=50)


def run(tmp_path, ground_truth: list[MotBox], predictions: list[MotBox]):
    write_ground_truth_sequence(tmp_path / "gt", SEQUENCE, ground_truth, NUM_FRAMES, fps=30, width=640, height=480)
    write_tracker_sequence(tmp_path / "trackers", "candidate", SEQUENCE, predictions)
    return evaluate_trackers(tmp_path / "gt", tmp_path / "trackers", [SEQUENCE], ["candidate"], tmp_path / "out")["candidate"]


def two_cars(first_id: int = 1, second_id: int = 2, swap_at: int | None = None) -> list[MotBox]:
    boxes = []
    for frame in range(1, NUM_FRAMES + 1):
        swapped = swap_at is not None and frame >= swap_at
        boxes.append(car(frame, second_id if swapped else first_id, row=0))
        boxes.append(car(frame, first_id if swapped else second_id, row=1))
    return boxes


def test_perfect_tracker_scores_one(tmp_path):
    metrics = run(tmp_path, two_cars(), two_cars(first_id=7, second_id=9))

    assert metrics.hota == pytest.approx(1.0)
    assert metrics.mota == pytest.approx(1.0)
    assert metrics.idf1 == pytest.approx(1.0)
    assert metrics.id_switches == 0
    assert metrics.gt_ids == 2 and metrics.predicted_ids == 2


def test_swapping_ids_counts_two_switches_and_lowers_association(tmp_path):
    metrics = run(tmp_path, two_cars(), two_cars(swap_at=6))

    assert metrics.id_switches == 2
    assert metrics.det_a == pytest.approx(1.0)  # phát hiện vẫn đúng hết, chỉ ID sai
    assert metrics.ass_a < 1.0 and metrics.idf1 < 1.0


def test_prediction_on_distractor_box_is_neither_rewarded_nor_penalised(tmp_path):
    ground_truth = [car(f, 1, row=0) for f in range(1, NUM_FRAMES + 1) if f != 5]
    ground_truth.append(MotBox(5, 1, 50, 0, 50, 50, mot_class=MOT_DISTRACTOR_CLASS))  # nhãn bị rơi, đã lấp
    predictions = [car(f, 1, row=0) for f in range(1, NUM_FRAMES + 1)]

    metrics = run(tmp_path, ground_truth, predictions)

    assert metrics.false_positives == 0
    assert metrics.false_negatives == 0
    assert metrics.mota == pytest.approx(1.0)
