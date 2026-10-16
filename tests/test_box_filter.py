"""Test bộ lọc box trùng bất kể lớp (vtrack.box_filter)."""

from collections import defaultdict
from typing import cast

import numpy as np
from ultralytics import YOLO

from vtrack.box_filter import CALLBACK_EVENT, DuplicateBoxFilter, agnostic_nms_keep, parse_nms_iou, set_agnostic_nms


def test_keeps_higher_score_of_overlapping_pair_regardless_of_class():
    # Hai box gần trùng trên cùng một xe (vd. một box "xe máy", một box "ô tô"): giữ box điểm cao hơn
    xyxy = np.array([[0, 0, 10, 10], [1, 1, 10, 10]], dtype=float)
    scores = np.array([0.3, 0.8])
    assert agnostic_nms_keep(xyxy, scores, 0.5).tolist() == [1]


def test_keeps_separate_and_lightly_overlapping_boxes():
    # Box cách xa và box chồng ít (IoU 1/3) đều được giữ, theo thứ tự gốc
    xyxy = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [100, 100, 110, 110]], dtype=float)
    scores = np.array([0.9, 0.2, 0.1])
    assert agnostic_nms_keep(xyxy, scores, 0.5).tolist() == [0, 1, 2]


def test_threshold_is_strict_upper_bound():
    # IoU đúng bằng ngưỡng thì vẫn giữ (chỉ bỏ khi IoU > ngưỡng)
    xyxy = np.array([[0, 0, 10, 10], [5, 0, 15, 10]], dtype=float)  # IoU = 50 / 150
    scores = np.array([0.9, 0.5])
    assert agnostic_nms_keep(xyxy, scores, 1 / 3).tolist() == [0, 1]
    assert agnostic_nms_keep(xyxy, scores, 0.3).tolist() == [0]


class FakeModel:
    def __init__(self) -> None:
        self.callbacks: defaultdict[str, list] = defaultdict(list)


def test_set_agnostic_nms_runs_before_tracker_and_does_not_stack():
    model = FakeModel()
    tracker_callback = object()
    model.callbacks[CALLBACK_EVENT].append(tracker_callback)  # tracker đã đăng ký từ trước (vd. app dùng lại model)
    set_agnostic_nms(cast(YOLO, model), 0.7)
    set_agnostic_nms(cast(YOLO, model), 0.5)
    callbacks = model.callbacks[CALLBACK_EVENT]
    assert len(callbacks) == 2 and callbacks[1] is tracker_callback
    assert isinstance(callbacks[0], DuplicateBoxFilter) and callbacks[0].iou_threshold == 0.5
    set_agnostic_nms(cast(YOLO, model), None)
    assert model.callbacks[CALLBACK_EVENT] == [tracker_callback]


def test_parse_nms_iou_reads_none_as_disabled():
    assert parse_nms_iou("none") is None
    assert parse_nms_iou("None") is None
    assert parse_nms_iou("0.6") == 0.6
