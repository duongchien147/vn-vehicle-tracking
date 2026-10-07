import json

import pytest
import torch

from vtrack.detectors import is_rtdetr_dir
from vtrack.rtdetr import to_yolo_boxes


def test_is_rtdetr_dir_reads_model_type(tmp_path):
    rtdetr = tmp_path / "rtdetr"
    rtdetr.mkdir()
    (rtdetr / "config.json").write_text(json.dumps({"model_type": "rt_detr_v2"}), encoding="utf-8")
    other = tmp_path / "other_hf_model"
    other.mkdir()
    (other / "config.json").write_text(json.dumps({"model_type": "detr"}), encoding="utf-8")
    openvino = tmp_path / "yolo_openvino_model"  # thư mục OpenVINO của YOLO không có config.json
    openvino.mkdir()

    assert is_rtdetr_dir(rtdetr)
    assert not is_rtdetr_dir(other)
    assert not is_rtdetr_dir(openvino)
    assert not is_rtdetr_dir(tmp_path / "weights.pt")


def test_to_yolo_boxes_converts_xyxy_and_filters_score_and_class():
    scores = torch.tensor([0.9, 0.05, 0.8])
    labels = torch.tensor([1, 0, 7])  # lớp 7 không thuộc dự án
    boxes = torch.tensor([[0.1, 0.2, 0.3, 0.6], [0.0, 0.0, 0.1, 0.1], [0.5, 0.5, 0.6, 0.6]])
    class_ids = {0: 0, 1: 1}

    result = to_yolo_boxes(scores, labels, boxes, class_ids, conf=0.1)

    assert len(result) == 1
    box = result[0]
    assert box.class_id == 1
    assert (box.center_x, box.center_y) == pytest.approx((0.2, 0.4))
    assert (box.width, box.height) == pytest.approx((0.2, 0.4))
    assert box.confidence == pytest.approx(0.9)
