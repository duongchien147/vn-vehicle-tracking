from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from ultralytics import YOLO

from vtrack.models import class_mapping, split_device, training_imgsz


def fake_model(overrides: dict, model_name: str = "") -> YOLO:
    """Chỉ cần hai thuộc tính mà training_imgsz đọc, không phải nạp mô hình thật."""
    return cast(YOLO, SimpleNamespace(overrides=overrides, model_name=model_name))


def test_class_mapping_by_name_for_coco_and_finetuned():
    coco = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    assert class_mapping(coco) == {3: 0, 2: 1, 5: 2, 7: 3}
    finetuned = {0: "motorbike", 1: "car", 2: "bus", 3: "truck"}
    assert class_mapping(finetuned) == {0: 0, 1: 1, 2: 2, 3: 3}


def test_training_imgsz_from_pt_overrides():
    assert training_imgsz(fake_model({"imgsz": 960})) == 960


def test_training_imgsz_from_export_metadata(tmp_path: Path):
    (tmp_path / "metadata.yaml").write_text("imgsz:\n- 960\n- 960\n", encoding="utf-8")
    assert training_imgsz(fake_model({}, str(tmp_path))) == 960


def test_training_imgsz_unknown():
    assert training_imgsz(fake_model({}, "khong_ton_tai")) is None


def test_split_device_suffix():
    assert split_device("weights/x_openvino_model#gpu") == ("weights/x_openvino_model", "intel:gpu")
    assert split_device("weights/x_openvino_model#cpu") == ("weights/x_openvino_model", "intel:cpu")
    assert split_device("weights/x.pt") == ("weights/x.pt", None)
    with pytest.raises(ValueError):
        split_device("weights/x_openvino_model#npu")
