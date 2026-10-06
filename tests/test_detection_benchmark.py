import pytest

from vtrack import CLASS_NAMES
from vtrack.detection_benchmark import ModelSpec, frames_to_eval_images
from vtrack.mot_format import MotBox


def test_model_spec_parses_optional_imgsz():
    assert ModelSpec.parse("weights/a.pt@1280") == ModelSpec("weights/a.pt", 1280)
    assert ModelSpec.parse("weights/a.pt") == ModelSpec("weights/a.pt", None)


def test_frames_to_eval_images_keeps_selected_frames_and_splits_ignore():
    labeled = [
        ("car", MotBox(1, 1, 100, 50, 200, 100)),
        ("ignore", MotBox(1, 9, 0, 0, 1000, 100)),
        ("car", MotBox(2, 1, 110, 50, 200, 100)),  # frame không được chọn
    ]

    (image,) = frames_to_eval_images(labeled, [1], width=1000, height=500, ignore_label="ignore")

    assert image.file_name == "frame_000001"
    assert len(image.truth) == 1 and image.truth[0].class_id == CLASS_NAMES.index("car")
    assert image.truth[0].center_x == pytest.approx(0.2) and image.truth[0].height == pytest.approx(0.2)
    assert len(image.ignore) == 1 and image.ignore[0].width == pytest.approx(1.0)
