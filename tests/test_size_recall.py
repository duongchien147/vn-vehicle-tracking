import pytest

from vtrack.coco_eval import EvalImage
from vtrack.label_review import YoloBox
from vtrack.size_recall import bin_names, height_bin, recall_by_height

MOTORBIKE, CAR = 0, 1


def test_height_bins_cover_every_height():
    assert bin_names() == ["<20", "20-32", "32-64", ">=64"]
    assert [height_bin(h) for h in (5, 20, 31.9, 32, 63, 64, 500)] == ["<20", "20-32", "20-32", "32-64", "32-64",
                                                                       ">=64", ">=64"]


def test_recall_by_height_separates_missed_wrong_class_and_low_confidence():
    small = YoloBox(MOTORBIKE, 0.1, 0.1, 0.01, 0.01)  # cao 10 px trên ảnh 1000 px
    large = YoloBox(MOTORBIKE, 0.5, 0.5, 0.05, 0.1)  # cao 100 px
    other_large = YoloBox(CAR, 0.8, 0.8, 0.1, 0.1)
    image = EvalImage("a.jpg", 1000, 1000, [small, large, other_large], [], [
        YoloBox(CAR, 0.5, 0.5, 0.05, 0.1, 0.9),  # đúng chỗ xe máy lớn nhưng sai lớp
        YoloBox(CAR, 0.8, 0.8, 0.1, 0.1, 0.1),  # đúng ô tô nhưng dưới ngưỡng 0.25 -> không tính
    ])
    rows = {row.height_bin: row for row in recall_by_height([image])}
    assert rows["<20"].truth_count == 1 and rows["<20"].found == 0
    assert rows[">=64"].truth_count == 2
    assert rows[">=64"].found == 0
    assert rows[">=64"].found_any_class == 1
    assert rows[">=64"].motorbike_as_car == 1
    assert rows[">=64"].recall_any_class == pytest.approx(0.5)
    assert rows["20-32"].truth_count == 0 and rows["20-32"].recall != rows["20-32"].recall  # nan khi không có xe
