import pytest

from vtrack.label_review import YoloBox, best_iou, deduplicate, overlap_fraction, find_duplicates, invalid_boxes, iou, match_boxes

CAR, MOTORBIKE = 1, 0


def box(class_id: int, x: float, y: float, w: float = 0.1, h: float = 0.1) -> YoloBox:
    return YoloBox(class_id, x, y, w, h)


def test_iou_identical_disjoint_and_half_overlap():
    a = box(CAR, 0.5, 0.5)
    assert iou(a, a) == pytest.approx(1.0)  # so số thực phải có dung sai làm tròn
    assert iou(a, box(CAR, 0.9, 0.9)) == 0.0
    # dịch nửa chiều rộng: giao = 0.05*0.1, hợp = 2*0.01 - 0.005 -> IoU = 1/3
    assert iou(a, box(CAR, 0.55, 0.5)) == pytest.approx(1 / 3)


def test_best_iou_counts_a_place_as_labelled_even_if_detected_twice():
    human = [box(CAR, 0.5, 0.5)]
    two_models_same_car = [box(CAR, 0.5, 0.5), box(CAR, 0.505, 0.5)]
    assert all(best_iou(p, human) > 0.8 for p in two_models_same_car)  # cả hai dự đoán đều thấy đã có nhãn
    assert best_iou(box(CAR, 0.1, 0.1), human) == 0.0
    assert best_iou(box(CAR, 0.1, 0.1), []) == 0.0


def test_overlap_fraction_small_box_inside_large_region():
    region = YoloBox(4, 0.5, 0.5, 0.4, 0.4)  # vùng ignore lớn
    inside = box(CAR, 0.5, 0.5)  # xe nhỏ nằm trọn trong vùng
    assert overlap_fraction(inside, region) == pytest.approx(1.0)
    assert iou(inside, region) < 0.1  # IoU nhỏ dù xe nằm trọn trong vùng -> phải dùng overlap_fraction
    assert overlap_fraction(box(CAR, 0.9, 0.9), region) == 0.0


def test_find_duplicates_only_same_class_heavy_overlap():
    boxes = [box(MOTORBIKE, 0.5, 0.5), box(MOTORBIKE, 0.501, 0.5), box(CAR, 0.5, 0.5), box(MOTORBIKE, 0.55, 0.5)]
    assert find_duplicates(boxes) == [(0, 1)]  # (0,3) chỉ chồng một phần: hai xe khác nhau che nhau


def test_match_boxes_reports_unmatched_on_both_sides():
    human = [box(MOTORBIKE, 0.2, 0.2), box(CAR, 0.5, 0.5), box(MOTORBIKE, 0.8, 0.8)]
    model = [box(CAR, 0.5, 0.5), box(MOTORBIKE, 0.21, 0.2), box(MOTORBIKE, 0.1, 0.9)]
    result = match_boxes(human, model)
    assert sorted(result.pairs) == [(0, 1), (1, 0)]
    assert result.unmatched_reference == [2]  # người gán có, mô hình không thấy
    assert result.unmatched_candidate == [2]  # mô hình thấy, người gán không có


def test_deduplicate_keeps_most_confident_per_place():
    low = YoloBox(MOTORBIKE, 0.5, 0.5, 0.1, 0.1, confidence=0.6)
    high = YoloBox(MOTORBIKE, 0.505, 0.5, 0.1, 0.1, confidence=0.9)
    elsewhere = YoloBox(CAR, 0.2, 0.2, 0.1, 0.1, confidence=0.7)
    assert deduplicate([low, high, elsewhere]) == [high, elsewhere]


def test_invalid_boxes_tiny_or_outside():
    boxes = [box(CAR, 0.5, 0.5), YoloBox(CAR, 0.5, 0.5, 0.001, 0.1), YoloBox(CAR, 1.2, 0.5, 0.1, 0.1)]
    assert invalid_boxes(boxes, image_width=1920, image_height=1080) == [1, 2]
