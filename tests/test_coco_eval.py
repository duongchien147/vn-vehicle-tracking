import pytest

from vtrack.coco_eval import EvalImage, evaluate, instance_counts, mean_ap50, to_coco, well_represented_classes
from vtrack.label_review import YoloBox

CAR, MOTORBIKE = 1, 0


def image(truth: list[YoloBox], predictions: list[YoloBox], ignore: list[YoloBox] | None = None) -> EvalImage:
    return EvalImage("a.jpg", 1000, 1000, truth, ignore or [], predictions)


def test_perfect_predictions_score_one():
    truth = [YoloBox(CAR, 0.3, 0.3, 0.1, 0.1), YoloBox(MOTORBIKE, 0.7, 0.7, 0.05, 0.1)]
    predictions = [YoloBox(b.class_id, b.center_x, b.center_y, b.width, b.height, 0.9) for b in truth]
    metrics = evaluate([image(truth, predictions)])
    assert metrics.map50 == pytest.approx(1.0)
    assert metrics.map50_95 == pytest.approx(1.0)
    assert metrics.ap50_per_class["car"] == pytest.approx(1.0)


def test_detection_inside_ignore_region_is_not_penalised():
    truth = [YoloBox(CAR, 0.3, 0.3, 0.1, 0.1)]
    region = YoloBox(4, 0.8, 0.8, 0.3, 0.3)
    detections = [YoloBox(CAR, 0.3, 0.3, 0.1, 0.1, 0.9), YoloBox(CAR, 0.8, 0.8, 0.1, 0.1, 0.95)]
    with_ignore = evaluate([image(truth, detections, ignore=[region])])
    without_ignore = evaluate([image(truth, detections)])
    assert with_ignore.map50 == pytest.approx(1.0)  # phát hiện trong vùng ignore không bị tính sai
    assert without_ignore.map50 < 1.0  # không có vùng ignore thì nó là phát hiện sai (điểm cao hơn cả xe thật)


def test_ignore_region_becomes_crowd_annotation_for_every_class():
    ground_truth, _ = to_coco([image([], [], ignore=[YoloBox(4, 0.5, 0.5, 0.2, 0.2)])])
    crowd = [a for a in ground_truth["annotations"] if a["iscrowd"] == 1]
    assert len(crowd) == 4  # một vùng ignore -> một annotation crowd cho mỗi lớp trong 4 lớp


def test_map50_95_drops_when_boxes_are_loose():
    """Box lệch nhẹ vẫn đúng ở IoU 0.5 nhưng sai ở IoU cao -> mAP50 = 1 còn mAP50-95 < 1 (và không phải -1)."""
    truth = [YoloBox(CAR, 0.3, 0.3, 0.1, 0.1)]
    shifted = [YoloBox(CAR, 0.31, 0.3, 0.1, 0.1, 0.9)]  # IoU ~0.82
    metrics = evaluate([image(truth, shifted)])
    assert metrics.map50 == pytest.approx(1.0)
    assert 0.0 < metrics.map50_95 < 1.0


def test_main_map_uses_only_well_represented_classes():
    many_cars = [YoloBox(CAR, 0.1 * i + 0.05, 0.5, 0.05, 0.05) for i in range(5)]
    one_truck = [YoloBox(3, 0.5, 0.2, 0.1, 0.1)]
    images = [image(many_cars + one_truck, [YoloBox(b.class_id, b.center_x, b.center_y, b.width, b.height, 0.9)
                                            for b in many_cars])]  # bắt đúng mọi ô tô, bỏ sót xe tải duy nhất
    assert instance_counts(images)["car"] == 5
    assert well_represented_classes(images, min_instances=3) == ["car"]
    metrics = evaluate(images)
    assert mean_ap50(metrics, ["car"]) == pytest.approx(1.0)  # mAP chính không bị 1 xe tải kéo xuống
    assert metrics.ap50_per_class["truck"] == pytest.approx(0.0)


def test_no_detections_gives_zero():
    assert evaluate([image([YoloBox(CAR, 0.3, 0.3, 0.1, 0.1)], [])]).map50 == 0.0
