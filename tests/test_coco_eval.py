import pytest

from vtrack import CLASS_NAMES
from vtrack.coco_eval import (
    EvalImage,
    average_precision,
    bootstrap_map50,
    evaluate,
    instance_counts,
    iou50_matches,
    mean_ap50,
    time_blocks,
    to_coco,
    well_represented_classes,
)
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


def test_map75_counts_only_tight_boxes():
    """Box dịch 20% chiều rộng: IoU = 0.8/1.2 ≈ 0.67 -> đúng ở IoU 0.5, sai ở IoU 0.75."""
    truth = [YoloBox(CAR, 0.3, 0.3, 0.1, 0.1)]
    shifted = [YoloBox(CAR, 0.32, 0.3, 0.1, 0.1, 0.9)]
    metrics = evaluate([image(truth, shifted)])
    assert metrics.map50 == pytest.approx(1.0)
    assert metrics.map75 == pytest.approx(0.0)


def mixed_images() -> list[EvalImage]:
    """Vài ảnh có đủ trường hợp: đúng, sai lớp, bỏ sót, thừa, phát hiện trong vùng ignore, điểm tin cậy bằng nhau."""
    return [
        image([YoloBox(CAR, 0.3, 0.3, 0.1, 0.1), YoloBox(MOTORBIKE, 0.6, 0.6, 0.05, 0.1)],
              [YoloBox(CAR, 0.3, 0.3, 0.1, 0.1, 0.9), YoloBox(CAR, 0.6, 0.6, 0.05, 0.1, 0.8),
               YoloBox(MOTORBIKE, 0.9, 0.9, 0.05, 0.05, 0.4)]),
        image([YoloBox(MOTORBIKE, 0.2, 0.2, 0.05, 0.1), YoloBox(MOTORBIKE, 0.4, 0.2, 0.05, 0.1)],
              [YoloBox(MOTORBIKE, 0.2, 0.2, 0.05, 0.1, 0.7), YoloBox(MOTORBIKE, 0.8, 0.8, 0.05, 0.05, 0.7),
               YoloBox(CAR, 0.5, 0.8, 0.1, 0.1, 0.95)],
              ignore=[YoloBox(4, 0.8, 0.8, 0.2, 0.2)]),
        image([YoloBox(CAR, 0.5, 0.5, 0.2, 0.2)], [YoloBox(CAR, 0.52, 0.5, 0.2, 0.2, 0.6)]),
    ]


def test_average_precision_from_matches_equals_cocoeval():
    """AP tính lại từ kết quả ghép (dùng cho bootstrap nhanh) phải trùng AP50 của COCOeval trên cùng tập ảnh."""
    images = mixed_images()
    metrics = evaluate(images)
    matches = iou50_matches(images)
    for index, name in enumerate(CLASS_NAMES[:2]):
        assert average_precision(matches[index]) == pytest.approx(metrics.ap50_per_class[name])


def test_bootstrap_interval_is_reproducible_and_contains_estimate():
    images = mixed_images() * 4
    classes = ["motorbike", "car"]
    low, high = bootstrap_map50(images, rounds=200, classes=classes, seed=1)
    assert (low, high) == bootstrap_map50(images, rounds=200, classes=classes, seed=1)
    assert low <= mean_ap50(evaluate(images), classes) <= high


def test_time_blocks_start_at_first_frame_of_each_clip():
    clips = ["a", "a", "a", "a", "b", "b"]
    seconds = [35, 37, 45, 56, 2, 14]
    assert time_blocks(clips, seconds, block_seconds=10) == ["a#0", "a#0", "a#1", "a#2", "b#0", "b#1"]


def test_singleton_groups_give_same_interval_as_per_image_bootstrap():
    images = mixed_images() * 4
    classes = ["motorbike", "car"]
    per_image = bootstrap_map50(images, rounds=200, classes=classes, seed=1)
    singletons = [str(index) for index in range(len(images))]
    assert bootstrap_map50(images, rounds=200, classes=classes, seed=1, groups=singletons) == per_image


def test_block_bootstrap_needs_enough_blocks():
    images = mixed_images() * 4
    low, high = bootstrap_map50(images, rounds=50, classes=["car"], groups=["x"] * 6 + ["y"] * 6)
    assert low != low and high != high  # nan: 2 khối không đủ để ước lượng phân vị


def test_correlated_blocks_widen_the_interval():
    """Ảnh trong cùng khối giống hệt nhau (như frame liền nhau của một clip): lấy mẫu theo ảnh tưởng có 12 ảnh độc lập,
    lấy mẫu theo khối chỉ có 6 đơn vị -> khoảng tin cậy phải rộng hơn."""
    good = image([YoloBox(CAR, 0.3, 0.3, 0.1, 0.1)], [YoloBox(CAR, 0.3, 0.3, 0.1, 0.1, 0.9)])
    bad = image([YoloBox(CAR, 0.3, 0.3, 0.1, 0.1)], [])
    images = [good, good, bad, bad, good, good, bad, bad, good, good, good, good]
    groups = [str(index // 2) for index in range(len(images))]
    per_image = bootstrap_map50(images, rounds=1000, classes=["car"])
    per_block = bootstrap_map50(images, rounds=1000, classes=["car"], groups=groups)
    assert per_block[1] - per_block[0] > per_image[1] - per_image[0]
