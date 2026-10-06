import json

from vtrack import CLASS_NAMES
from vtrack.coco_source import load_coco, to_project_classes
from vtrack.label_completion import complete_labels
from vtrack.label_review import YoloBox

CAR = CLASS_NAMES.index("car")


def test_adds_only_confident_predictions_away_from_existing_boxes():
    labeled_car = YoloBox(CAR, 0.2, 0.2, 0.1, 0.1)
    rickshaw = YoloBox(-1, 0.6, 0.6, 0.1, 0.1)  # lớp không dùng, vẫn chiếm chỗ
    predictions = [
        YoloBox(CAR, 0.2, 0.2, 0.1, 0.1, confidence=0.9),   # trùng xe đã có nhãn
        YoloBox(CAR, 0.6, 0.6, 0.1, 0.1, confidence=0.9),   # là xe ba bánh, mô hình gọi nhầm ô tô
        YoloBox(CAR, 0.8, 0.2, 0.1, 0.1, confidence=0.3),   # không đủ chắc
        YoloBox(CAR, 0.8, 0.8, 0.1, 0.1, confidence=0.7),   # xe thiếu nhãn -> thêm
        YoloBox(CAR, 0.805, 0.8, 0.1, 0.1, confidence=0.6), # box thứ hai cho cùng xe -> bỏ
    ]

    result = complete_labels([labeled_car], [labeled_car, rickshaw], predictions)

    assert result.added == [predictions[3]]
    assert result.labels == [labeled_car, predictions[3]]


def test_prediction_inside_a_larger_labeled_box_is_not_added():
    motorbike_with_rider = YoloBox(0, 0.5, 0.5, 0.1, 0.3)  # nhãn bao cả người lái
    vehicle_only = YoloBox(0, 0.5, 0.6, 0.08, 0.08, confidence=0.9)  # thầy chỉ khoanh phần xe, IoU ~0.2

    result = complete_labels([motorbike_with_rider], [motorbike_with_rider], [vehicle_only])

    assert result.added == []


def test_load_coco_maps_classes_and_drops_unused(tmp_path):
    path = tmp_path / "ann.json"
    path.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "a.png", "width": 200, "height": 100}],
        "categories": [{"id": 1, "name": "Sedan"}, {"id": 2, "name": "Three-wheeler"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [20, 10, 40, 20]},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [100, 50, 20, 20]},
        ],
    }), encoding="utf-8")

    (image,) = load_coco(path)
    boxes = to_project_classes(image, {"Sedan": "car"})

    assert len(image.boxes) == 2  # vẫn giữ box lớp không dùng để chặn nhãn bổ sung
    assert boxes == [YoloBox(CAR, 0.2, 0.2, 0.2, 0.2)]


def test_to_eval_image_splits_truth_and_ignore_regions():
    from vtrack.coco_source import CocoImage, to_eval_image

    car = YoloBox(-1, 0.2, 0.2, 0.1, 0.1)
    bicycle = YoloBox(-1, 0.5, 0.5, 0.1, 0.1)
    image = CocoImage("a.jpg", 100, 100, [("car", car), ("bicycle", bicycle), ("tram", car)])

    result = to_eval_image(image, {"car": "car"}, {"bicycle"})

    assert result.truth == [YoloBox(CAR, 0.2, 0.2, 0.1, 0.1)]
    assert result.ignore == [bicycle]
