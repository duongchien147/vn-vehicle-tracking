import pytest

from vtrack.label_review import YoloBox
from vtrack.pseudo_label import (
    TileWindow,
    flip_horizontal,
    fuse,
    select_pseudo_labels,
    tile_box_to_image,
    tile_windows,
    touches_inner_edge,
)

CAR, MOTORBIKE = 1, 0


def test_flip_horizontal_mirrors_center_and_is_reversible():
    box = YoloBox(CAR, 0.2, 0.3, 0.1, 0.1, 0.9)
    flipped = flip_horizontal([box])[0]
    assert flipped.center_x == pytest.approx(0.8)
    assert (flipped.center_y, flipped.width, flipped.height, flipped.confidence) == (0.3, 0.1, 0.1, 0.9)
    assert flip_horizontal([flipped])[0].center_x == pytest.approx(0.2)


def test_fuse_merges_agreeing_boxes_and_averages_coordinates():
    a = [YoloBox(CAR, 0.50, 0.5, 0.2, 0.2, 0.8)]
    b = [YoloBox(CAR, 0.52, 0.5, 0.2, 0.2, 0.8)]
    fused = fuse([a, b], weights=[1, 1], iou_thr=0.55, skip_box_thr=0.0)
    assert len(fused) == 1
    assert fused[0].center_x == pytest.approx(0.51)
    assert fused[0].confidence == pytest.approx(0.8)


def test_fuse_downweights_box_seen_by_one_source_only():
    seen_by_both = YoloBox(CAR, 0.3, 0.3, 0.1, 0.1, 0.8)
    seen_once = YoloBox(MOTORBIKE, 0.8, 0.8, 0.05, 0.05, 0.8)
    fused = fuse([[seen_by_both, seen_once], [seen_by_both]], weights=[1, 1], iou_thr=0.55, skip_box_thr=0.0)
    scores = {box.class_id: box.confidence for box in fused}
    assert scores[CAR] == pytest.approx(0.8)
    assert scores[MOTORBIKE] == pytest.approx(0.4)  # 0.8 * 1 nguồn / 2 nguồn


def test_select_pseudo_labels_thresholds_then_removes_cross_class_duplicates():
    motorbike = YoloBox(MOTORBIKE, 0.5, 0.5, 0.1, 0.1, 0.6)
    same_place_as_car = YoloBox(CAR, 0.505, 0.5, 0.1, 0.1, 0.3)  # mô hình khác gọi cùng xe là ô tô, điểm thấp hơn
    weak = YoloBox(CAR, 0.1, 0.1, 0.1, 0.1, 0.1)
    assert select_pseudo_labels([motorbike, same_place_as_car, weak], keep_conf=0.15, dedup_iou=0.6) == [motorbike]


def test_tile_windows_cover_image_with_overlap():
    windows = tile_windows(1920, 1080, tiles=2, overlap=0.2)
    assert len(windows) == 4
    assert min(w.left for w in windows) == 0 and max(w.right for w in windows) == 1920
    assert min(w.top for w in windows) == 0 and max(w.bottom for w in windows) == 1080
    left_tile, right_tile = windows[0], windows[1]
    assert right_tile.left < left_tile.right  # hai ô cạnh nhau chồng lấn


def test_tile_box_maps_back_to_image_coordinates():
    window = TileWindow(left=960, top=540, right=1920, bottom=1080)  # ô góc dưới phải
    centre_of_tile = YoloBox(CAR, 0.5, 0.5, 0.1, 0.2, 0.9)
    mapped = tile_box_to_image(centre_of_tile, window, width=1920, height=1080)
    assert (mapped.center_x, mapped.center_y) == (pytest.approx(0.75), pytest.approx(0.75))
    assert (mapped.width, mapped.height) == (pytest.approx(0.05), pytest.approx(0.1))


def test_touches_inner_edge_only_on_sides_with_neighbouring_tile():
    window = TileWindow(left=0, top=0, right=1100, bottom=620)  # ô góc trên trái: mép phải và mép dưới là mép cắt
    at_image_border = YoloBox(CAR, 10 / 1920, 300 / 1080, 20 / 1920, 40 / 1080)  # chạm mép trái ảnh -> giữ
    at_cut = YoloBox(CAR, 1090 / 1920, 300 / 1080, 20 / 1920, 40 / 1080)  # chạm mép cắt bên phải -> bỏ
    assert not touches_inner_edge(at_image_border, window, 1920, 1080)
    assert touches_inner_edge(at_cut, window, 1920, 1080)


def test_fuse_handles_no_predictions():
    assert fuse([[], []], weights=[1, 1], iou_thr=0.55, skip_box_thr=0.0) == []
