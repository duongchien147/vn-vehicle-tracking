import pytest

from vtrack.mot_format import (
    MOT_DISTRACTOR_CLASS,
    MotBox,
    drop_boxes_in_regions,
    fill_track_gaps,
    read_cvat_mot,
    read_mot_file,
    resolve_duplicate_ids,
    write_mot_ground_truth,
    write_mot_results,
)


def test_read_converts_zero_based_frames_to_one_based(tmp_path):
    path = tmp_path / "gt.txt"
    path.write_text("0,7,10,20,30,40,1.0,-1,-1,-1\n5,7,11,21,30,40,1.0,-1,-1,-1\n", encoding="utf-8")

    boxes = read_mot_file(path, first_frame=0)

    assert [b.frame for b in boxes] == [1, 6]
    assert boxes[0] == MotBox(frame=1, track_id=7, left=10, top=20, width=30, height=40)


def test_ground_truth_round_trip_keeps_boxes(tmp_path):
    boxes = [MotBox(2, 1, 1.5, 2.5, 10, 20), MotBox(1, 3, 0, 0, 5, 5, mot_class=MOT_DISTRACTOR_CLASS)]
    path = tmp_path / "gt" / "gt.txt"

    write_mot_ground_truth(path, boxes)
    lines = path.read_text(encoding="utf-8").splitlines()

    assert lines == ["1,3,0.00,0.00,5.00,5.00,1,8,1", "2,1,1.50,2.50,10.00,20.00,1,1,1"]


def test_reading_written_ground_truth_keeps_distractor_class(tmp_path):
    boxes = [MotBox(1, 1, 0, 0, 5, 5), MotBox(2, 1, 0, 0, 5, 5, mot_class=MOT_DISTRACTOR_CLASS)]
    path = tmp_path / "gt.txt"
    write_mot_ground_truth(path, boxes)

    assert read_mot_file(path) == boxes


def test_results_have_ten_columns_with_confidence(tmp_path):
    path = tmp_path / "tracker.txt"
    write_mot_results(path, [MotBox(1, 4, 1, 2, 3, 4, confidence=0.5)])

    assert path.read_text(encoding="utf-8").strip() == "1,4,1.00,2.00,3.00,4.00,0.5000,-1,-1,-1"


def test_fill_track_gaps_interpolates_missing_frames_as_distractors():
    track = [MotBox(1, 1, 0, 0, 10, 10), MotBox(4, 1, 30, 60, 10, 40)]

    filled = fill_track_gaps(track)

    assert [b.frame for b in filled] == [2, 3]
    assert filled[0].left == pytest.approx(10) and filled[0].top == pytest.approx(20)
    assert filled[1].height == pytest.approx(30)
    assert all(b.mot_class == MOT_DISTRACTOR_CLASS and b.track_id == 1 for b in filled)


def test_fill_track_gaps_ignores_continuous_tracks_and_track_ends():
    continuous = [MotBox(f, 1, 0, 0, 10, 10) for f in (1, 2, 3)]
    other_track = [MotBox(5, 2, 0, 0, 10, 10)]

    assert fill_track_gaps(continuous + other_track) == []


def test_resolve_duplicate_ids_keeps_box_closest_to_track_and_neutralises_the_other():
    previous = MotBox(1, 5, 100, 100, 50, 50)
    good = MotBox(2, 5, 102, 101, 50, 50)
    stray = MotBox(2, 5, 300, 300, 40, 40)

    resolved = resolve_duplicate_ids([previous, stray, good])

    kept = [b for b in resolved if b.frame == 2 and b.track_id == 5]
    assert kept == [good]
    neutral = [b for b in resolved if b.mot_class == MOT_DISTRACTOR_CLASS]
    assert len(neutral) == 1 and neutral[0].left == 300 and neutral[0].track_id == 6


def test_read_cvat_mot_maps_labels_and_separates_ignore_regions(tmp_path):
    path = tmp_path / "gt.txt"
    path.write_text("1,1,10,10,20,20,1,2,1.0\n1,2,0,0,500,100,1,5,1.0\n", encoding="utf-8")

    vehicles, regions = read_cvat_mot(path, ["motorbike", "car", "bus", "truck", "ignore"], "ignore")

    assert vehicles == [MotBox(1, 1, 10, 10, 20, 20)]
    assert regions == [MotBox(1, 2, 0, 0, 500, 100)]


def test_drop_boxes_in_regions_uses_fraction_inside_same_frame():
    region = MotBox(1, 99, 0, 0, 100, 100)
    inside = MotBox(1, 1, 10, 10, 20, 20)
    half_inside = MotBox(1, 2, 90, 10, 20, 20)  # chỉ 50% nằm trong vùng < ngưỡng 0.6 -> giữ
    other_frame = MotBox(2, 3, 10, 10, 20, 20)

    kept = drop_boxes_in_regions([inside, half_inside, other_frame], [region], min_overlap=0.6)

    assert kept == [half_inside, other_frame]
