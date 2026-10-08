from vtrack.error_analysis import classify_detections, find_id_switches
from vtrack.label_review import YoloBox
from vtrack.mot_format import MotBox

CAR, BUS, MOTORBIKE = 1, 2, 0


def test_classify_detections_separates_correct_wrong_class_missed_and_extra():
    truth = [YoloBox(CAR, 0.2, 0.2, 0.1, 0.1), YoloBox(BUS, 0.5, 0.5, 0.2, 0.2), YoloBox(MOTORBIKE, 0.8, 0.8, 0.05, 0.1)]
    predictions = [
        YoloBox(CAR, 0.2, 0.2, 0.1, 0.1, 0.9),  # đúng
        YoloBox(CAR, 0.5, 0.5, 0.2, 0.2, 0.8),  # đúng chỗ, sai lớp (buýt -> ô tô)
        YoloBox(CAR, 0.1, 0.9, 0.05, 0.05, 0.7),  # không có xe ở đó
    ]
    outcome = classify_detections(truth, predictions)
    assert len(outcome.correct) == 1 and outcome.correct[0][0].class_id == CAR
    assert len(outcome.wrong_class) == 1 and outcome.wrong_class[0][0].class_id == BUS
    assert [b.class_id for b in outcome.missed] == [MOTORBIKE]
    assert len(outcome.false_positives) == 1


def test_find_id_switches_reports_change_of_matched_track():
    truth = [MotBox(frame=f, track_id=1, left=10 * f, top=0, width=50, height=50) for f in range(1, 5)]
    predicted = [MotBox(frame=f, track_id=7 if f <= 2 else 9, left=10 * f, top=0, width=50, height=50) for f in range(1, 5)]
    switches = find_id_switches(truth, predicted)
    assert len(switches) == 1
    assert (switches[0].frame, switches[0].old_id, switches[0].new_id) == (3, 7, 9)


def test_no_switch_when_track_keeps_its_id_through_a_gap():
    truth = [MotBox(frame=f, track_id=1, left=0, top=0, width=50, height=50) for f in (1, 2, 5)]
    predicted = [MotBox(frame=f, track_id=4, left=0, top=0, width=50, height=50) for f in (1, 2, 5)]
    assert find_id_switches(truth, predicted) == []
