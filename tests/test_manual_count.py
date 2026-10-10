import numpy as np
import supervision as sv

from vtrack.count_eval import make_line_zone
from vtrack.manual_count import (
    Crossing,
    count_crossings,
    in_direction,
    match_crossings,
    read_crossings,
    write_crossings,
)


def crossing(time_s: float, class_name: str = "motorbike", direction: str = "out") -> Crossing:
    return Crossing(int(time_s * 30), time_s, class_name, direction)


def test_csv_round_trip_and_counts(tmp_path):
    crossings = [crossing(2.0), crossing(1.0, "car", "in")]
    write_crossings(tmp_path / "c.csv", crossings)
    loaded = read_crossings(tmp_path / "c.csv")
    assert loaded == sorted(crossings, key=lambda c: c.frame)
    counts = count_crossings(loaded)
    assert counts["motorbike"]["out"] == 1 and counts["car"]["in"] == 1 and counts["bus"]["in"] == 0


def test_match_finds_missed_extra_and_wrong_class():
    truth = [crossing(1.0), crossing(5.0, "bus"), crossing(9.0), crossing(3.0, direction="in")]
    predicted = [crossing(1.4), crossing(5.2, "car"), crossing(20.0), crossing(3.1, direction="out")]
    match = match_crossings(truth, predicted, tolerance_s=1.0)
    assert len(match.pairs) == 2
    assert [(t.class_name, p.class_name) for t, p in match.wrong_class] == [("bus", "car")]
    assert sorted(c.time_s for c in match.missed) == [3.0, 9.0]  # lượt 3.0 bị pipeline đếm sai hướng
    assert sorted(c.time_s for c in match.extra) == [3.1, 20.0]


def test_match_is_optimal_not_greedy():
    """Ghép tham lam cặp gần nhất trước (1.7–1.6) thì 1.0 và 2.2 lệch quá xa nhau, mất một cặp; ghép tối ưu giữ cả hai."""
    match = match_crossings([crossing(1.0), crossing(1.7)], [crossing(1.6), crossing(2.2)], tolerance_s=0.65)
    assert len(match.pairs) == 2 and not match.missed and not match.extra


def test_in_direction_agrees_with_line_zone():
    """Xe đi theo vectơ in_direction phải được LineZone đếm là "in" (với cả vạch ngang lẫn vạch chéo)."""
    width, height = 1000, 800
    for line in [(0.0, 0.6, 1.0, 0.6), (0.0, 0.6, 1.0, 0.85), (1.0, 0.6, 0.0, 0.6)]:
        zone = make_line_zone(line, width, height)
        dx, dy = in_direction(line, width, height)
        middle_x, middle_y = (line[0] + line[2]) / 2 * width, (line[1] + line[3]) / 2 * height
        crossed_in = False
        for step in range(-20, 21):
            # BOTTOM_CENTER của box nằm tại điểm trên đường đi
            x, y = middle_x + dx * step * 5, middle_y + dy * step * 5
            detections = sv.Detections(xyxy=np.array([[x - 10, y - 30, x + 10, y]], dtype=float),
                                       tracker_id=np.array([1]), class_id=np.array([0]))
            entered, exited = zone.trigger(detections)
            crossed_in |= bool(entered[0])
            assert not exited[0]
        assert crossed_in, line
