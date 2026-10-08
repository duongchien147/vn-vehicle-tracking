from vtrack.count_eval import compare_counts, count_ground_truth, empty_counts, track_classes
from vtrack.mot_format import MotBox

LINE = (0.0, 0.5, 1.0, 0.5)  # vạch ngang giữa ảnh 100x100
SIZE = 100


def track(track_id: int, label: str, tops: list[float]) -> list[tuple[str, MotBox]]:
    """Một xe 10x10 px đi theo chiều dọc; đáy box = top + 10."""
    return [(label, MotBox(frame=i + 1, track_id=track_id, left=45, top=top, width=10, height=10))
            for i, top in enumerate(tops)]


def test_track_crossing_line_is_counted_once_with_its_class():
    labeled = track(1, "car", [20, 30, 40, 50, 60])  # đáy đi từ 30 xuống 70: cắt vạch y=50 một lần
    counts = count_ground_truth(labeled, LINE, SIZE, SIZE)
    assert counts["car"]["in"] + counts["car"]["out"] == 1
    assert sum(sum(c.values()) for name, c in counts.items() if name != "car") == 0


def test_track_not_reaching_line_and_ignore_regions_are_not_counted():
    stays_above = track(1, "motorbike", [10, 15, 20, 25])  # đáy 20 -> 35, chưa tới vạch
    ignore = track(2, "ignore", [20, 30, 40, 50, 60])  # vùng ignore cắt vạch nhưng không phải xe
    counts = count_ground_truth(stays_above + ignore, LINE, SIZE, SIZE)
    assert all(v == 0 for c in counts.values() for v in c.values())


def test_track_class_is_majority_label():
    labeled = [("car", MotBox(1, 7, 0, 0, 1, 1)), ("truck", MotBox(2, 7, 0, 0, 1, 1)), ("car", MotBox(3, 7, 0, 0, 1, 1))]
    assert track_classes(labeled) == {7: "car"}


def test_compare_counts_errors_per_class_and_total():
    truth, predicted = empty_counts(), empty_counts()
    truth["car"]["in"], truth["truck"]["in"] = 4, 1
    predicted["car"]["in"], predicted["truck"]["in"] = 5, 0  # xe tải bị gọi là ô tô
    rows = {r.class_name: r for r in compare_counts(truth, predicted)}
    assert rows["car"].absolute_error == 1 and rows["truck"].absolute_error == 1
    assert rows["all"].absolute_error == 0  # tổng số xe vẫn đúng, chỉ sai lớp
    assert rows["car"].relative_error == 0.25
