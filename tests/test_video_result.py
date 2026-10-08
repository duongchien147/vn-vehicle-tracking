import csv
from pathlib import Path

from vtrack.pipeline import CrossingEvent, VideoResult


def make_result(events: list[CrossingEvent]) -> VideoResult:
    return VideoResult(output_path=Path("x.mp4"), events=events, frames=60, processing_fps=5.0, imgsz=960)


def test_cumulative_counts_sum_both_directions_up_to_each_time_step():
    result = make_result([
        CrossingEvent(frame=10, time_s=0.4, track_id=1, display_id=1, class_name="motorbike", direction="in"),
        CrossingEvent(frame=40, time_s=1.5, track_id=2, display_id=2, class_name="motorbike", direction="out"),
        CrossingEvent(frame=45, time_s=1.8, track_id=3, display_id=3, class_name="car", direction="in"),
    ])

    rows = result.cumulative_counts(duration_s=2.0)

    assert [r["time_s"] for r in rows] == [0.0, 1.0, 2.0]
    assert [r["motorbike"] for r in rows] == [0, 1, 2]
    assert [r["car"] for r in rows] == [0, 0, 1]


def test_write_events_csv_has_one_row_per_event(tmp_path):
    result = make_result([CrossingEvent(frame=3, time_s=0.1, track_id=7, display_id=7, class_name="bus", direction="in")])

    path = result.write_events_csv(tmp_path / "events.csv")

    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows == [{"frame": "3", "time_s": "0.1", "track_id": "7", "display_id": "7", "class_name": "bus", "direction": "in"}]
