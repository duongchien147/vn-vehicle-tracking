"""Chấm sai số đếm xe qua vạch. Số xe thật lấy bằng cách cho các track nhãn tay (CVAT) đi qua ĐÚNG bộ đếm của pipeline
(cùng LineZone, cùng điểm neo BOTTOM_CENTER), nên chênh lệch với số pipeline đếm được chỉ phản ánh lỗi phát hiện và
tracking, không lẫn khác biệt về định nghĩa "cắt qua vạch"."""

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import supervision as sv

from vtrack import CLASS_NAMES
from vtrack.mot_format import MotBox

DIRECTIONS = ("in", "out")


def make_line_zone(line: tuple[float, float, float, float], width: int, height: int) -> sv.LineZone:
    """Vạch đếm của pipeline. `line` = (x1, y1, x2, y2) tương đối 0–1. Neo ở đáy giữa box (BOTTOM_CENTER): điểm xe
    chạm mặt đường, ít bị lệch do chiều cao xe khi nhìn chéo từ trên cao."""
    return sv.LineZone(
        start=sv.Point(line[0] * width, line[1] * height),
        end=sv.Point(line[2] * width, line[3] * height),
        triggering_anchors=[sv.Position.BOTTOM_CENTER],
    )


def empty_counts() -> dict[str, dict[str, int]]:
    return {name: {direction: 0 for direction in DIRECTIONS} for name in CLASS_NAMES}


def track_classes(labeled: list[tuple[str, MotBox]]) -> dict[int, str]:
    """Lớp của mỗi track nhãn = nhãn xuất hiện nhiều nhất trên các box của nó (CVAT gán một nhãn cho cả track)."""
    votes: defaultdict[int, Counter[str]] = defaultdict(Counter)
    for label, box in labeled:
        votes[box.track_id][label] += 1
    return {track_id: counter.most_common(1)[0][0] for track_id, counter in votes.items()}


def count_ground_truth(
    labeled: list[tuple[str, MotBox]], line: tuple[float, float, float, float], width: int, height: int,
) -> dict[str, dict[str, int]]:
    """Số xe thật cắt qua vạch theo lớp và hướng. Box nhãn không thuộc 4 lớp xe (vùng ignore) bị bỏ."""
    classes = {tid: name for tid, name in track_classes(labeled).items() if name in CLASS_NAMES}
    by_frame: defaultdict[int, list[MotBox]] = defaultdict(list)
    for _, box in labeled:
        if box.track_id in classes:
            by_frame[box.frame].append(box)

    line_zone = make_line_zone(line, width, height)
    counts = empty_counts()
    for frame in sorted(by_frame):
        boxes = by_frame[frame]
        detections = sv.Detections(
            xyxy=np.array([[b.left, b.top, b.left + b.width, b.top + b.height] for b in boxes], dtype=float),
            tracker_id=np.array([b.track_id for b in boxes]),
            class_id=np.array([CLASS_NAMES.index(classes[b.track_id]) for b in boxes]),
        )
        crossed_in, crossed_out = line_zone.trigger(detections)
        for direction, crossed in zip(DIRECTIONS, (crossed_in, crossed_out)):
            for i in np.flatnonzero(crossed):
                counts[classes[boxes[i].track_id]][direction] += 1
    return counts


@dataclass(frozen=True)
class CountComparison:
    class_name: str  # một lớp xe, hoặc "all" = tổng mọi lớp
    truth_in: int
    truth_out: int
    predicted_in: int
    predicted_out: int

    @property
    def truth_total(self) -> int:
        return self.truth_in + self.truth_out

    @property
    def predicted_total(self) -> int:
        return self.predicted_in + self.predicted_out

    @property
    def absolute_error(self) -> int:
        """Sai số theo từng hướng cộng lại: đếm thừa chiều này không bù được cho đếm thiếu chiều kia."""
        return abs(self.predicted_in - self.truth_in) + abs(self.predicted_out - self.truth_out)

    @property
    def relative_error(self) -> float:
        return self.absolute_error / self.truth_total if self.truth_total else float("nan")


def compare_counts(
    truth: dict[str, dict[str, int]], predicted: dict[str, dict[str, int]],
) -> list[CountComparison]:
    """Mỗi lớp một dòng, cộng dòng "all" ở cuối."""
    rows = [CountComparison(name, truth[name]["in"], truth[name]["out"], predicted[name]["in"], predicted[name]["out"])
            for name in CLASS_NAMES]
    rows.append(CountComparison(
        "all", sum(r.truth_in for r in rows), sum(r.truth_out for r in rows),
        sum(r.predicted_in for r in rows), sum(r.predicted_out for r in rows),
    ))
    return rows
