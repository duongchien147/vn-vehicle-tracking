"""Số lượt xe qua vạch do NGƯỜI đếm tay (xem video, bấm phím lúc đáy xe chạm vạch) làm nhãn chuẩn cho sai số đếm
trên các clip chưa có nhãn tracking (vd. clip đêm). Đếm tay theo lớp và hướng là cách AI City Challenge (bài đếm xe)
tạo nhãn chuẩn; người đếm chỉ xem video gốc, không thấy kết quả của mô hình.

Ngoài so tổng số theo lớp/hướng (vtrack.count_eval), còn ghép từng lượt người đếm với lượt pipeline đếm theo thời
điểm (ghép tối ưu, lệch <= tolerance) để biết lượt nào bị sót, đếm thừa hay sai lớp.
"""

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from vtrack.count_eval import DIRECTIONS, empty_counts

# Người bấm phím chậm hơn lúc xe chạm vạch một chút, và đáy box của tracker không trùng hẳn chỗ bánh xe -> cho lệch 1 s
MATCH_TOLERANCE_S = 1.0
CSV_FIELDS = ["frame", "time_s", "class_name", "direction"]


@dataclass(frozen=True)
class Crossing:
    """Một lượt xe qua vạch (người đếm hoặc pipeline)."""

    frame: int
    time_s: float
    class_name: str
    direction: str  # "in" | "out", cùng quy ước với supervision.LineZone


def read_crossings(path: str | Path) -> list[Crossing]:
    """Đọc CSV có các cột frame, time_s, class_name, direction (file đếm tay, hoặc _events.csv của pipeline)."""
    with Path(path).open(newline="", encoding="utf-8") as f:
        return [Crossing(int(row["frame"]), float(row["time_s"]), row["class_name"], row["direction"])
                for row in csv.DictReader(f)]


def write_crossings(path: str | Path, crossings: Sequence[Crossing]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for c in sorted(crossings, key=lambda c: c.frame):
            writer.writerow({"frame": c.frame, "time_s": round(c.time_s, 3), "class_name": c.class_name,
                             "direction": c.direction})


def count_crossings(crossings: Sequence[Crossing]) -> dict[str, dict[str, int]]:
    """Số lượt theo lớp và hướng, cùng dạng với VideoResult.counts() để dùng compare_counts."""
    counts = empty_counts()
    for c in crossings:
        counts[c.class_name][c.direction] += 1
    return counts


def in_direction(line: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float]:
    """Vectơ đơn vị (pixel) chỉ chiều mà xe đi theo thì LineZone đếm là "in": pháp tuyến (dy, -dx) của vạch
    start -> end. Có test đối chiếu với supervision.LineZone để không lệch quy ước."""
    dx, dy = (line[2] - line[0]) * width, (line[3] - line[1]) * height
    length = float(np.hypot(dx, dy))
    return dy / length, -dx / length


@dataclass
class CrossingMatch:
    pairs: list[tuple[Crossing, Crossing]]  # (người đếm, pipeline) cùng hướng, lệch <= tolerance
    missed: list[Crossing]  # người đếm có, pipeline không
    extra: list[Crossing]  # pipeline đếm, người không có

    @property
    def wrong_class(self) -> list[tuple[Crossing, Crossing]]:
        return [(truth, predicted) for truth, predicted in self.pairs if truth.class_name != predicted.class_name]


def match_crossings(
    truth: Sequence[Crossing], predicted: Sequence[Crossing], tolerance_s: float = MATCH_TOLERANCE_S,
) -> CrossingMatch:
    """Ghép một-một từng hướng bằng thuật toán Hungary (như ghép box khi chấm MOT), chỉ ghép hai lượt lệch nhau không
    quá tolerance_s, theo thứ tự ưu tiên: (1) ghép được nhiều cặp nhất; (2) cặp cùng lớp; (3) lệch thời gian ít nhất.

    Nếu chỉ cực tiểu tổng độ lệch thời gian, một độ trễ đều của người bấm (vd. 0.5 s) làm phép ghép trượt sang xe kế bên
    trong dòng xe dày, sinh ra lỗi sai lớp và sót/thừa giả ở hai đầu chuỗi (đã gặp khi tự kiểm công cụ)."""
    match = CrossingMatch([], [], [])
    for direction in DIRECTIONS:
        truth_d = [c for c in truth if c.direction == direction]
        predicted_d = [c for c in predicted if c.direction == direction]
        paired_truth: set[int] = set()
        paired_predicted: set[int] = set()
        if truth_d and predicted_d:
            gaps = np.abs(np.array([[t.time_s - p.time_s for p in predicted_d] for t in truth_d]))
            other_class = np.array([[t.class_name != p.class_name for p in predicted_d] for t in truth_d])
            valid = gaps <= tolerance_s
            # Thứ tự ưu tiên chặt: tổng độ lệch thời gian <= n * tolerance_s < phạt một cặp khác lớp; tổng mọi chi phí
            # <= n * (tolerance_s + phạt khác lớp) < thưởng một cặp hợp lệ (n = số cặp tối đa)
            most_pairs = min(len(truth_d), len(predicted_d))
            class_penalty = tolerance_s * (most_pairs + 1)
            pair_reward = (tolerance_s + class_penalty) * (most_pairs + 1)
            cost = np.where(valid, gaps + class_penalty * other_class - pair_reward, 0.0)
            rows, cols = linear_sum_assignment(cost)
            for row, col in zip(rows, cols):
                if valid[row, col]:
                    match.pairs.append((truth_d[row], predicted_d[col]))
                    paired_truth.add(int(row))
                    paired_predicted.add(int(col))
        match.missed += [c for i, c in enumerate(truth_d) if i not in paired_truth]
        match.extra += [c for i, c in enumerate(predicted_d) if i not in paired_predicted]
    return match

