"""Thống kê đầu ra tracking KHÔNG cần nhãn chuẩn — dùng cho cảnh chưa có nhãn (vd. ban đêm) để so tương đối với cảnh
đã biết kết quả thật. Đây là chỉ số thay thế, không phải độ chính xác: nhiều track ngắn gợi ý track bị đứt/đổi ID,
nhưng cũng có thể là xe thật đi qua nhanh."""

from collections import Counter
from dataclasses import dataclass

import numpy as np

from vtrack.mot_format import MotBox


@dataclass
class TrackStats:
    frames: int
    boxes_per_frame: float
    median_confidence: float
    tracks: int
    median_track_frames: float
    short_track_fraction: float  # tỉ lệ track sống ít hơn `short_frames` frame


def track_statistics(boxes: list[MotBox], frames: int, short_frames: int) -> TrackStats:
    lengths = np.array(list(Counter(box.track_id for box in boxes).values()))
    confidences = np.array([box.confidence for box in boxes])
    return TrackStats(
        frames=frames,
        boxes_per_frame=len(boxes) / frames if frames else 0.0,
        median_confidence=float(np.median(confidences)) if len(confidences) else 0.0,
        tracks=len(lengths),
        median_track_frames=float(np.median(lengths)) if len(lengths) else 0.0,
        short_track_fraction=float((lengths < short_frames).mean()) if len(lengths) else 0.0,
    )
