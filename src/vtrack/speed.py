"""Đo tốc độ suy luận trên CPU (thí nghiệm E2): chỉ detector, và cả pipeline detector + tracker + bỏ phiếu lớp.
Bỏ qua vài frame đầu (khởi động) vì lần chạy đầu chậm hơn hẳn do cấp phát bộ nhớ, biên dịch kernel..."""

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from vtrack.models import class_mapping
from vtrack.tracking import track_video

# Số frame chạy "khởi động" trước khi bấm giờ — chuẩn thường dùng khi đo tốc độ mạng nơ-ron
WARMUP_FRAMES = 10


@dataclass
class SpeedResult:
    ms_per_frame: float  # trung vị: ít bị ảnh hưởng bởi vài frame chậm bất thường (tiến trình khác chen vào)
    ms_p90: float  # frame chậm (phân vị 90) — quan trọng khi chạy thời gian thực
    fps: float  # 1000 / trung vị


def _summarize(milliseconds: list[float]) -> SpeedResult:
    median = float(np.median(milliseconds))
    return SpeedResult(ms_per_frame=median, ms_p90=float(np.percentile(milliseconds, 90)), fps=1000 / median)


def measure_detector(model: YOLO, frames: list[np.ndarray], imgsz: int, conf: float) -> SpeedResult:
    """Thời gian mỗi frame của riêng detector (tiền xử lý + suy luận + hậu xử lý của ultralytics)."""
    classes = list(class_mapping(model.names))
    for frame in frames[:WARMUP_FRAMES]:
        model.predict(frame, imgsz=imgsz, conf=conf, classes=classes, verbose=False)
    timings = []
    for frame in frames[WARMUP_FRAMES:]:
        started = time.perf_counter()
        model.predict(frame, imgsz=imgsz, conf=conf, classes=classes, verbose=False)
        timings.append((time.perf_counter() - started) * 1000)
    return _summarize(timings)


def measure_pipeline(model: YOLO, video: str | Path, tracker: str, imgsz: int, conf: float, num_frames: int) -> SpeedResult:
    """Thời gian mỗi frame của cả vòng lặp track_video (đọc frame + detector + tracker + bỏ phiếu lớp),
    tức đúng thứ demo chạy, chưa gồm vẽ và ghi video."""
    timings = []
    started = time.perf_counter()
    for step, _ in enumerate(track_video(video, model, tracker, imgsz, conf, max_frames=num_frames)):
        now = time.perf_counter()
        if step >= WARMUP_FRAMES:
            timings.append((now - started) * 1000)
        started = now
    return _summarize(timings)
