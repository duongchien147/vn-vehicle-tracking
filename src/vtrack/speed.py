"""Đo tốc độ suy luận trên CPU (thí nghiệm E2): chỉ detector, và cả pipeline detector + tracker + bỏ phiếu lớp.

Máy đo là CPU laptop dòng U (i5-8250U): chạy liên tục vài phút là nóng lên và tự hạ xung (thermal throttling), nên
đo liền một mạch thì mô hình đo sau bị thiệt. Cách đo detector vì vậy:
- chia thành nhiều đợt ngắn, nghỉ giữa các đợt cho CPU hạ nhiệt;
- trong mỗi đợt, các mô hình chạy lần lượt trên CÙNG các frame, nghỉ ngắn khi chuyển mô hình, và thứ tự đảo chiều
  sau mỗi đợt (ABBA) để không mô hình nào luôn chạy lúc máy nóng nhất;
- ghi thời gian từng frame (kèm số đợt) để kiểm tra có xu hướng chậm dần không; báo trung vị + IQR.
Khởi động trước khi bấm giờ tính theo THỜI GIAN chứ không chỉ số frame: trên máy này OpenVINO cần ~20 lần chạy
(~5 s) mới ổn định (8 frame đầu 265–782 ms, sau đó ~62 ms), 10 frame cố định là không đủ. Mỗi lượt của một mô hình
trong một đợt còn chạy thêm vài frame khởi động ngắn.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from vtrack.detectors import Detector
from vtrack.tracking import DEFAULT_AGNOSTIC_NMS_IOU, track_video

WARMUP_FRAMES = 10  # khởi động tối thiểu (chuẩn thường dùng khi đo tốc độ mạng nơ-ron)...
WARMUP_SECONDS = 15.0  # ...và kéo dài ít nhất chừng này giây cho mỗi mô hình (xem docstring)
ROUND_WARMUP_FRAMES = 3  # khởi động lại ở đầu mỗi lượt: mô hình khác vừa chạy đã đẩy dữ liệu của nó khỏi cache
ROUNDS = 5
ROUND_FRAMES = 30  # 5 đợt x 30 frame = 150 frame được bấm giờ cho mỗi mô hình
COOLDOWN_S = 10.0  # nghỉ giữa các đợt
# Nghỉ khi chuyển sang mô hình khác trong một đợt: luồng của engine vừa chạy (OpenMP của PyTorch, TBB của OpenVINO)
# còn quay chờ một lúc sau khi xong và tranh CPU với mô hình kế tiếp. Đo thử không có khoảng nghỉ: PyTorch chạy ngay
# sau OpenVINO chậm 120–163 ms/frame, sau 2 s nghỉ chỉ 95 ms.
SWITCH_PAUSE_S = 2.0


@dataclass
class SpeedResult:
    ms_per_frame: float  # trung vị: ít bị ảnh hưởng bởi vài frame chậm bất thường (tiến trình khác chen vào)
    ms_q1: float  # phân vị 25 và 75: IQR = q3 - q1 cho biết độ dao động
    ms_q3: float
    ms_p90: float  # frame chậm (phân vị 90) — quan trọng khi chạy thời gian thực
    fps: float  # 1000 / trung vị


@dataclass(frozen=True)
class FrameTiming:
    model: str
    round_index: int
    frame_index: int
    ms: float


def summarize(milliseconds: Sequence[float]) -> SpeedResult:
    values = np.asarray(milliseconds, dtype=float)
    median = float(np.median(values))
    return SpeedResult(ms_per_frame=median, ms_q1=float(np.percentile(values, 25)), ms_q3=float(np.percentile(values, 75)),
                       ms_p90=float(np.percentile(values, 90)), fps=1000 / median)


def round_order(names: Sequence[str], round_index: int) -> list[str]:
    """Thứ tự chạy trong một đợt: xuôi ở đợt chẵn, ngược ở đợt lẻ (A B C, C B A, A B C...)."""
    return list(names) if round_index % 2 == 0 else list(reversed(names))


def measure_detectors(
    detectors: dict[str, Detector], frames: list[np.ndarray], conf: float,
    rounds: int = ROUNDS, round_frames: int = ROUND_FRAMES, cooldown_s: float = COOLDOWN_S,
) -> list[FrameTiming]:
    """Bấm giờ từng frame của riêng detector (tiền xử lý + suy luận + hậu xử lý) qua giao diện chung `Detector`,
    nên YOLO và RT-DETRv2 được đo theo cùng một cách. `frames` cần ít nhất rounds x round_frames ảnh."""
    if len(frames) < rounds * round_frames:
        raise ValueError(f"Cần {rounds * round_frames} frame, chỉ có {len(frames)}")
    for detector in detectors.values():
        time.sleep(SWITCH_PAUSE_S)
        warmup_started, runs = time.perf_counter(), 0
        while runs < WARMUP_FRAMES or time.perf_counter() - warmup_started < WARMUP_SECONDS:
            detector.predict(frames[runs % len(frames)], conf)
            runs += 1

    timings = []
    for round_index in range(rounds):
        if round_index > 0:
            time.sleep(cooldown_s)
        round_frames_list = frames[round_index * round_frames:(round_index + 1) * round_frames]
        for name in round_order(list(detectors), round_index):
            time.sleep(SWITCH_PAUSE_S)  # cả lượt đầu đợt 0: ngay trước đó là phần khởi động của mô hình khác
            detector = detectors[name]
            for frame in round_frames_list[:ROUND_WARMUP_FRAMES]:
                detector.predict(frame, conf)
            for offset, frame in enumerate(round_frames_list):
                started = time.perf_counter()
                detector.predict(frame, conf)
                timings.append(FrameTiming(name, round_index, round_index * round_frames + offset,
                                           (time.perf_counter() - started) * 1000))
    return timings


def measure_pipeline(
    model: YOLO, video: str | Path, tracker: str, imgsz: int, conf: float, num_frames: int,
    agnostic_nms_iou: float | None = DEFAULT_AGNOSTIC_NMS_IOU,
) -> SpeedResult:
    """Thời gian mỗi frame của cả vòng lặp track_video (đọc frame + detector + bộ lọc box trùng + tracker + bỏ phiếu
    lớp), tức đúng thứ demo chạy, chưa gồm vẽ và ghi video. Tracker cần frame liên tiếp nên đo một mạch, không chia
    đợt. `agnostic_nms_iou` None = tắt bộ lọc box trùng, để đo riêng phần thời gian của bộ lọc."""
    timings = []
    started = time.perf_counter()
    frames = track_video(video, model, tracker, imgsz, conf, max_frames=num_frames, agnostic_nms_iou=agnostic_nms_iou)
    for step, _ in enumerate(frames):
        now = time.perf_counter()
        if step >= WARMUP_FRAMES:
            timings.append((now - started) * 1000)
        started = now
    return summarize(timings)
