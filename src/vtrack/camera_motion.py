"""Đo chuyển động của camera trong video (trôi, zoom, cắt cảnh) để tìm các đoạn camera đứng yên.

Đếm xe bằng vạch cố định trên khung hình chỉ đúng khi camera đứng yên, nên video đánh giá
phải được cắt từ các đoạn ổn định do module này tìm ra.
"""

from dataclasses import dataclass

import cv2
import numpy as np

# Chuyển động được đo trên ảnh thu về chiều rộng này để kết quả so sánh được giữa các độ phân giải
ANALYSIS_WIDTH = 640


@dataclass
class MotionSample:
    time_s: float
    shift_px: float  # dịch chuyển toàn cục so với mẫu trước, tính trên ảnh rộng ANALYSIS_WIDTH
    zoom_pct: float  # thay đổi tỉ lệ so với mẫu trước (%)
    scene_cut: bool  # hai mẫu liên tiếp khác hẳn nhau (cắt cảnh / đổi góc máy)


def _global_motion(prev_gray: np.ndarray, gray: np.ndarray) -> tuple[float, float] | None:
    """Ước lượng dịch chuyển + zoom của cả khung hình từ các điểm đặc trưng.

    RANSAC bỏ qua các điểm nằm trên xe đang chạy, chỉ giữ chuyển động chung của nền (tức của camera).
    """
    points = cv2.goodFeaturesToTrack(prev_gray, maxCorners=300, qualityLevel=0.01, minDistance=8)
    if points is None or len(points) < 3:
        return None
    # nextPts chỉ là chỗ ghi kết quả (không bật OPTFLOW_USE_INITIAL_FLOW nên giá trị ban đầu bị bỏ qua)
    moved, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, points, np.empty_like(points))
    ok = status.ravel() == 1
    if ok.sum() < 3:
        return None
    matrix, _ = cv2.estimateAffinePartial2D(points[ok], moved[ok], method=cv2.RANSAC)
    if matrix is None:
        return None
    shift = float(np.hypot(matrix[0, 2], matrix[1, 2]))
    zoom = abs(float(np.hypot(matrix[0, 0], matrix[1, 0])) - 1.0) * 100
    return shift, zoom


def measure_camera_motion(video_path: str, samples_per_second: float = 2.0) -> list[MotionSample]:
    """Lấy mẫu `samples_per_second` frame mỗi giây và đo chuyển động camera giữa các mẫu liên tiếp."""
    capture = cv2.VideoCapture(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps / samples_per_second))
    samples: list[MotionSample] = []
    prev_gray = None
    index = 0
    while capture.grab():
        if index % step == 0:
            _, frame = capture.retrieve()
            height = round(frame.shape[0] * ANALYSIS_WIDTH / frame.shape[1])
            gray = cv2.cvtColor(cv2.resize(frame, (ANALYSIS_WIDTH, height)), cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                # Cắt cảnh: độ sáng trung bình trên ảnh rất nhỏ thay đổi mạnh (ngưỡng 40/255 chọn theo thực nghiệm)
                thumb_diff = cv2.absdiff(cv2.resize(prev_gray, (64, 36)), cv2.resize(gray, (64, 36)))
                scene_cut = float(thumb_diff.mean()) > 40
                motion = _global_motion(prev_gray, gray)
                shift, zoom = motion if motion is not None else (float("inf"), float("inf"))
                samples.append(MotionSample(index / fps, shift, zoom, scene_cut or motion is None))
            prev_gray = gray
        index += 1
    capture.release()
    return samples


def stable_segments(
    samples: list[MotionSample],
    max_shift_px: float = 2.0,
    max_zoom_pct: float = 1.0,
    min_length_s: float = 30.0,
) -> list[tuple[float, float]]:
    """Các đoạn (bắt đầu, kết thúc) tính bằng giây mà camera đứng yên liên tục ít nhất `min_length_s`.

    Ngưỡng mặc định: camera trên chân máy đo được ~0.2px giữa hai mẫu; zoom/lia tay cho hàng chục px.
    """
    segments = []
    start = None
    for sample in samples:
        steady = sample.shift_px <= max_shift_px and sample.zoom_pct <= max_zoom_pct and not sample.scene_cut
        if steady and start is None:
            start = sample.time_s
        elif not steady and start is not None:
            if sample.time_s - start >= min_length_s:
                segments.append((start, sample.time_s))
            start = None
    if start is not None and samples and samples[-1].time_s - start >= min_length_s:
        segments.append((start, samples[-1].time_s))
    return segments
