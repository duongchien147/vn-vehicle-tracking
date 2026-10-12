"""Chọn thời điểm lấy frame trong video để gán nhãn (theo đoạn thời gian và bước nhảy cố định), và loại frame gần trùng."""

import re

import cv2
import numpy as np


def parse_timestamp(text: str) -> float:
    """'1:35' -> 95.0 giây; '1:02:03' -> 3723.0 giây."""
    seconds = 0.0
    for part in text.strip().split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


def sample_timestamps(start_s: float, end_s: float, every_s: float) -> list[float]:
    """Các thời điểm start, start + every, ... nhỏ hơn end (không lấy chính `end` để đoạn liền kề không trùng frame)."""
    if every_s <= 0:
        raise ValueError("every_s phải > 0")
    timestamps = []
    index = 0
    while start_s + index * every_s < end_s:  # nhân theo chỉ số thay vì cộng dồn để không tích luỹ sai số làm tròn
        timestamps.append(start_s + index * every_s)
        index += 1
    return timestamps


FRAME_NAME_PATTERN = re.compile(r"^(?P<clip>.+)_(?P<minutes>\d+)m(?P<seconds>\d+)s$")


def parse_frame_name(stem: str) -> tuple[str, float] | None:
    """Tên frame do extract_label_frames.py đặt, 'pvd_goc_b_00m05s' -> ('pvd_goc_b', 5.0); tên khác -> None."""
    match = FRAME_NAME_PATTERN.match(stem)
    if match is None:
        return None
    return match["clip"], int(match["minutes"]) * 60 + float(match["seconds"])


# Chữ ký frame để so trùng: ảnh xám thu nhỏ 160x90 (đủ để thấy xe dịch chuyển, bỏ qua nhiễu nén video)
SIGNATURE_SIZE = (160, 90)


def frame_signature(image: np.ndarray) -> np.ndarray:
    """Ảnh xám thu nhỏ của frame (float), dùng để đo hai frame giống nhau tới đâu."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return cv2.resize(gray, SIGNATURE_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)


def keep_distinct(signatures: list[np.ndarray], min_distance: float) -> list[int]:
    """Chỉ số các frame được giữ: duyệt theo thứ tự, bỏ frame có chênh lệch điểm ảnh trung bình (thang 0–255) với một
    frame đã giữ nhỏ hơn min_distance. Dùng cho video bị ghép lặp lại, nơi cùng một đoạn xuất hiện nhiều lần."""
    kept: list[int] = []
    for index, signature in enumerate(signatures):
        if all(np.abs(signature - signatures[other]).mean() >= min_distance for other in kept):
            kept.append(index)
    return kept
