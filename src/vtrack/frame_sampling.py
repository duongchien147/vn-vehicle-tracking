"""Chọn thời điểm lấy frame trong video để gán nhãn (theo đoạn thời gian và bước nhảy cố định)."""

import re


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
