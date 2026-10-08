"""Cách hiển thị xe trên video kết quả: số thứ tự và nhãn chữ.

Số thứ tự hiển thị: Tracker cấp ID cho cả những track "thử" (phát hiện chập chờn, bị
pipeline ẩn vì sống < 3 frame), nên ID gốc nhảy cóc lên hàng nghìn dù chỉ vài trăm xe. Ở đây xe được đánh số 1, 2, 3...
theo thứ tự thật sự hiện lên trên video. Chỉ đổi cách hiển thị: ID gốc của tracker vẫn ghi trong CSV để truy ngược."""

import numpy as np


class DisplayIds:
    def __init__(self) -> None:
        self._numbers: dict[int, int] = {}  # ID gốc của tracker -> số thứ tự hiển thị

    def get(self, track_id: int) -> int:
        """Số hiển thị của track; track mới thấy lần đầu nhận số kế tiếp."""
        if track_id not in self._numbers:
            self._numbers[track_id] = len(self._numbers) + 1
        return self._numbers[track_id]


# Chỉ ghi chữ "#ID lớp" cho box cao từ 4% chiều cao khung hình trở lên (~43 px ở 1080p); xe nhỏ hơn chỉ vẽ khung.
# Lý do: ở cuối đường, hàng chục xe nhỏ chen nhau làm nhãn chữ đè lên nhau thành một đám không đọc được.
LABEL_MIN_HEIGHT_FRACTION = 0.04


def label_mask(xyxy: np.ndarray, frame_height: int, min_fraction: float = LABEL_MIN_HEIGHT_FRACTION) -> np.ndarray:
    """Mảng bool: box nào đủ lớn để ghi nhãn chữ (chỉ ảnh hưởng hiển thị, không ảnh hưởng phát hiện/đếm)."""
    heights = xyxy[:, 3] - xyxy[:, 1]
    return heights >= min_fraction * frame_height
