"""Cách hiển thị xe trên video kết quả: số thứ tự, lớp hiển thị và nhãn chữ.

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


class CountedClasses:
    """Khoá lớp hiển thị của một xe ở đúng lớp nó được đếm khi cắt vạch.

    Lớp bỏ phiếu theo track vẫn có thể đổi sau khi xe đã được đếm: ví dụ xe buýt ban đêm được gọi đúng là "bus" lúc ở
    gần, rồi bị gọi chắc chắn là "truck" khi đi xa chỉ còn thấy đuôi xe, đủ phiếu để vượt. Khoá lại để video khớp bảng
    đếm và CSV. Chỉ ảnh hưởng hiển thị: lớp dùng để đếm và để chấm điểm không đổi."""

    def __init__(self) -> None:
        self._locked: dict[int, int] = {}  # ID gốc của tracker -> lớp lúc cắt vạch lần đầu

    def lock(self, track_id: int, class_id: int) -> None:
        self._locked.setdefault(track_id, class_id)

    def shown(self, track_ids: np.ndarray, class_ids: np.ndarray) -> np.ndarray:
        """Lớp để vẽ: lớp đã khoá nếu xe đã được đếm, ngược lại là lớp bỏ phiếu hiện tại."""
        return np.array([self._locked.get(int(t), int(c)) for t, c in zip(track_ids, class_ids)], dtype=int)


# Chỉ ghi chữ "#ID lớp" cho box cao từ 4% chiều cao khung hình trở lên (~43 px ở 1080p); xe nhỏ hơn chỉ vẽ khung.
# Lý do: ở cuối đường, hàng chục xe nhỏ chen nhau làm nhãn chữ đè lên nhau thành một đám không đọc được.
LABEL_MIN_HEIGHT_FRACTION = 0.04


def label_mask(xyxy: np.ndarray, frame_height: int, min_fraction: float = LABEL_MIN_HEIGHT_FRACTION) -> np.ndarray:
    """Mảng bool: box nào đủ lớn để ghi nhãn chữ (chỉ ảnh hưởng hiển thị, không ảnh hưởng phát hiện/đếm)."""
    heights = xyxy[:, 3] - xyxy[:, 1]
    return heights >= min_fraction * frame_height
