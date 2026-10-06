"""Ổn định nhãn lớp theo track: một xe có thể bị phân loại "nhảy" car <-> truck giữa các frame,
nên lớp cuối cùng của mỗi track được chọn bằng bỏ phiếu có trọng số theo độ tin cậy."""

from collections import defaultdict

import numpy as np


class TrackClassVoter:
    def __init__(self) -> None:
        # track_id -> {class_id: tổng độ tin cậy}
        self._votes: defaultdict[int, defaultdict[int, float]] = defaultdict(
            lambda: defaultdict(float)
        )

    def update(
        self,
        tracker_ids: np.ndarray,
        class_ids: np.ndarray,
        confidences: np.ndarray | None = None,
    ) -> np.ndarray:
        """Cộng phiếu của frame hiện tại và trả về lớp đã bỏ phiếu cho từng detection."""
        if confidences is None:
            confidences = np.ones(len(class_ids), dtype=float)
        voted = np.empty(len(class_ids), dtype=int)
        for i, (tid, cls, conf) in enumerate(zip(tracker_ids, class_ids, confidences)):
            votes = self._votes[int(tid)]
            votes[int(cls)] += float(conf)
            voted[i] = max(votes, key=votes.__getitem__)
        return voted

    def class_of(self, tracker_id: int) -> int | None:
        votes = self._votes.get(int(tracker_id))
        return max(votes, key=votes.__getitem__) if votes else None
