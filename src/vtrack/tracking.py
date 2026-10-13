"""Vòng lặp phát hiện -> theo dõi -> bỏ phiếu lớp theo track trên từng frame video.
Dùng chung cho pipeline tạo video (pipeline.py) và bộ chấm tracking (scripts/eval_tracking.py),
để số liệu đánh giá đo đúng thứ mà demo chạy."""

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import supervision as sv
from ultralytics import YOLO

from vtrack import CLASS_NAMES
from vtrack.counting import TrackClassVoter
from vtrack.models import class_mapping
from vtrack.mot_format import MotBox

# ByteTrack/BoT-SORT dùng cả detection điểm thấp cho bước ghép thứ hai (nối track bị che khuất);
# việc tạo track mới do `new_track_thresh` trong config tracker quyết định, nên conf đầu vào để thấp.
DEFAULT_DETECTION_CONF = 0.1
# Tracker đã chỉnh trên tập dev VnTraffic, tắt bù chuyển động camera (GMC): camera trong phạm vi đồ án đứng yên
# (tắt GMC: IDR như nhau, pipeline OpenVINO nhanh hơn 1.4 lần, outputs/eval_tracking/gmc_dev/). Từ 14/10/2026 (mô hình
# v6) nâng ngưỡng mở track mới 0.25 -> 0.6 và ngưỡng ghép thứ nhất 0.25 -> 0.4: trên dev đổi ID 93.5 -> 62.5, IDR giữ
# nguyên (outputs/eval_tracking/tuning_v6/, chú thích trong file cấu hình). Trước đó: configs/botsort_tuned_nogmc.yaml.
# Video camera rung/quay theo thì dùng configs/botsort_tuned.yaml (có GMC).
DEFAULT_TRACKER = "configs/botsort_v6.yaml"
# Track phải sống đủ 3 frame mới hiện/được đếm: bằng min_hits mặc định của SORT. Trên VnTraffic giảm đổi ID
# 128 -> 118 và MOTA +0.04, đổi lại IDR -0.004 (mất 2 frame đầu của mỗi xe). Bộ chấm tracking vẫn báo kết quả thô.
DEFAULT_MIN_TRACK_AGE = 3


class TrackAgeFilter:
    """Chỉ cho hiện track khi nó đã xuất hiện ở ít nhất `min_age` frame (tính cả frame hiện tại).

    Lọc track "rác" sống vài frame (box chập chờn trong cụm xe đông) mà vẫn chạy online được: quyết định
    ở mỗi frame chỉ dựa vào quá khứ, không cần biết trước track sẽ sống bao lâu. Cái giá là mỗi track
    thật mất `min_age - 1` frame đầu. Không ảnh hưởng tới tracker (tracker vẫn nhận mọi detection).
    """

    def __init__(self, min_age: int = 1) -> None:
        if min_age < 1:
            raise ValueError(f"min_age phải >= 1, nhận {min_age}")
        self.min_age = min_age
        self._seen: Counter[int] = Counter()

    def update(self, track_ids: np.ndarray) -> np.ndarray:
        """Ghi nhận các track của frame hiện tại; trả về mặt nạ bool những track đủ tuổi để hiện."""
        keep = np.zeros(len(track_ids), dtype=bool)
        for i, track_id in enumerate(track_ids):
            self._seen[int(track_id)] += 1
            keep[i] = self._seen[int(track_id)] >= self.min_age
        return keep


def filter_young_tracks(boxes: list[MotBox], min_age: int) -> list[MotBox]:
    """Áp TrackAgeFilter lên kết quả tracking đã lưu, theo đúng thứ tự frame — cho kết quả y hệt bật bộ lọc
    lúc chạy (bộ lọc không tác động ngược vào tracker), nên chấm được nhiều giá trị N mà không chạy lại video."""
    # Mỗi track xuất hiện tối đa 1 lần mỗi frame, nên đưa cả chuỗi box đã xếp theo frame vào một lần
    # cũng đếm tuổi y như gọi update() từng frame
    ordered = sorted(boxes, key=lambda b: (b.frame, b.track_id))
    keep = TrackAgeFilter(min_age).update(np.array([b.track_id for b in ordered], dtype=int))
    return [box for box, kept in zip(ordered, keep) if kept]


@dataclass
class TrackedFrame:
    index: int  # chỉ số frame trong video gốc, đánh từ 0
    image: np.ndarray
    detections: sv.Detections  # có tracker_id; class_id đã là lớp dự án sau bỏ phiếu theo track
    track_ids: np.ndarray
    class_ids: np.ndarray


def track_video(
    source: str | Path,
    model: YOLO,
    tracker: str,
    imgsz: int,
    conf: float = DEFAULT_DETECTION_CONF,
    vid_stride: int = 1,
    max_frames: int | None = None,
    min_track_age: int = 1,
) -> Iterator[TrackedFrame]:
    """Sinh lần lượt từng frame đã được theo dõi. `max_frames` giới hạn số frame đọc (vd. bằng độ dài nhãn chuẩn);
    `min_track_age` > 1 ẩn track mới cho tới khi nó sống đủ số frame đó (xem TrackAgeFilter)."""
    mapping = class_mapping(model.names)
    if not mapping:
        raise ValueError(f"Mô hình không có lớp phương tiện nào: {model.names}")
    voter = TrackClassVoter()
    age_filter = TrackAgeFilter(min_track_age)
    frames = sv.get_video_frames_generator(str(source), stride=vid_stride)

    for step, image in enumerate(frames):
        if max_frames is not None and step >= max_frames:
            break
        # persist=False ở frame đầu để tạo tracker mới khi dùng lại model cho video/tracker khác
        result = model.track(
            image, persist=step > 0, tracker=tracker, conf=conf, imgsz=imgsz,
            classes=list(mapping), verbose=False,
        )[0]
        detections = sv.Detections.from_ultralytics(result)
        if detections.tracker_id is None or detections.class_id is None:
            # LineZone chỉ cảnh báo rồi bỏ qua detections không có tracker_id -> thay bằng rỗng
            detections = sv.Detections.empty()
            track_ids = class_ids = np.empty(0, dtype=int)
        else:
            # Bỏ phiếu lớp trên mọi detection (cả track chưa đủ tuổi: thêm phiếu thì lớp càng chắc), rồi mới lọc tuổi
            model_classes = np.array([mapping[int(c)] for c in detections.class_id])
            voted = voter.update(detections.tracker_id, model_classes, detections.confidence)
            keep = age_filter.update(detections.tracker_id)
            track_ids, class_ids = detections.tracker_id[keep], voted[keep]
            # chỉ số bằng mặt nạ bool luôn trả về Detections (kiểu khai báo của supervision rộng hơn)
            detections = cast(sv.Detections, detections[keep])
            detections.class_id = class_ids
            detections.data["class_name"] = np.array([CLASS_NAMES[c] for c in class_ids])
        yield TrackedFrame(
            index=step * vid_stride, image=image, detections=detections,
            track_ids=track_ids, class_ids=class_ids,
        )


def to_mot_boxes(tracked: TrackedFrame) -> list[MotBox]:
    """Box của một frame đã theo dõi -> định dạng MOT (frame đánh từ 1, toạ độ pixel góc trên-trái)."""
    detections = tracked.detections
    if detections.confidence is None:  # frame không có track nào
        return []
    return [
        MotBox(frame=tracked.index + 1, track_id=int(track_id), left=float(x1), top=float(y1),
               width=float(x2 - x1), height=float(y2 - y1), confidence=float(score))
        for (x1, y1, x2, y2), track_id, score in zip(detections.xyxy, tracked.track_ids, detections.confidence)
    ]
