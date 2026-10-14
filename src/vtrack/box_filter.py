"""Bỏ box trùng bất kể lớp (NMS không phân lớp) trước khi đưa box vào tracker.

YOLO26 là mô hình end-to-end: ultralytics bỏ qua NMS hoàn toàn với loại mô hình này (nhánh `end2end` trong
ultralytics/utils/nms.py), nên trên cùng một xe vẫn có thể còn box phụ: khác lớp (vừa "xe máy" vừa "ô tô"), hoặc chỉ ôm
một phần xe. Tracker của ultralytics ghép box theo IoU, không xét lớp, nên box phụ mở track thừa hoặc giành lượt ghép
của track thật, gây đổi ID. Khác với nâng ngưỡng mở track mới, bộ lọc này không chặn xe nhỏ đứng riêng ở xa.
"""

import numpy as np
from ultralytics import YOLO

# Sự kiện ultralytics gọi sau khi hậu xử lý xong mỗi ảnh; tracker cũng cập nhật ở đúng sự kiện này, nên bộ lọc phải
# đứng trước callback của tracker trong danh sách.
CALLBACK_EVENT = "on_predict_postprocess_end"


def agnostic_nms_keep(xyxy: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """NMS tham lam bất kể lớp: duyệt box theo điểm giảm dần, bỏ box có IoU > `iou_threshold` với box đã giữ.
    Trả về chỉ số các box được giữ, theo thứ tự tăng dần (giữ nguyên thứ tự gốc)."""
    order = np.argsort(-scores, kind="stable")
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    kept: list[int] = []
    for index in order:
        if kept:
            others = xyxy[kept]
            inter_w = np.clip(np.minimum(xyxy[index, 2], others[:, 2]) - np.maximum(xyxy[index, 0], others[:, 0]), 0, None)
            inter_h = np.clip(np.minimum(xyxy[index, 3], others[:, 3]) - np.maximum(xyxy[index, 1], others[:, 1]), 0, None)
            inter = inter_w * inter_h
            iou = inter / (areas[index] + areas[kept] - inter)
            if (iou > iou_threshold).any():
                continue
        kept.append(int(index))
    return np.sort(np.array(kept, dtype=int))


class DuplicateBoxFilter:
    """Callback của ultralytics: lọc box trùng trong kết quả của từng ảnh, trước khi tracker nhận."""

    def __init__(self, iou_threshold: float) -> None:
        self.iou_threshold = iou_threshold

    def __call__(self, predictor) -> None:
        for i, result in enumerate(predictor.results):
            boxes = result.boxes
            if boxes is None or len(boxes) < 2:
                continue
            keep = agnostic_nms_keep(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), self.iou_threshold)
            if len(keep) < len(boxes):
                predictor.results[i] = result[keep]


def set_agnostic_nms(model: YOLO, iou_threshold: float | None) -> None:
    """Bật (đặt ngưỡng IoU) hoặc tắt (None) bộ lọc box trùng cho mọi lần predict/track sau đó của `model`.
    Gọi lại nhiều lần không bị cộng dồn; bộ lọc luôn đứng đầu danh sách để chạy trước tracker."""
    callbacks = model.callbacks[CALLBACK_EVENT]
    callbacks[:] = [callback for callback in callbacks if not isinstance(callback, DuplicateBoxFilter)]
    if iou_threshold is not None:
        callbacks.insert(0, DuplicateBoxFilter(iou_threshold))
