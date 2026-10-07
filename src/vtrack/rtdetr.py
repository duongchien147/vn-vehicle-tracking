"""Chạy mô hình RT-DETRv2 (họ Transformer, train bằng notebooks/kaggle_train_rtdetr) trên CPU và trả kết quả cùng
định dạng `YoloBox` với YOLO — để mọi bộ chấm/đo tốc độ dùng chung cho cả hai họ mô hình.

Tiền xử lý phải giống hệt lúc train (train_rtdetr.py): ảnh RGB, resize thẳng về hình vuông imgsz x imgsz bằng
torchvision (có antialias), chia 255, không chuẩn hoá mean/std. Hậu xử lý là của tác giả (không có NMS):
sigmoid điểm từng lớp rồi lấy top-300 cặp (truy vấn, lớp).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms.v2 import functional as F
from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

from vtrack.label_review import YoloBox
from vtrack.models import class_mapping


def to_yolo_boxes(
    scores: torch.Tensor, labels: torch.Tensor, boxes_xyxy: torch.Tensor, class_ids: dict[int, int], conf: float,
) -> list[YoloBox]:
    """Kết quả hậu xử lý (box xyxy chuẩn hoá 0–1) -> box YOLO theo lớp dự án; bỏ điểm < conf và lớp lạ."""
    result = []
    for score, label, (x1, y1, x2, y2) in zip(scores.tolist(), labels.tolist(), boxes_xyxy.tolist()):
        if score < conf or label not in class_ids:
            continue
        result.append(YoloBox(class_ids[label], (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, score))
    return result


@dataclass
class RtDetrDetector:
    model: RTDetrV2ForObjectDetection
    processor: RTDetrImageProcessor
    imgsz: int
    class_ids: dict[int, int]  # id lớp của mô hình -> id lớp dự án

    @classmethod
    def load(cls, model_dir: str | Path, imgsz: int | None = None) -> "RtDetrDetector":
        """`imgsz` mặc định = kích thước lúc train (lưu trong preprocessor_config.json)."""
        model = RTDetrV2ForObjectDetection.from_pretrained(str(Path(model_dir).resolve())).eval()
        processor = RTDetrImageProcessor.from_pretrained(str(Path(model_dir).resolve()))
        size = imgsz or int(processor.size["height"])
        names = {int(k): v for k, v in (model.config.id2label or {}).items()}
        return cls(model, processor, size, class_mapping(names))

    @torch.inference_mode()
    def predict(self, image_bgr: np.ndarray, conf: float) -> list[YoloBox]:
        rgb = torch.from_numpy(np.ascontiguousarray(image_bgr[:, :, ::-1])).permute(2, 0, 1)
        pixels = F.resize(rgb, [self.imgsz, self.imgsz]).float().div(255).unsqueeze(0)
        outputs = self.model(pixel_values=pixels)
        result = self.processor.post_process_object_detection(outputs, threshold=0.0)[0]
        return to_yolo_boxes(result["scores"], result["labels"], result["boxes"], self.class_ids, conf)
