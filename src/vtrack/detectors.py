"""Nạp detector bất kể họ mô hình (YOLO .pt / OpenVINO, hay RT-DETRv2) thành một giao diện chung:
ảnh BGR + ngưỡng điểm -> danh sách `YoloBox` theo 4 lớp dự án. Bộ chấm mAP và đo tốc độ chỉ làm việc với giao diện
này, nên so sánh CNN và Transformer đi qua đúng cùng một đường chấm."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from vtrack.box_filter import set_agnostic_nms
from vtrack.complexity import Complexity, count_gflops, count_params_m, yolo_input_hw
from vtrack.label_review import YoloBox
from vtrack.models import FALLBACK_IMGSZ, load_model, training_imgsz
from vtrack.pseudo_label import predict_teacher

RTDETR_MODEL_TYPE = "rt_detr_v2"  # trường `model_type` trong config.json do save_pretrained của transformers ghi


@dataclass
class Detector:
    name: str
    imgsz: int
    predict: Callable[[np.ndarray, float], list[YoloBox]]  # (ảnh BGR, ngưỡng điểm) -> box lớp dự án
    # Số tham số + GFLOPs ở đúng đầu vào khi chạy video 1080p; None nếu không phải mô hình PyTorch (OpenVINO)
    complexity: Callable[[], Complexity] | None = None


def is_rtdetr_dir(path: str | Path) -> bool:
    """Thư mục mô hình RT-DETRv2 (save_pretrained), phân biệt với .pt / thư mục OpenVINO của YOLO."""
    config_path = Path(path) / "config.json"
    if not config_path.is_file():
        return False
    return json.loads(config_path.read_text(encoding="utf-8")).get("model_type") == RTDETR_MODEL_TYPE


def load_detector(weights: str, imgsz: int | None = None, agnostic_nms_iou: float | None = None) -> Detector:
    """`imgsz` None = kích thước lúc train/export của mô hình. `agnostic_nms_iou` bật bộ lọc box trùng bất kể lớp
    (vtrack.box_filter, chỉ cho YOLO) để chấm đúng thứ tracker nhận khi bật bộ lọc này."""
    if is_rtdetr_dir(weights):
        if agnostic_nms_iou is not None:
            raise ValueError("Bộ lọc box trùng chỉ dùng cho mô hình YOLO")
        # Import muộn: transformers nạp chậm (vài giây) và chỉ cần khi thật sự dùng RT-DETRv2
        from vtrack.rtdetr import RtDetrDetector

        rtdetr = RtDetrDetector.load(weights, imgsz)
        square = (rtdetr.imgsz, rtdetr.imgsz)  # RT-DETR kéo giãn ảnh về hình vuông

        def rtdetr_complexity() -> Complexity:
            pixels = torch.zeros(1, 3, *square)
            return Complexity(count_params_m(rtdetr.model), count_gflops(rtdetr.model, pixel_values=pixels), square)

        return Detector(Path(weights).name, rtdetr.imgsz, rtdetr.predict, rtdetr_complexity)

    model = load_model(weights)
    set_agnostic_nms(model, agnostic_nms_iou)
    size = imgsz or training_imgsz(model) or FALLBACK_IMGSZ
    network = model.model

    def yolo_complexity() -> Complexity:
        if not isinstance(network, torch.nn.Module):
            raise TypeError("Không đo được độ phức tạp của mô hình không phải PyTorch")
        input_hw = yolo_input_hw(size)
        return Complexity(count_params_m(network), count_gflops(network, torch.zeros(1, 3, *input_hw)), input_hw)

    # predict_teacher gộp người lái vào xe máy với mô hình COCO -> cùng quy ước nhãn với dữ liệu Việt Nam
    name = Path(weights).stem + ("" if agnostic_nms_iou is None else f"+nms{agnostic_nms_iou}")
    return Detector(name, size, lambda image, conf: predict_teacher(model, image, size, conf),
                    yolo_complexity if Path(weights).suffix == ".pt" else None)
