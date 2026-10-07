"""Độ phức tạp của mô hình: số tham số và GFLOPs, đo bằng MỘT công cụ cho mọi họ mô hình (torch.utils.flop_counter
có sẵn trong PyTorch) ở đúng kích thước đầu vào mà mô hình nhận khi chạy video 1080p.

Quy ước: 1 phép nhân-cộng = 2 FLOPs (giống GFLOPs ultralytics in ra). FlopCounterMode đếm các phép chính (conv,
matmul, attention); phép phụ như grid_sample trong deformable attention của RT-DETR không được đếm (không đáng kể).
Đo một lần truyền xuôi (backbone + neck + head), batch 1, không gồm tiền xử lý / hậu xử lý.
"""

import math
from dataclasses import dataclass

import torch
from torch.utils.flop_counter import FlopCounterMode

VIDEO_HW = (1080, 1920)  # video đánh giá đều 1080p
YOLO_STRIDE = 32  # ultralytics đệm ảnh letterbox tới bội số của stride lớn nhất


@dataclass(frozen=True)
class Complexity:
    params_m: float  # triệu tham số
    gflops: float
    input_hw: tuple[int, int]  # kích thước tensor đầu vào đã đo


def yolo_input_hw(imgsz: int, frame_hw: tuple[int, int] = VIDEO_HW, stride: int = YOLO_STRIDE) -> tuple[int, int]:
    """Kích thước ultralytics đưa vào mạng khi suy luận (letterbox tối thiểu: giữ tỉ lệ, cạnh dài = imgsz,
    đệm cạnh ngắn tới bội số stride). Video 1920x1080 @960 -> 544x960."""
    height, width = frame_hw
    scale = imgsz / max(height, width)
    return (math.ceil(round(height * scale) / stride) * stride, math.ceil(round(width * scale) / stride) * stride)


def count_gflops(module: torch.nn.Module, *args: torch.Tensor, **kwargs: torch.Tensor) -> float:
    counter = FlopCounterMode(display=False)
    with torch.inference_mode(), counter:
        module(*args, **kwargs)
    return counter.get_total_flops() / 1e9


def count_params_m(module: torch.nn.Module) -> float:
    return sum(p.numel() for p in module.parameters()) / 1e6
