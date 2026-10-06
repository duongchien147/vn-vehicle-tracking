"""Nạp mô hình YOLO (.pt hoặc thư mục *_openvino_model), ánh xạ lớp của mô hình về 4 lớp dự án,
và lấy kích thước ảnh (imgsz) mà mô hình đã được train/export."""

from pathlib import Path

import yaml
from ultralytics import YOLO

from vtrack import CLASS_NAMES

# Mô hình mặc định hiện tại (v5 = v3 + UVH-26, imgsz 960; chọn theo tiêu chí chốt trước: mAP50 val Hà Nội + IDR VnTraffic — xem ghi chú dự án)
DEFAULT_WEIGHTS = "weights/yolo26n_vn_v5_best.pt"

# Dùng khi không đọc được imgsz lúc train từ file trọng số; 640 là mặc định của ultralytics
FALLBACK_IMGSZ = 640

# Tên lớp của mô hình (COCO hoặc dataset fine-tune) -> tên lớp dự án
NAME_ALIASES = {
    "motorcycle": "motorbike",
    "motorbike": "motorbike",
    "car": "car",
    "bus": "bus",
    "truck": "truck",
}


def load_model(weights: str) -> YOLO:
    return YOLO(weights, task="detect")


def class_mapping(model_names: dict[int, str]) -> dict[int, int]:
    """id lớp của mô hình -> id lớp dự án; các lớp không phải phương tiện bị bỏ qua."""
    mapping = {}
    for model_id, name in model_names.items():
        alias = NAME_ALIASES.get(name.lower())
        if alias is not None:
            mapping[model_id] = CLASS_NAMES.index(alias)
    return mapping


def training_imgsz(model: YOLO) -> int | None:
    """imgsz mà mô hình đã được train (file .pt) hoặc export (thư mục OpenVINO...); None nếu không rõ.

    Suy luận phải dùng đúng giá trị này: mô hình v2 train ở 960 mà chạy ở 640 thì xe nhỏ bị bỏ sót nhiều.
    - File .pt: ultralytics giữ lại `imgsz` lúc train trong `model.overrides`.
    - Mô hình export: `imgsz` nằm trong `metadata.yaml` cạnh file mô hình.
    """
    value = model.overrides.get("imgsz")
    if value is None:
        metadata_path = Path(str(model.model_name)) / "metadata.yaml"
        if metadata_path.exists():
            value = yaml.safe_load(metadata_path.read_text(encoding="utf-8")).get("imgsz")
    if value is None:
        return None
    # imgsz có thể là một số hoặc [cao, rộng]; ultralytics thu cạnh dài của ảnh về đúng giá trị lớn nhất
    return max(value) if isinstance(value, (list, tuple)) else int(value)
