"""Nạp mô hình YOLO (.pt hoặc thư mục *_openvino_model), ánh xạ lớp của mô hình về 4 lớp dự án,
và lấy kích thước ảnh (imgsz) mà mô hình đã được train/export."""

from pathlib import Path

import yaml
from ultralytics import YOLO

from vtrack import CLASS_NAMES

# Mô hình chính (từ 15/10/2026): YOLO26s-v6 = YOLO26s học dữ liệu v6 (v5 + 288 frame đêm), seed 42; nhận theo tiêu chí
# chốt trước configs/accept_yolo26s_v6.yaml. Bối cảnh triển khai là máy chủ có GPU, nơi YOLO26s nhanh ngang YOLO26n mà
# chính xác hơn. Trước đó: v6 (YOLO26n, 13–15/10/2026), v5 (YOLO26n, tới 13/10/2026).
DEFAULT_WEIGHTS = "weights/yolo26s_vn_v6_best.pt"
# Bản nhẹ cho máy không có GPU rời (laptop, thiết bị đặt tại camera): YOLO26n học cùng dữ liệu v6; web demo chạy trực
# tiếp trên laptop dùng bản OpenVINO của nó
LIGHT_WEIGHTS = "weights/yolo26n_vn_v6_best.pt"

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


# Thiết bị chạy mô hình OpenVINO, ghi sau dấu '#': 'weights/x_openvino_model#cpu' hoặc '#gpu' (GPU Intel tích hợp).
# Không ghi thì ultralytics để OpenVINO tự chọn (AUTO): trên laptop có GPU Intel tích hợp, mô hình FP32 được đưa lên GPU
# đó còn mô hình INT8 có thể chạy trên CPU -> khi đo tốc độ hay so sánh phải ghi rõ thiết bị (đã kiểm 10/10/2026).
OPENVINO_DEVICES = {"cpu": "intel:cpu", "gpu": "intel:gpu"}


def split_device(weights: str) -> tuple[str, str | None]:
    """'x_openvino_model#gpu' -> ('x_openvino_model', 'intel:gpu'); không có '#' -> (weights, None)."""
    path, _, device = weights.partition("#")
    if not device:
        return path, None
    if device not in OPENVINO_DEVICES:
        raise ValueError(f"Thiết bị không hợp lệ '{device}', chọn một trong {sorted(OPENVINO_DEVICES)}")
    return path, OPENVINO_DEVICES[device]


def device_name(weights: str) -> str:
    """Thiết bị sẽ chạy mô hình, để ghi kèm số frame/s: OpenVINO theo hậu tố '#', còn lại GPU NVIDIA nếu có, không thì CPU."""
    _, device = split_device(weights)
    if device == OPENVINO_DEVICES["gpu"]:
        return "GPU Intel tích hợp (OpenVINO)"
    if device == OPENVINO_DEVICES["cpu"]:
        return "CPU (OpenVINO)"
    import torch  # nạp muộn: chỉ cần khi chạy PyTorch

    return f"GPU {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"


def load_model(weights: str) -> YOLO:
    path, device = split_device(weights)
    model = YOLO(path, task="detect")
    if device is not None:
        model.overrides["device"] = device  # mọi lần predict/track sau đó đều chạy trên thiết bị này
    return model


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
