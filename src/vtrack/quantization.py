"""Chuẩn bị lượng tử hoá INT8 (OpenVINO + NNCF) cho mô hình YOLO: chọn ảnh hiệu chỉnh.

Ảnh hiệu chỉnh lấy ngẫu nhiên từ tập TRAIN, không lấy từ val như mặc định của ultralytics: val chứa 10 frame
Hà Nội là tập dev dùng để quyết định có nhận bản INT8 hay không, hiệu chỉnh trên chính tập đó sẽ thiên lệch.
"""

import random
from pathlib import Path

import yaml

CALIBRATION_IMAGES = 300  # số ảnh NNCF dùng mặc định (subset_size=300) và ultralytics khuyến nghị tối thiểu
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def sample_calibration_images(image_dir: Path, count: int = CALIBRATION_IMAGES, seed: int = 0) -> list[Path]:
    """Chọn ngẫu nhiên (cố định theo seed) `count` ảnh trong thư mục; ít hơn thì lấy hết.

    Sắp xếp trước khi chọn để kết quả không phụ thuộc thứ tự liệt kê file của hệ điều hành.
    """
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if count >= len(images):
        return images
    return sorted(random.Random(seed).sample(images, count))


def write_calibration_dataset(images: list[Path], names: dict[int, str], out_dir: Path) -> Path:
    """Ghi danh sách ảnh + data.yaml kiểu ultralytics (val = danh sách ảnh hiệu chỉnh), trả về đường dẫn data.yaml.

    ultralytics lấy ảnh hiệu chỉnh INT8 từ split `val` của data.yaml, nên trỏ val (và train, cho đủ khoá) vào danh sách.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    list_path = out_dir / "calibration_images.txt"
    list_path.write_text("".join(f"{p.resolve().as_posix()}\n" for p in images), encoding="utf-8")
    data = {"train": list_path.resolve().as_posix(), "val": list_path.resolve().as_posix(), "names": names}
    data_yaml = out_dir / "calibration_data.yaml"
    data_yaml.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return data_yaml
