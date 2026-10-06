"""Tải các dataset nguồn khai báo trong configs/datasets.yaml (mục `download` của từng nguồn) vào data/raw/.
Bộ nào đã có (thấy data.yaml) thì bỏ qua. Roboflow cần ROBOFLOW_API_KEY trong file .env.

Ví dụ:
    python scripts/download_datasets.py
"""

import argparse
import shutil
from pathlib import Path

import yaml
from PIL import Image

from vtrack.dataset import VISDRONE_NAMES, visdrone_to_yolo_lines

VISDRONE_ZIP_DIR = Path("data/raw/visdrone_zip")


def read_env(path: Path = Path(".env")) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    return dict(line.split("=", 1) for line in lines if "=" in line and not line.startswith("#"))


def download_roboflow(spec: dict, target: Path) -> None:
    from roboflow import Roboflow  # chỉ cần khi thật sự tải từ Roboflow

    api_key = read_env().get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Thiếu ROBOFLOW_API_KEY trong .env")
    project = Roboflow(api_key=api_key).workspace(spec["workspace"]).project(spec["project"])
    project.version(spec["version"]).download("yolov8", location=str(target))


def download_visdrone(spec: dict, target: Path) -> None:
    """Tải VisDrone2019-DET (một split) và đổi sang cấu trúc giống Roboflow: <split>/images, <split>/labels, data.yaml."""
    from ultralytics.utils import ASSETS_URL
    from ultralytics.utils.downloads import download

    source = VISDRONE_ZIP_DIR / f"VisDrone2019-DET-{spec['split']}"
    if not source.exists():
        download([f"{ASSETS_URL}/VisDrone2019-DET-{spec['split']}.zip"], dir=VISDRONE_ZIP_DIR, unzip=True, delete=False)

    images_out = target / spec["split"] / "images"
    labels_out = target / spec["split"] / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    annotations = sorted((source / "annotations").glob("*.txt"))
    for annotation in annotations:
        image_path = source / "images" / f"{annotation.stem}.jpg"
        with Image.open(image_path) as image:
            width, height = image.size
        rows = annotation.read_text(encoding="utf-8").splitlines()
        (labels_out / annotation.name).write_text("\n".join(visdrone_to_yolo_lines(rows, width, height)))
        shutil.copy2(image_path, images_out / image_path.name)
    names = {"names": VISDRONE_NAMES, "nc": len(VISDRONE_NAMES)}
    (target / "data.yaml").write_text(yaml.safe_dump(names, sort_keys=False), encoding="utf-8")
    print(f"  đã chuyển {len(annotations)} ảnh VisDrone sang định dạng YOLO")


DOWNLOADERS = {"roboflow": download_roboflow, "visdrone": download_visdrone}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/datasets.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    for source in config["sources"]:
        target = Path(source["path"])
        if (target / "data.yaml").exists():
            print(f"{source['name']:12s} đã có ở {target}")
            continue
        spec = source.get("download")
        if spec is None:
            print(f"{source['name']:12s} không có mục download (dữ liệu tự tạo) — bỏ qua")
            continue
        print(f"{source['name']:12s} đang tải ({spec['type']}) ...")
        DOWNLOADERS[spec["type"]](spec, target)


if __name__ == "__main__":
    main()
