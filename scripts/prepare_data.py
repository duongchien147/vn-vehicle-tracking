"""Gộp các dataset nguồn (configs/datasets.yaml) thành một dataset YOLO 4 lớp, loại rò rỉ ảnh gần trùng
giữa các tập, rồi ghi data.yaml và bảng thống kê stats.csv.

Ví dụ:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --config configs/datasets.yaml
"""

import argparse
import csv
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import yaml

from vtrack import CLASS_NAMES
from vtrack.dataset import (
    dhash,
    group_near_duplicates,
    has_min_boxes,
    remap_label_lines,
    sample_items,
    split_by_group,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
JPEG_QUALITY = 92  # đủ cao để không thấy mất chi tiết khi thu nhỏ ảnh lớn


@dataclass
class Record:
    source: str
    image_path: Path
    labels: list[str]  # dòng nhãn YOLO đã đổi sang lớp dự án
    split: str = ""  # train / val / test (gán sau khi chia)
    copy_index: int = 0  # > 0: bản lặp của cùng một ảnh (oversampling)

    @property
    def stem(self) -> str:
        # tiền tố nguồn tránh trùng tên file giữa các dataset; hậu tố _rN phân biệt các bản lặp
        suffix = f"_r{self.copy_index}" if self.copy_index else ""
        return f"{self.source}_{self.image_path.stem}{suffix}"


def source_class_names(root: Path) -> list[str]:
    """Tên lớp của dataset nguồn: từ data.yaml (Roboflow, VisDrone) hoặc classes.txt (nhãn tự gán, X-AnyLabeling)."""
    if (root / "data.yaml").exists():
        return yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))["names"]
    return (root / "classes.txt").read_text(encoding="utf-8").split()


def collect_images(source: dict, split_dir: str = "*") -> list[tuple[Path, list[str]]]:
    """Ảnh của một dataset nguồn kèm nhãn đã đổi sang lớp dự án. split_dir="*" = mọi tập (bỏ qua cách chia gốc)."""
    root = Path(source["path"])
    source_names = source_class_names(root)
    class_map = {str(k): v for k, v in source["class_map"].items()}
    items = []
    for image_path in sorted(root.glob(f"{split_dir}/images/*")):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
        lines = label_path.read_text().splitlines() if label_path.exists() else []
        items.append((image_path, remap_label_lines(lines, source_names, class_map)))
    return items


def predefined_records(source: dict) -> list[Record]:
    """Nguồn đã chia sẵn (frame Hà Nội chia theo đoạn thời gian): giữ nguyên cách chia, lặp ảnh theo `repeat`."""
    records = []
    for target_split, source_split in source["splits"].items():
        repeat = source.get("repeat", {}).get(target_split, 1)
        for image_path, labels in collect_images(source, source_split):
            records += [Record(source["name"], image_path, labels, target_split, copy) for copy in range(repeat)]
        print(f"{source['name']:12s} {source_split:>5s} -> {target_split:5s} x{repeat}")
    return records


def image_hash(path: Path) -> int:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Không đọc được ảnh (file hỏng?): {path}")
    return dhash(image)


def write_image(source: Path, target: Path, max_side: int | None) -> None:
    """Chép ảnh sang thư mục đích; ảnh có cạnh dài hơn `max_side` được thu nhỏ (nhãn YOLO chuẩn hoá nên không đổi)."""
    image = cv2.imread(str(source))
    if image is None:
        raise ValueError(f"Không đọc được ảnh (file hỏng?): {source}")
    longest = max(image.shape[:2])
    if max_side is None or longest <= max_side:
        shutil.copy2(source, target)
        return
    scale = max_side / longest
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(target.with_suffix(".jpg")), resized, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/datasets.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    # 1. Gom ảnh và nhãn từ mọi nguồn
    records: list[Record] = []  # đã biết tập (train-only hoặc chia sẵn)
    to_split: list[Record] = []  # nguồn nhiều cảnh, sẽ được chia train/val/test theo nhóm ảnh gần trùng
    for source in config["sources"]:
        if source["use_for"] == "predefined":
            records += predefined_records(source)
            continue
        all_items = collect_images(source)
        min_boxes = source.get("min_boxes", {})
        eligible = [(path, labels) for path, labels in all_items if has_min_boxes(labels, min_boxes)]
        chosen = [Record(source["name"], path, labels) for path, labels in sample_items(eligible, source.get("max_images"), config["seed"])]
        if source["use_for"] == "train":
            records += [Record(r.source, r.image_path, r.labels, "train") for r in chosen]
        else:
            to_split += chosen
        print(f"{source['name']:12s} {len(chosen):5d} / {len(all_items):5d} ảnh  ({source['use_for']})")

    # 2. Nhóm ảnh gần trùng trong các nguồn được chia, rồi chia theo nhóm để không rò rỉ sang test
    groups = group_near_duplicates([image_hash(r.image_path) for r in to_split], config["near_duplicate_max_distance"])
    print(f"{len(to_split)} ảnh cần chia -> {len(set(groups))} nhóm ảnh gần trùng")
    for record, split in zip(to_split, split_by_group(groups, config["split"], config["seed"])):
        record.split = split
    records += to_split

    # 3. Ghi ra thư mục đích
    output = Path(config["output"])
    if output.exists():
        shutil.rmtree(output)
    stats: Counter[tuple[str, str, str]] = Counter()  # (tập, nguồn, lớp) -> số box
    image_counts: Counter[tuple[str, str]] = Counter()  # (tập, nguồn) -> số ảnh (tính cả bản lặp)
    for record in records:
        for sub in ("images", "labels"):
            (output / record.split / sub).mkdir(parents=True, exist_ok=True)
        image_target = output / record.split / "images" / f"{record.stem}{record.image_path.suffix}"
        write_image(record.image_path, image_target, config.get("max_image_side"))
        (output / record.split / "labels" / f"{record.stem}.txt").write_text("\n".join(record.labels))
        image_counts[(record.split, record.source)] += 1
        for line in record.labels:
            stats[(record.split, record.source, CLASS_NAMES[int(line.split()[0])])] += 1

    # Không ghi khoá `path`: ultralytics sẽ lấy thư mục chứa data.yaml làm gốc,
    # nên cùng một thư mục dataset chạy được cả trên máy lẫn trên Kaggle.
    data_yaml: dict[str, object] = {split: f"{split}/images" for split in config["split"]}
    data_yaml["names"] = dict(enumerate(CLASS_NAMES))
    (output / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")

    with (output / "stats.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "source", "images", *CLASS_NAMES])
        for split, source_name in sorted(image_counts):
            boxes = [stats[(split, source_name, name)] for name in CLASS_NAMES]
            writer.writerow([split, source_name, image_counts[(split, source_name)], *boxes])
    shutil.copy2(args.config, output / "datasets_config.yaml")  # lưu lại cấu hình đã dùng

    print(f"\nĐã ghi {output}")
    for split in config["split"]:
        n_images = sum(v for (s, _), v in image_counts.items() if s == split)
        boxes = {name: sum(v for (s, _, c), v in stats.items() if s == split and c == name) for name in CLASS_NAMES}
        print(f"  {split:5s} {n_images:5d} ảnh  " + "  ".join(f"{k}={v}" for k, v in boxes.items()))


if __name__ == "__main__":
    main()
