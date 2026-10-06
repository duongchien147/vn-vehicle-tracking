"""Dựng tập con UVH-26 cho train v5 theo configs/uvh26.yaml: chọn ảnh (ưu tiên ảnh có buýt/tải), tải PNG gốc từ
Hugging Face, đổi nhãn COCO 14 lớp -> 4 lớp dự án, bổ sung xe thiếu nhãn bằng YOLO26x, lưu JPEG + nhãn YOLO.
Chạy lại được: ảnh đã xử lý (đã có file nhãn) thì bỏ qua. Ra: <output>/train/{images,labels,labels_added}.

Ví dụ:
    python scripts/build_uvh26_subset.py
"""

import argparse
import json
import random
import time
from pathlib import Path

import cv2
import yaml
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from vtrack import CLASS_NAMES
from vtrack.coco_source import CocoImage, load_coco, to_project_classes
from vtrack.label_completion import complete_labels
from vtrack.label_review import YoloBox, write_yolo_file
from vtrack.pseudo_label import predict_teacher

REPO_ID = "iisc-aim/UVH-26"
JPEG_QUALITY = 95


def select_images(images: list[CocoImage], config: dict) -> list[CocoImage]:
    """Lấy ngẫu nhiên (có seed) `images_with_bus_or_truck` ảnh có buýt/tải, rồi `images_random` ảnh trong phần còn lại."""
    rng = random.Random(config["seed"])
    wanted = set(config["bus_or_truck_classes"])
    with_bus_or_truck = [im for im in images if any(name in wanted for name, _ in im.boxes)]
    chosen = rng.sample(with_bus_or_truck, config["images_with_bus_or_truck"])
    chosen_names = {im.file_name for im in chosen}
    rest = [im for im in images if im.file_name not in chosen_names]
    return chosen + rng.sample(rest, config["images_random"])


def save_resized(image, path: Path, max_side: int) -> None:
    height, width = image.shape[:2]
    scale = max_side / max(height, width)
    if scale < 1:
        image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/uvh26.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    completion = config["completion"]

    repo_paths = {p.rsplit("/", 1)[1]: p for p in Path(config["file_list"]).read_text().split()}
    selected = select_images(load_coco(config["annotations"]), config)
    out = Path(config["output"]) / "train"
    for sub in ("images", "labels", "labels_added"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    (Path(config["output"]) / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")

    teacher = YOLO(completion["teacher"], task="detect")
    started, done, n_labels, n_added = time.perf_counter(), 0, 0, 0
    for index, image in enumerate(selected, start=1):
        stem = Path(image.file_name).stem
        label_path = out / "labels" / f"{stem}.txt"
        if not label_path.exists():
            local = hf_hub_download(REPO_ID, repo_paths[image.file_name], repo_type="dataset",
                                    local_dir=config["raw_dir"])
            pixels = cv2.imread(local)
            if pixels is None:
                raise SystemExit(f"Không đọc được ảnh {local}")
            occupied = [YoloBox(-1, b.center_x, b.center_y, b.width, b.height) for _, b in image.boxes]
            result = complete_labels(
                to_project_classes(image, config["class_map"]), occupied,
                predict_teacher(teacher, pixels, completion["imgsz"], completion["min_confidence"]),
                completion["min_confidence"], completion["max_overlap_iou"], completion["max_inside_fraction"],
            )
            save_resized(pixels, out / "images" / f"{stem}.jpg", config["max_image_side"])
            write_yolo_file(out / "labels_added" / f"{stem}.txt", result.added, with_confidence=True)
            write_yolo_file(label_path, result.labels)  # ghi sau cùng: có file nhãn = ảnh đã xử lý xong
            done += 1
            n_labels += len(result.labels)
            n_added += len(result.added)
        if index % 50 == 0 or index == len(selected):
            elapsed = time.perf_counter() - started
            eta = elapsed / max(done, 1) * (len(selected) - index) / 60
            print(f"  {index}/{len(selected)} ảnh, lần này xử lý {done}: {n_labels} nhãn (bổ sung {n_added}), "
                  f"còn ~{eta:.0f} phút", flush=True)

    run_config = {**config, "selected": len(selected), "processed_this_run": done}
    (Path(config["output"]) / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False),
                                                           encoding="utf-8")
    print(f"Xong: {out}")


if __name__ == "__main__":
    main()
