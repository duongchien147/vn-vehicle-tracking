"""Chấm mAP kiểu COCO cho nhiều mô hình trên tập chấm phụ lấy từ val UVH-26 (camera giao thông Ấn Độ), đủ cả 4 lớp.
Bỏ ảnh val chụp từ cùng camera với ảnh train v5 đã học (UVH chia train/val theo ảnh), xe ba bánh/xe đạp/Others là
vùng ignore. Cấu hình: mục `evaluation` trong configs/uvh26.yaml. Kết quả: outputs/eval_uvh26/<--name>/.

Ví dụ:
    python scripts/eval_uvh26.py --show-distances          # xem phân bố khoảng cách để chọn ngưỡng camera
    python scripts/eval_uvh26.py --models weights/yolo26n.pt@1280 weights/yolo26n_vn_v3_best.pt weights/yolo26n_vn_v5_best.pt
"""

import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np
import yaml
from huggingface_hub import hf_hub_download

from vtrack.coco_eval import EvalImage, instance_counts, well_represented_classes
from vtrack.coco_source import CocoImage, load_coco, to_eval_image
from vtrack.dataset import dhash_bits, nearest_hamming_distances
from vtrack.detection_benchmark import EVAL_CONF, ModelSpec, describe, score_model

REPO_ID = "iisc-aim/UVH-26"
DEFAULT_MODELS = ["weights/yolo26n.pt@1280", "weights/yolo26n_vn_v3_best.pt", "weights/yolo26n_vn_v5_best.pt"]
MIN_INSTANCES_FOR_MAIN_MAP = 30  # cùng ngưỡng với test Hà Nội
DISTANCE_BIN = 8  # độ rộng cột khi in phân bố khoảng cách Hamming (trên 256 bit)


def sample_val_images(config: dict) -> list[CocoImage]:
    """Lấy ngẫu nhiên (có seed) `n_images` ảnh val; tải về raw_dir nếu chưa có."""
    evaluation = config["evaluation"]
    images = sorted(load_coco(evaluation["annotations"]), key=lambda im: im.file_name)
    chosen = random.Random(evaluation["seed"]).sample(images, evaluation["n_images"])
    repo_paths = {p.rsplit("/", 1)[1]: p for p in Path(evaluation["file_list"]).read_text().split()}
    for image in chosen:
        hf_hub_download(REPO_ID, repo_paths[image.file_name], repo_type="dataset", local_dir=config["raw_dir"])
    return chosen


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise SystemExit(f"Không đọc được ảnh (file hỏng?): {path}")
    return image


def local_path(config: dict, image: CocoImage) -> Path:
    return next(Path(config["raw_dir"]).glob(f"UVH-26-Val/data/*/{image.file_name}"))


def camera_distances(config: dict, images: list[CocoImage]) -> np.ndarray:
    """Khoảng cách Hamming nhỏ nhất từ mỗi ảnh val tới các ảnh train UVH mà v5 đã học."""
    train_images = sorted((Path(config["output"]) / "train" / "images").glob("*.jpg"))
    reference = np.stack([dhash_bits(read_image(p)) for p in train_images])
    query = np.stack([dhash_bits(read_image(local_path(config, im))) for im in images])
    return nearest_hamming_distances(query, reference)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/uvh26.yaml")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="trọng số[@imgsz]")
    parser.add_argument("--bootstrap", type=int, default=20)
    parser.add_argument("--show-distances", action="store_true", help="Chỉ in phân bố khoảng cách camera rồi dừng")
    parser.add_argument("--name", default="v3_vs_v5")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    evaluation = config["evaluation"]

    images = sample_val_images(config)
    distances = camera_distances(config, images)
    out_dir = Path("outputs/eval_uvh26") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "camera_distances.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([("file_name", "nearest_train_distance"),
                                 *((im.file_name, int(d)) for im, d in zip(images, distances))])
    if args.show_distances or evaluation["max_camera_distance"] is None:
        histogram = np.bincount(distances // DISTANCE_BIN)
        for index, count in enumerate(histogram):
            print(f"  khoảng cách {index * DISTANCE_BIN:3d}–{(index + 1) * DISTANCE_BIN - 1:3d}: {count:4d} ảnh")
        print("Chọn ngưỡng ở chỗ trũng, ghi vào evaluation.max_camera_distance trong config rồi chạy lại.")
        return

    kept = [im for im, d in zip(images, distances) if d > evaluation["max_camera_distance"]]
    eval_images = [to_eval_image(im, config["class_map"], set(evaluation["ignore_classes"])) for im in kept]
    counts = instance_counts(eval_images)
    main_classes = well_represented_classes(eval_images, MIN_INSTANCES_FOR_MAIN_MAP)
    print(f"{len(kept)}/{len(images)} ảnh khác camera với train (ngưỡng {evaluation['max_camera_distance']}), "
          f"số box: {counts}; mAP chính trên {main_classes}")

    rows = []
    for spec in args.models:
        score = score_model(ModelSpec.parse(spec), eval_images,
                            lambda i: read_image(local_path(config, kept[i])), main_classes, args.bootstrap)
        rows.append(score.to_row(counts))
        print(describe(score), flush=True)

    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    run_config = {**vars(args), "evaluation": evaluation, "eval_conf": EVAL_CONF,
                  "n_images_kept": len(kept), "n_images_sampled": len(images)}
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nĐã ghi {out_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
