"""Gán nhãn tự động frame video bằng nhiều mô hình "thầy" (configs/auto_label*.yaml; YOLO hoặc thư mục RT-DETRv2).
Ghi <split>/<raw-subdir>/ (mọi box sau khi gộp, kèm điểm ở cột 6 — để chọn lại ngưỡng mà không chạy lại) và
<split>/labels/ (box có điểm >= keep_conf). Tập có nhãn người đã kiểm (HUMAN_VERIFIED.txt) chỉ được ghi raw, để đo
chất lượng nhãn tự động — không bao giờ ghi đè labels/ của tập đó.

Ví dụ:
    python scripts/auto_label.py data/labeling/hanoi --splits train
    python scripts/auto_label.py data/labeling/hanoi --splits train --keep-conf 0.15 --reuse-raw      # chỉ đổi ngưỡng
    python scripts/auto_label.py data/labeling/hanoi --splits val --config configs/auto_label_sahi.yaml \
        --raw-subdir labels_auto_raw_sahi                                                              # đo cấu hình mới
"""

import argparse
import shutil
import time
from pathlib import Path

import cv2
import yaml

from vtrack.detectors import load_detector
from vtrack.label_review import is_human_verified, read_scored_yolo_file, write_yolo_file
from vtrack.pseudo_label import Teacher, pseudo_label_image, select_pseudo_labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", help="Thư mục chứa <split>/images/")
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--config", default="configs/auto_label.yaml")
    parser.add_argument("--raw-subdir", default="labels_auto_raw", help="Thư mục con ghi nhãn có điểm")
    parser.add_argument("--keep-conf", type=float, help="Ghi đè keep_conf trong config")
    parser.add_argument("--reuse-raw", action="store_true", help="Không chạy mô hình, chỉ lọc lại nhãn raw theo ngưỡng mới")
    args = parser.parse_args()
    if args.raw_subdir == "labels":
        raise SystemExit("--raw-subdir không được là 'labels'.")

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    keep_conf = args.keep_conf if args.keep_conf is not None else config["keep_conf"]
    specs = [Teacher(**spec) for spec in config["teachers"]]
    teachers = [] if args.reuse_raw else [(load_detector(t.weights, t.imgsz).predict, t) for t in specs]

    for split in args.splits:
        split_dir = Path(args.root) / split
        write_labels = not is_human_verified(split_dir)
        if not write_labels:
            print(f"{split}: nhãn người đã kiểm -> chỉ ghi {args.raw_subdir}/, không đụng labels/")
        raw_dir = split_dir / args.raw_subdir
        raw_dir.mkdir(exist_ok=True)
        (split_dir / "labels").mkdir(exist_ok=True)

        images = sorted((split_dir / "images").glob("*.jpg"))
        started = time.perf_counter()
        kept_total = 0
        for index, image_path in enumerate(images, start=1):
            raw_path = raw_dir / f"{image_path.stem}.txt"
            if args.reuse_raw:
                fused = read_scored_yolo_file(raw_path)
            else:
                image = cv2.imread(str(image_path))
                if image is None:
                    raise SystemExit(f"Không đọc được ảnh: {image_path}")
                fused = pseudo_label_image(
                    image, teachers, config["flip_tta"], config["predict_conf"], config["wbf_iou"], config["skip_box_thr"]
                )
                write_yolo_file(raw_path, fused, with_confidence=True)
            kept = select_pseudo_labels(fused, keep_conf, config["dedup_iou"])
            if write_labels:
                write_yolo_file(split_dir / "labels" / f"{image_path.stem}.txt", kept)
            kept_total += len(kept)
            elapsed = time.perf_counter() - started
            print(f"\r  {split}: {index}/{len(images)} ảnh, {elapsed / index:.1f}s/ảnh", end="", flush=True)
        print(f"\n{split}: {len(images)} ảnh, {kept_total / max(len(images), 1):.0f} box/ảnh ở keep_conf={keep_conf}")
        shutil.copy2(args.config, raw_dir / "auto_label_config.yaml")  # lưu lại cấu hình đã dùng cạnh kết quả
        if write_labels:
            note = f"{keep_conf}  (nguồn: {args.raw_subdir}, {args.config})"
            (split_dir / "labels_keep_conf.txt").write_text(note, encoding="utf-8")


if __name__ == "__main__":
    main()
