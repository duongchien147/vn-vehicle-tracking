"""Chấm detector trên UA-DETRAC chia theo điều kiện ánh sáng (ngày / đêm) — đo mức suy giảm khi trời tối trên cùng
một bộ dữ liệu, vì dự án không có cảnh đêm Việt Nam nào có nhãn. Ảnh đọc thẳng từ file zip (không giải nén 140k ảnh).
Cấu hình: configs/ua_detrac.yaml. Kết quả: outputs/eval_ua_detrac/{results.csv, run_config.json, sequences.csv}.

Ví dụ:
    python scripts/eval_ua_detrac.py
"""

import argparse
import csv
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from vtrack.coco_eval import instance_counts, well_represented_classes
from vtrack.coco_source import CocoImage, load_coco, to_eval_image
from vtrack.detection_benchmark import EVAL_CONF, ModelSpec, describe, score_model

DEFAULT_MODELS = ["weights/yolo26n.pt@1280", "weights/yolo26n_vn_v3_best.pt", "weights/yolo26n_vn_v5_best.pt"]
MIN_INSTANCES_FOR_MAIN_MAP = 30  # cùng ngưỡng với các tập test khác


def sequence_and_frame(file_name: str) -> tuple[str, int]:
    """'Insight-MVT_Annotation_Train/MVI_63563/img00101.jpg' -> ('MVI_63563', 101)."""
    parts = file_name.split("/")
    return parts[-2], int(parts[-1].removeprefix("img").removesuffix(".jpg"))


class ZipImages:
    """Đọc ảnh UA-DETRAC thẳng từ file zip theo tên video + tên frame."""

    def __init__(self, zip_path: str, prefix: str) -> None:
        self.archive = zipfile.ZipFile(zip_path)
        self.prefix = prefix

    def weather(self, sequence: str) -> str:
        """Thuộc tính điều kiện ánh sáng/thời tiết gốc của video (sence_weather trong file XML nhãn của UA-DETRAC)."""
        xml_name = next(n for n in self.archive.namelist() if n.endswith(f"/{sequence}.xml"))
        header = self.archive.read(xml_name)[:1000].decode("utf-8", "ignore")
        match = re.search(r'sence_weather="(\w+)"', header)
        if match is None:
            raise SystemExit(f"{xml_name}: không có thuộc tính sence_weather")
        return match.group(1)

    def read(self, file_name: str) -> np.ndarray:
        sequence, _ = sequence_and_frame(file_name)
        data = self.archive.read(f"{self.prefix}/{sequence}/{Path(file_name).name}")
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"Không giải mã được ảnh {file_name}")
        return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/ua_detrac.yaml")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="trọng số[@imgsz]")
    parser.add_argument("--bootstrap", type=int, default=20)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path("outputs/eval_ua_detrac")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lấy thưa: frame 1, 1 + step, 1 + 2*step... của mỗi video
    by_sequence: dict[str, list[CocoImage]] = defaultdict(list)
    for image in load_coco(config["annotations"]):
        sequence, frame = sequence_and_frame(image.file_name)
        if (frame - 1) % config["frame_step"] == 0:
            by_sequence[sequence].append(image)
    images = ZipImages(config["images_zip"], config["images_prefix"])

    # Độ sáng trung bình (thang xám) frame đầu mỗi video — chỉ để đối chiếu với nhãn điều kiện gốc
    weather = {seq: images.weather(seq) for seq in sorted(by_sequence)}
    brightness = {seq: float(cv2.cvtColor(images.read(ims[0].file_name), cv2.COLOR_BGR2GRAY).mean())
                  for seq, ims in sorted(by_sequence.items())}
    with (out_dir / "sequences.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([("sequence", "weather", "first_frame_brightness", "sampled_frames"),
                                 *((s, weather[s], round(b, 1), len(by_sequence[s])) for s, b in brightness.items())])
    for condition, values in config["conditions"].items():
        levels = [brightness[s] for s in brightness if weather[s] in values]
        print(f"  {condition} ({'+'.join(values)}): {len(levels)} video, độ sáng frame đầu {min(levels):.0f}–{max(levels):.0f}")

    class_map, ignore_classes = config["class_map"], set(config["ignore_classes"])
    subsets = {}
    for condition, values in config["conditions"].items():
        chosen = [im for seq, ims in by_sequence.items() if weather[seq] in values for im in ims]
        n_sequences = sum(1 for seq in by_sequence if weather[seq] in values)
        subsets[condition] = (chosen, [to_eval_image(im, class_map, ignore_classes) for im in chosen], n_sequences)
    # mAP chính tính trên CÙNG một tập lớp cho cả ngày và đêm (lớp đủ mẫu ở cả hai), nếu không hai số không so được
    main_classes = [name for name in well_represented_classes(subsets["night"][1], MIN_INSTANCES_FOR_MAIN_MAP)
                    if name in well_represented_classes(subsets["day"][1], MIN_INSTANCES_FOR_MAIN_MAP)]

    rows = []
    for condition, (chosen, eval_images, n_sequences) in subsets.items():
        counts = instance_counts(eval_images)
        print(f"== {condition}: {n_sequences} video, {len(chosen)} frame, số box {counts}; mAP chính trên {main_classes}")
        for spec in args.models:
            score = score_model(ModelSpec.parse(spec), eval_images,
                                lambda i, chosen=chosen: images.read(chosen[i].file_name), main_classes, args.bootstrap)
            rows.append({"condition": condition, "n_sequences": n_sequences, "n_frames": len(chosen),
                         "main_classes": "+".join(main_classes), **score.to_row(counts)})
            print(describe(score), flush=True)

    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    run_config = {**vars(args), "config": config, "eval_conf": EVAL_CONF}
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nĐã ghi {out_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
