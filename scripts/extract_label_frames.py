"""Trích frame từ video theo configs/label_frames.yaml vào <output>/<split>/images/ để gán nhãn.

Ví dụ:
    python scripts/extract_label_frames.py
"""

import argparse
import shutil
from pathlib import Path

import cv2
import yaml

from vtrack.frame_sampling import parse_timestamp, sample_timestamps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/label_frames.yaml")
    parser.add_argument("--splits", nargs="+", help="Chỉ trích các tập này (mặc định: tất cả) — tránh ghi lại tập đã gán nhãn")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    output = Path(config["output"])
    capture = cv2.VideoCapture(config["video"])
    if not capture.isOpened():
        raise SystemExit(f"Không mở được video: {config['video']}")

    counts: dict[str, int] = {}
    for segment in config["segments"]:
        split = segment["split"]
        if args.splits and split not in args.splits:
            continue
        images_dir = output / split / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        times = sample_timestamps(parse_timestamp(segment["start"]), parse_timestamp(segment["end"]), segment["every_s"])
        for time_s in times:
            capture.set(cv2.CAP_PROP_POS_MSEC, time_s * 1000)
            ok, frame = capture.read()
            if not ok:
                raise SystemExit(f"Không đọc được frame tại {time_s:.1f}s")
            minutes, seconds = divmod(int(time_s), 60)
            # Tên file chứa thời điểm trong video gốc, để truy ngược frame về đoạn nào
            name = f"{config['prefix']}_{minutes:02d}m{seconds:02d}s.jpg"
            cv2.imwrite(str(images_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        counts[split] = counts.get(split, 0) + len(times)
    capture.release()

    shutil.copy2(args.config, output / "label_frames_config.yaml")  # lưu lại cấu hình đã dùng
    for split, count in counts.items():
        print(f"{split:5s} {count:4d} frame -> {output / split / 'images'}")


if __name__ == "__main__":
    main()
