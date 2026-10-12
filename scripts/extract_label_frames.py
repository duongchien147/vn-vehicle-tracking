"""Trích frame từ video theo configs/label_frames.yaml vào <output>/<split>/images/ để gán nhãn.
Mỗi đoạn có thể có `video` và `prefix` riêng (mặc định lấy ở gốc config), để một config lấy frame từ nhiều clip.
`dedup_min_distance` (tuỳ chọn, ở gốc config): bỏ frame gần trùng với frame đã lấy trong cùng tập — cho video bị ghép
lặp lại (cùng một đoạn xuất hiện nhiều lần), để không gán nhãn và huấn luyện lặp một cảnh.

Ví dụ:
    python scripts/extract_label_frames.py
    python scripts/extract_label_frames.py --config configs/label_frames_extra_test.yaml
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

from vtrack.frame_sampling import frame_signature, keep_distinct, parse_timestamp, sample_timestamps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/label_frames.yaml")
    parser.add_argument("--splits", nargs="+", help="Chỉ trích các tập này (mặc định: tất cả) — tránh ghi lại tập đã gán nhãn")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    output = Path(config["output"])
    counts: dict[str, int] = {}
    written: dict[str, list[tuple[Path, np.ndarray]]] = {}  # tập -> [(file ảnh, chữ ký)] để loại frame gần trùng
    for segment in config["segments"]:
        split = segment["split"]
        if args.splits and split not in args.splits:
            continue
        video = segment.get("video", config.get("video"))
        prefix = segment.get("prefix", config.get("prefix"))
        capture = cv2.VideoCapture(video)
        if not capture.isOpened():
            raise SystemExit(f"Không mở được video: {video}")
        images_dir = output / split / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        times = sample_timestamps(parse_timestamp(segment["start"]), parse_timestamp(segment["end"]), segment["every_s"])
        for time_s in times:
            capture.set(cv2.CAP_PROP_POS_MSEC, time_s * 1000)
            ok, frame = capture.read()
            if not ok:
                raise SystemExit(f"{video}: không đọc được frame tại {time_s:.1f}s")
            minutes, seconds = divmod(int(time_s), 60)
            # Tên file chứa thời điểm trong video gốc, để truy ngược frame về đoạn nào
            name = f"{prefix}_{minutes:02d}m{seconds:02d}s.jpg"
            cv2.imwrite(str(images_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            written.setdefault(split, []).append((images_dir / name, frame_signature(frame)))
        counts[split] = counts.get(split, 0) + len(times)
        capture.release()

    min_distance = config.get("dedup_min_distance")
    if min_distance:
        for split, frames in written.items():
            kept = set(keep_distinct([signature for _, signature in frames], min_distance))
            for index, (image_path, _) in enumerate(frames):
                if index not in kept:
                    image_path.unlink()
            print(f"{split}: bỏ {len(frames) - len(kept)} frame gần trùng (ngưỡng {min_distance})")
            counts[split] = len(kept)

    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, output / "label_frames_config.yaml")  # lưu lại cấu hình đã dùng
    for split, count in counts.items():
        print(f"{split:5s} {count:4d} frame -> {output / split / 'images'}")


if __name__ == "__main__":
    main()
