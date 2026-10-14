"""Chuẩn bị gán nhãn tracking bằng CVAT: cắt clip, chạy detector + tracker, ghi nhãn sơ bộ "CVAT for video 1.1".
Kết quả trong --out: clip.mp4 (tạo task CVAT từ file này) + prelabel_cvat.xml (Actions -> Upload annotations)
+ labels.txt (danh sách nhãn cần khai báo trong CVAT) + run_config.json.

Ví dụ:
    python scripts/prelabel_tracking.py --source data/videos/pvd_goc_a.mp4 --start 20 --duration 20 \
        --tracker configs/botsort.yaml --out data/tracking_gt/pvd_goc_a
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import supervision as sv
from ultralytics import YOLO

from vtrack import CLASS_NAMES, IGNORE_LABEL
from vtrack.cvat_format import tracks_to_cvat_xml
from vtrack.models import DEFAULT_WEIGHTS, FALLBACK_IMGSZ, training_imgsz
from vtrack.mot_format import MotBox
from vtrack.tracking import track_video
from vtrack.video_io import cut_clip

# Track sống ngắn hơn mức này (0.5 s ở 30 fps) gần như toàn là box chập chờn; bỏ khỏi nhãn sơ bộ để người gán
# không phải xoá từng cái — xe thật bị bỏ thì người gán vẽ thêm
MIN_PRELABEL_TRACK_FRAMES = 15
KEYFRAME_EVERY = 10


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True)
    parser.add_argument("--start", type=float, required=True, help="Giây bắt đầu trong video nguồn")
    parser.add_argument("--duration", type=float, required=True, help="Độ dài clip (giây)")
    parser.add_argument("--tracker", required=True, help="Cấu hình tracker đã chọn trên tập dev")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out)
    if (out_dir / "gt").exists():
        raise SystemExit(f"{out_dir / 'gt'} đã có nhãn người gán — không ghi đè")
    fps = sv.VideoInfo.from_video_path(args.source).fps
    num_frames = round(args.duration * fps)
    clip = cut_clip(args.source, out_dir / "clip.mp4", args.start, num_frames)
    print(f"Đã cắt {clip} ({num_frames} frame, {fps:g} fps)")

    model = YOLO(args.weights, task="detect")
    imgsz = training_imgsz(model) or FALLBACK_IMGSZ
    boxes: list[MotBox] = []
    track_labels: dict[int, str] = {}
    lengths: dict[int, int] = defaultdict(int)
    for tracked in track_video(clip, model, args.tracker, imgsz):
        for (x1, y1, x2, y2), track_id, class_id in zip(tracked.detections.xyxy, tracked.track_ids, tracked.class_ids):
            boxes.append(MotBox(tracked.index + 1, int(track_id), float(x1), float(y1), float(x2 - x1), float(y2 - y1)))
            track_labels[int(track_id)] = CLASS_NAMES[class_id]  # lớp bỏ phiếu mới nhất = phiếu tích luỹ cả track
            lengths[int(track_id)] += 1

    kept = [box for box in boxes if lengths[box.track_id] >= MIN_PRELABEL_TRACK_FRAMES]
    xml = tracks_to_cvat_xml(kept, track_labels, KEYFRAME_EVERY, num_frames)
    (out_dir / "prelabel_cvat.xml").write_text(xml, encoding="utf-8")
    (out_dir / "labels.txt").write_text("\n".join([*CLASS_NAMES, IGNORE_LABEL]) + "\n", encoding="utf-8")
    n_kept = len({box.track_id for box in kept})
    print(f"Nhãn sơ bộ: {n_kept}/{len(lengths)} track (bỏ track < {MIN_PRELABEL_TRACK_FRAMES} frame), "
          f"keyframe mỗi {KEYFRAME_EVERY} frame -> {out_dir / 'prelabel_cvat.xml'}")
    run_config = {**vars(args), "num_frames": num_frames, "fps": fps, "imgsz": imgsz,
                  "min_prelabel_track_frames": MIN_PRELABEL_TRACK_FRAMES, "keyframe_every": KEYFRAME_EVERY}
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
