"""Phân tích clip CHƯA có nhãn (vd. cảnh đêm): chạy detector + tracker, tính chỉ số thay thế không cần nhãn
(xe/frame, độ tin cậy trung vị, số track, độ dài track, tỉ lệ track ngắn) và lưu ảnh frame có box để xem bằng mắt.
So tương đối với một clip đã biết kết quả thật (clip test PVD ban ngày). KHÔNG phải độ chính xác.
Kết quả: outputs/unlabeled_analysis/<--name>/{stats.csv, run_config.json, frames/}.

Ví dụ:
    python scripts/analyze_unlabeled_clips.py --name night
"""

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import supervision as sv

from vtrack import CLASS_NAMES
from vtrack.detection_benchmark import ModelSpec
from vtrack.models import FALLBACK_IMGSZ, load_model, training_imgsz
from vtrack.track_stats import track_statistics
from vtrack.tracking import DEFAULT_DETECTION_CONF, DEFAULT_TRACKER, TrackedFrame, to_mot_boxes, track_video

DEFAULT_CLIPS = [
    "data/tracking_gt/pvd_goc_a/clip.mp4",  # mốc ban ngày: đã biết kết quả thật (HOTA 0.718 với v5)
    "data/videos/ham_chui_dem_1.mp4",
    "data/videos/ham_chui_dem_2.mp4",
    "data/videos/ham_chui_dem_3.mp4",
]
DEFAULT_MODELS = ["weights/yolo26n_vn_v5_best.pt", "weights/yolo26n_vn_v3_best.pt", "weights/yolo26n.pt@1280"]
SHORT_TRACK_SECONDS = 1.0  # track sống dưới 1 giây: thường là box chập chờn hoặc mảnh của track bị đứt
BOX_COLORS = [(0, 255, 0), (255, 200, 0), (0, 0, 255), (255, 0, 255)]  # BGR theo thứ tự CLASS_NAMES


def save_snapshot(tracked: TrackedFrame, path: Path) -> None:
    """Frame có box + "#id lớp điểm" để xem bằng mắt (màu theo lớp)."""
    image = tracked.image.copy()
    detections = tracked.detections
    if detections.confidence is not None:
        for (x1, y1, x2, y2), track_id, class_id, score in zip(
            detections.xyxy.astype(int), tracked.track_ids, tracked.class_ids, detections.confidence
        ):
            color = BOX_COLORS[class_id]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"#{track_id} {CLASS_NAMES[class_id]} {score:.2f}", (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips", nargs="+", default=DEFAULT_CLIPS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="trọng số[@imgsz]")
    parser.add_argument("--seconds", type=float, default=20.0, help="Số giây đầu của mỗi clip được xử lý")
    parser.add_argument("--snapshot-every-s", type=float, default=5.0, help="Lưu ảnh mỗi bấy nhiêu giây (mô hình đầu)")
    parser.add_argument("--tracker", default=DEFAULT_TRACKER)
    parser.add_argument("--name", default="night")
    args = parser.parse_args()

    out_dir = Path("outputs/unlabeled_analysis") / args.name
    rows = []
    for model_index, spec_text in enumerate(args.models):
        spec = ModelSpec.parse(spec_text)
        model = load_model(spec.weights)
        imgsz = spec.imgsz or training_imgsz(model) or FALLBACK_IMGSZ
        for clip in args.clips:
            fps = sv.VideoInfo.from_video_path(clip).fps
            num_frames = round(args.seconds * fps)
            snapshot_step = round(args.snapshot_every_s * fps)
            boxes = []
            for tracked in track_video(clip, model, args.tracker, imgsz, DEFAULT_DETECTION_CONF, max_frames=num_frames):
                boxes += to_mot_boxes(tracked)
                if model_index == 0 and tracked.index % snapshot_step == 0:
                    save_snapshot(tracked, out_dir / "frames" / f"{Path(clip).stem}_{tracked.index:05d}.jpg")
            stats = track_statistics(boxes, num_frames, short_frames=round(SHORT_TRACK_SECONDS * fps))
            row = {"model": spec_text, "clip": Path(clip).stem, "imgsz": imgsz,
                   **{k: round(v, 3) if isinstance(v, float) else v for k, v in asdict(stats).items()}}
            rows.append(row)
            print(f"  {Path(spec.weights).stem:20s} {Path(clip).stem:16s} {stats.boxes_per_frame:5.1f} xe/frame | "
                  f"tin cậy trung vị {stats.median_confidence:.2f} | {stats.tracks:4d} track, dài trung vị "
                  f"{stats.median_track_frames:5.0f} frame, {stats.short_track_fraction:.0%} track < {SHORT_TRACK_SECONDS:g} s",
                  flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "stats.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nĐã ghi {out_dir / 'stats.csv'} và ảnh trong {out_dir / 'frames'}")


if __name__ == "__main__":
    main()
