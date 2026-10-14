"""Chấm tracking (HOTA, MOTA, IDF1, số lần đổi ID...) theo chuẩn MOTChallenge bằng TrackEval, cho mọi tổ hợp
mô hình x tracker trên các sequence có nhãn tracking trong configs/tracking_sequences.yaml.
Kết quả: outputs/eval_tracking/<tên>/{results.csv, run_config.json, gt/, trackers/<tổ hợp>/data/<sequence>.txt}.

Ví dụ:
    python scripts/eval_tracking.py
    python scripts/eval_tracking.py --trackers configs/bytetrack.yaml ocsort.yaml --name bytetrack_vs_ocsort
    # Chỉnh tracker trên tập dev theo lưới tham số (biến thể yaml ghi vào <thư mục kết quả>/tracker_configs/)
    python scripts/eval_tracking.py --grid configs/tracker_tuning.yaml --sequences vntraffic --name tuning_vntraffic
"""

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import supervision as sv
import yaml
from ultralytics import YOLO

from vtrack.models import DEFAULT_WEIGHTS, FALLBACK_IMGSZ, load_model, training_imgsz
from vtrack import IGNORE_LABEL
from vtrack.mot_format import (
    MOT_TARGET_CLASS,
    MotBox,
    drop_boxes_in_regions,
    fill_track_gaps,
    read_cvat_mot,
    read_mot_file,
    resolve_duplicate_ids,
)
from vtrack.tracker_config import expand_grid, write_variant
from vtrack.tracking import (
    DEFAULT_AGNOSTIC_NMS_IOU, DEFAULT_DETECTION_CONF, filter_young_tracks, to_mot_boxes, track_video,
)
from vtrack.tracking_eval import evaluate_trackers, write_ground_truth_sequence, write_tracker_sequence

# ByteTrack và BoT-SORT là 2 tracker so sánh chính trong kế hoạch; OC-SORT thêm làm mốc (chuẩn mạnh về che khuất).
# Bản trong configs/ là bản sao chỉnh được; tên không có thư mục là cấu hình mặc định có sẵn của ultralytics.
DEFAULT_TRACKERS = ["configs/bytetrack.yaml", "configs/botsort.yaml", "ocsort.yaml"]


@dataclass
class PreparedSequence:
    num_frames: int
    ignore_regions: list[MotBox]  # box tracker nằm trong đây bị bỏ trước khi chấm


def read_ground_truth(spec: dict) -> tuple[list[MotBox], list[MotBox]]:
    """(box xe, vùng ignore) theo `format` của sequence: "mot" (mặc định) hoặc "cvat_mot" (CVAT xuất MOT 1.1)."""
    if spec.get("format", "mot") == "cvat_mot":
        label_names = Path(spec["labels"]).read_text(encoding="utf-8").split()
        return read_cvat_mot(spec["ground_truth"], label_names, IGNORE_LABEL)
    return read_mot_file(spec["ground_truth"], first_frame=spec.get("first_frame", 1)), []


def prepare_ground_truth(name: str, spec: dict, gt_root: Path) -> PreparedSequence:
    """Chuyển nhãn của một sequence sang thư mục chuẩn MOTChallenge."""
    vehicles, ignore_regions = read_ground_truth(spec)
    # Như UA-DETRAC: vùng ignore loại cả nhãn chuẩn lẫn box tracker nằm trong nó
    resolved = resolve_duplicate_ids(drop_boxes_in_regions(vehicles, ignore_regions))
    boxes = [box for box in resolved if box.mot_class == MOT_TARGET_CLASS]
    duplicate_boxes = [box for box in resolved if box.mot_class != MOT_TARGET_CLASS]
    gap_boxes = fill_track_gaps(boxes) if spec.get("fill_gaps", False) else []
    num_frames = max(box.frame for box in resolved)
    info = sv.VideoInfo.from_video_path(spec["video"])
    if info.total_frames is not None and info.total_frames < num_frames:
        raise SystemExit(f"{name}: nhãn có {num_frames} frame nhưng video chỉ có {info.total_frames}")
    write_ground_truth_sequence(gt_root, name, resolved + gap_boxes, num_frames, info.fps, info.width, info.height)
    print(f"{name}: {num_frames} frame, {len({b.track_id for b in boxes})} track, {len(boxes)} box nhãn; trung tính:"
          f" {len(gap_boxes)} box lấp chỗ rơi nhãn + {len(duplicate_boxes)} box trùng ID trong cùng frame;"
          f" {len(ignore_regions)} box vùng ignore")
    return PreparedSequence(num_frames, ignore_regions)


def run_tracker(
    video: str, model: YOLO, tracker: str, imgsz: int, conf: float, num_frames: int, agnostic_nms_iou: float | None,
) -> tuple[list[MotBox], float]:
    """Chạy detector + tracker trên `num_frames` frame đầu; trả về box dạng MOT và tốc độ (frame/giây, CPU)."""
    boxes = []
    started = time.perf_counter()
    for tracked in track_video(video, model, tracker, imgsz, conf, max_frames=num_frames,
                               agnostic_nms_iou=agnostic_nms_iou):
        boxes += to_mot_boxes(tracked)
    return boxes, num_frames / (time.perf_counter() - started)


def parse_nms_iou(value: str) -> float | None:
    return None if value.lower() == "none" else float(value)


def make_run_name(base_name: str, min_track_age: int) -> str:
    return base_name if min_track_age == 1 else f"{base_name}-age{min_track_age}"


# Hai cấu hình chênh nhau ít hơn mức này ở chỉ số chọn thì coi như hoà (1 sequence, không có khoảng tin cậy)
SELECTION_TIE = 0.005


def print_selection(rows: list[dict], metric: str) -> None:
    """Chọn cấu hình tracker theo tiêu chí đã chốt trước (chỉ xét kết quả không lọc tuổi track); hoà thì chọn nhanh hơn."""
    candidates = [row for row in rows if row["min_track_age"] == 1]
    best_value = max(row[metric] for row in candidates)
    tied = [row for row in candidates if best_value - row[metric] < SELECTION_TIE]
    chosen = max(tied, key=lambda row: row["fps_cpu"])
    print(f"\nChọn theo {metric} (hoà nếu chênh < {SELECTION_TIE}, khi đó lấy cấu hình nhanh hơn): {chosen['run']} "
          f"— {metric} {chosen[metric]:.3f}, {chosen['fps_cpu']:.1f} frame/s; các cấu hình hoà: {[r['run'] for r in tied]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/tracking_sequences.yaml")
    parser.add_argument("--sequences", nargs="+", help="Mặc định: mọi sequence trong config")
    parser.add_argument("--weights", nargs="+", default=[DEFAULT_WEIGHTS])
    parser.add_argument("--imgsz", type=int, help="Mặc định: imgsz lúc train của từng mô hình")
    parser.add_argument("--trackers", nargs="+", default=DEFAULT_TRACKERS)
    parser.add_argument("--grid", help="File lưới tham số tracker (vd. configs/tracker_tuning.yaml); thay cho --trackers")
    parser.add_argument("--min-track-ages", nargs="+", type=int, default=[1],
                        help="Chấm thêm với bộ lọc TrackAgeFilter (1 = không lọc); --grid tự đặt giá trị này")
    parser.add_argument("--conf", type=float, default=DEFAULT_DETECTION_CONF)
    parser.add_argument("--agnostic-nms-ious", nargs="+", type=parse_nms_iou,
                        default=[DEFAULT_AGNOSTIC_NMS_IOU],
                        help="Ngưỡng IoU của bộ lọc box trùng trước tracker (vtrack.box_filter); 'none' = tắt")
    parser.add_argument("--name", default="trackers", help="Tên thí nghiệm -> outputs/eval_tracking/<tên>")
    args = parser.parse_args()

    all_sequences = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["sequences"]
    sequences = {name: all_sequences[name] for name in (args.sequences or all_sequences)}
    out_dir = Path("outputs/eval_tracking") / args.name
    gt_root, trackers_root = out_dir / "gt", out_dir / "trackers"
    prepared = {name: prepare_ground_truth(name, spec, gt_root) for name, spec in sequences.items()}

    trackers, min_track_ages, selection_metric = args.trackers, args.min_track_ages, None
    if args.grid:
        grid_config = yaml.safe_load(Path(args.grid).read_text(encoding="utf-8"))
        variants = [v for e in grid_config["experiments"] for v in expand_grid(e["base"], e["grid"])]
        trackers = [str(write_variant(v, out_dir / "tracker_configs")) for v in variants]
        min_track_ages = grid_config.get("min_track_ages", [1])
        selection_metric = grid_config.get("selection_metric")

    nms_ious: list[float | None] = args.agnostic_nms_ious
    run_info = {}
    for weights in args.weights:
        model = load_model(weights)
        imgsz = args.imgsz or training_imgsz(model) or FALLBACK_IMGSZ
        for tracker, nms_iou in product(trackers, nms_ious):
            base_name = f"{Path(weights).stem}-{Path(tracker).stem}" + ("" if nms_iou is None else f"-nms{nms_iou}")
            speeds = []
            for name, spec in sequences.items():
                print(f"  chạy {base_name} @{imgsz} trên {name} ...", flush=True)
                boxes, fps = run_tracker(spec["video"], model, tracker, imgsz, args.conf, prepared[name].num_frames,
                                         nms_iou)
                boxes = drop_boxes_in_regions(boxes, prepared[name].ignore_regions)
                speeds.append(fps)
                for min_age in min_track_ages:
                    write_tracker_sequence(trackers_root, make_run_name(base_name, min_age), name,
                                           filter_young_tracks(boxes, min_age))
            for min_age in min_track_ages:
                run_info[make_run_name(base_name, min_age)] = {
                    "weights": weights, "tracker": tracker, "imgsz": imgsz, "min_track_age": min_age,
                    "agnostic_nms_iou": nms_iou, "fps_cpu": sum(speeds) / len(speeds),
                }

    metrics = evaluate_trackers(gt_root, trackers_root, list(sequences), list(run_info), out_dir / "trackeval")
    rows = []
    for run_name, info in run_info.items():
        row = {"run": run_name, **info, **asdict(metrics[run_name])}
        rows.append({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
        m = metrics[run_name]
        print(f"  {run_name:40s} HOTA {m.hota:.3f} (DetA {m.det_a:.3f}, AssA {m.ass_a:.3f}) | MOTA {m.mota:.3f} | "
              f"IDF1 {m.idf1:.3f} (IDR {m.idr:.3f}) | đổi ID {m.id_switches} | {info['fps_cpu']:.1f} frame/s")

    if selection_metric:
        print_selection(rows, selection_metric)

    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    run_config = {**vars(args), "sequences": sequences, "num_frames": {name: seq.num_frames for name, seq in prepared.items()}}
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nĐã ghi {out_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
