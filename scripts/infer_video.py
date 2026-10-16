"""Chạy pipeline trên một video: xuất video có annotate (H.264), CSV các sự kiện xe qua vạch
và file cấu hình đã dùng (để truy ngược kết quả).

Ví dụ:
    python scripts/infer_video.py --source data/videos/nga_tu_hanoi_a.mp4
    python scripts/infer_video.py --source data/videos/nga_tu_hanoi_a.mp4 --weights weights/yolo26n_vn_v2_best_openvino_model \
        --tracker configs/botsort.yaml --line 0 0.55 1 0.55 --vid-stride 2
"""

import os

# Luồng OpenMP của torch mặc định chờ quay vòng sau mỗi lần dùng và tranh lõi CPU với OpenVINO (v5 INT8 trên CPU
# 34 -> 90 ms/frame, đo 10/10/2026); phải đặt trước khi torch được import lần đầu. Đổi lại PyTorch trên CPU chậm ~15%.
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

# Các import dưới đây phải chạy sau khi đặt biến môi trường ở trên
import argparse
import json
from pathlib import Path

from vtrack.box_filter import parse_nms_iou
from vtrack.models import DEFAULT_WEIGHTS, device_name
from vtrack.pipeline import process_video
from vtrack.tracking import DEFAULT_AGNOSTIC_NMS_IOU, DEFAULT_MIN_TRACK_AGE, DEFAULT_TRACKER


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="Đường dẫn video đầu vào")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS,
                        help=".pt hoặc thư mục *_openvino_model (thêm #cpu / #gpu để chọn thiết bị)")
    parser.add_argument("--tracker", default=DEFAULT_TRACKER)
    parser.add_argument("--agnostic-nms-iou", type=parse_nms_iou, default=DEFAULT_AGNOSTIC_NMS_IOU,
                        help="Ngưỡng IoU của bộ lọc box trùng trước tracker (vtrack.box_filter), none = tắt; mặc định như pipeline")
    parser.add_argument("--line", type=float, nargs=4, default=[0.0, 0.6, 1.0, 0.6],
                        metavar=("X1", "Y1", "X2", "Y2"), help="Vạch đếm, toạ độ tương đối [0, 1]")
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--imgsz", type=int, default=None, help="Mặc định: đúng imgsz lúc train/export của mô hình")
    parser.add_argument("--vid-stride", type=int, default=1, help="Xử lý 1 trên mỗi N frame")
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()

    source = Path(args.source)
    out_dir = Path(args.out_dir)
    output = out_dir / f"{source.stem}_tracked.mp4"

    def show_progress(done: int, total: int) -> None:
        if total and (done % 50 == 0 or done == total):
            print(f"\r  {done}/{total} frame", end="", flush=True)

    result = process_video(
        source=source,
        output=output,
        model=args.weights,
        tracker=args.tracker,
        line=tuple(args.line),
        conf=args.conf,
        imgsz=args.imgsz,
        vid_stride=args.vid_stride,
        agnostic_nms_iou=args.agnostic_nms_iou,
        on_progress=show_progress,
    )
    print()

    csv_path = result.write_events_csv(out_dir / f"{source.stem}_events.csv")

    # imgsz thực tế đã dùng (kể cả khi đọc tự động), bộ lọc tuổi track mặc định của pipeline, và số frame/tốc độ để
    # web demo nạp lại được kết quả (vtrack.saved_runs) mà không phải xử lý lại
    run_config = {**vars(args), "imgsz": result.imgsz, "min_track_age": DEFAULT_MIN_TRACK_AGE,
                  "frames": result.frames, "processing_fps": round(result.processing_fps, 2),
                  "device": device_name(args.weights)}
    config_path = out_dir / f"{source.stem}_run_config.json"
    config_path.write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Video:    {result.output_path}")
    print(f"CSV:      {csv_path}")
    print(f"Cấu hình: {config_path}")
    print(f"{result.frames} frame, imgsz={result.imgsz}, {result.processing_fps:.1f} FPS xử lý")
    for name, c in result.counts().items():
        print(f"  {name:10s} in={c['in']:4d}  out={c['out']:4d}")


if __name__ == "__main__":
    main()
