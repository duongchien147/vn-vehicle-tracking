"""Pipeline chính: phát hiện -> theo dõi -> bỏ phiếu lớp theo track -> đếm qua vạch
-> video có annotate (H.264) + danh sách sự kiện đếm. Dùng chung cho CLI và Streamlit."""

import csv
import math
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from vtrack import CLASS_NAMES
from vtrack.count_eval import make_line_zone
from vtrack.models import FALLBACK_IMGSZ, load_model, training_imgsz
from vtrack.tracking import DEFAULT_DETECTION_CONF, DEFAULT_MIN_TRACK_AGE, DEFAULT_TRACKER, track_video
from vtrack.video_io import to_h264


@dataclass
class CrossingEvent:
    frame: int  # chỉ số frame trong video gốc
    time_s: float
    track_id: int
    class_name: str
    direction: str  # "in" hoặc "out"


@dataclass
class VideoResult:
    output_path: Path
    events: list[CrossingEvent]
    frames: int
    processing_fps: float
    imgsz: int  # kích thước ảnh đã dùng khi suy luận (ghi lại để truy ngược kết quả)

    def counts(self) -> dict[str, dict[str, int]]:
        table = {name: {"in": 0, "out": 0} for name in CLASS_NAMES}
        for event in self.events:
            table[event.class_name][event.direction] += 1
        return table

    def cumulative_counts(self, duration_s: float, step_s: float = 1.0) -> list[dict[str, float]]:
        """Tổng số xe đã qua vạch (cả hai chiều) theo lớp, tại mỗi mốc 0, step_s, 2*step_s... — để vẽ biểu đồ."""
        rows = []
        for step in range(math.floor(duration_s / step_s) + 1):
            time_s = step * step_s
            row: dict[str, float] = {"time_s": time_s}
            for name in CLASS_NAMES:
                row[name] = sum(1 for e in self.events if e.class_name == name and e.time_s <= time_s)
            rows.append(row)
        return rows

    def write_events_csv(self, path: str | Path) -> Path:
        path = Path(path)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["frame", "time_s", "track_id", "class_name", "direction"])
            writer.writeheader()
            writer.writerows(asdict(e) for e in self.events)
        return path


def process_video(
    source: str | Path,
    output: str | Path,
    model: YOLO | str,
    tracker: str = DEFAULT_TRACKER,
    line: tuple[float, float, float, float] = (0.0, 0.6, 1.0, 0.6),
    conf: float = DEFAULT_DETECTION_CONF,
    imgsz: int | None = None,
    vid_stride: int = 1,
    min_track_age: int = DEFAULT_MIN_TRACK_AGE,
    max_frames: int | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> VideoResult:
    """`line` là toạ độ tương đối (x1, y1, x2, y2) trong [0, 1] của khung hình.

    `conf` để thấp (0.1) vì ByteTrack/BoT-SORT dùng cả detection điểm thấp để nối tiếp
    track bị che khuất; việc tạo track mới vẫn do `new_track_thresh` trong config tracker quyết định.
    `imgsz=None` (mặc định) = dùng đúng imgsz lúc train/export của mô hình, đọc từ file trọng số.
    `max_frames` giới hạn số frame xử lý (demo trên CPU: chỉ chạy vài chục giây đầu cho nhanh).
    """
    if isinstance(model, str):
        model = load_model(model)
    imgsz = imgsz or training_imgsz(model) or FALLBACK_IMGSZ

    source, output = Path(source), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    info = sv.VideoInfo.from_video_path(str(source))
    w, h = info.width, info.height
    total = math.ceil(info.total_frames / vid_stride) if info.total_frames else 0
    if max_frames is not None:
        total = min(total, max_frames) if total else max_frames

    line_zone = make_line_zone(line, w, h)  # dùng chung với bộ chấm sai số đếm (vtrack.count_eval)
    thickness = sv.calculate_optimal_line_thickness((w, h))
    text_scale = sv.calculate_optimal_text_scale((w, h))
    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness)
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=int(info.fps / vid_stride * 2),
        position=sv.Position.BOTTOM_CENTER,
    )
    line_annotator = sv.LineZoneAnnotator(
        thickness=thickness, text_thickness=thickness, text_scale=text_scale
    )

    events: list[CrossingEvent] = []
    running = Counter[str]()
    raw_path = output.with_name(output.stem + "_raw.mp4")
    frames = 0
    started = time.perf_counter()

    out_info = sv.VideoInfo(width=w, height=h, fps=info.fps / vid_stride)
    with sv.VideoSink(str(raw_path), out_info) as sink:
        for frames, tracked in enumerate(
            track_video(source, model, tracker, imgsz, conf, vid_stride, max_frames, min_track_age), start=1
        ):
            detections, track_ids, class_ids = (
                tracked.detections, tracked.track_ids, tracked.class_ids
            )
            crossed_in, crossed_out = line_zone.trigger(detections)
            frame_index = tracked.index
            for direction, crossed in (("in", crossed_in), ("out", crossed_out)):
                for i in np.flatnonzero(crossed):
                    name = CLASS_NAMES[class_ids[i]]
                    running[name] += 1
                    events.append(
                        CrossingEvent(
                            frame=frame_index,
                            time_s=frame_index / info.fps,
                            track_id=int(track_ids[i]),
                            class_name=name,
                            direction=direction,
                        )
                    )

            annotated = tracked.image.copy()
            if len(detections):
                labels = [
                    f"#{tid} {CLASS_NAMES[c]}" for tid, c in zip(track_ids, class_ids)
                ]
                annotated = trace_annotator.annotate(annotated, detections)
                annotated = box_annotator.annotate(annotated, detections)
                annotated = label_annotator.annotate(annotated, detections, labels)
            annotated = line_annotator.annotate(annotated, line_counter=line_zone)
            draw_count_panel(annotated, running, text_scale, thickness)
            sink.write_frame(annotated)

            if on_progress is not None:
                on_progress(frames, total)

    elapsed = time.perf_counter() - started
    to_h264(raw_path, output)
    raw_path.unlink()
    return VideoResult(
        output_path=output,
        events=events,
        frames=frames,
        processing_fps=frames / elapsed if elapsed > 0 else 0.0,
        imgsz=imgsz,
    )


def draw_count_panel(
    frame: np.ndarray, counts: Counter[str], text_scale: float, thickness: int
) -> None:
    """Bảng đếm tích luỹ theo lớp ở góc trên bên trái."""
    line_height = int(30 * text_scale) + 10
    width = int(220 * text_scale) + 20
    cv2.rectangle(
        frame, (0, 0), (width, line_height * len(CLASS_NAMES) + 10), (0, 0, 0), -1
    )
    for row, name in enumerate(CLASS_NAMES):
        cv2.putText(
            frame,
            f"{name}: {counts[name]}",
            (10, line_height * (row + 1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (255, 255, 255),
            max(1, thickness // 2),
            cv2.LINE_AA,
        )
