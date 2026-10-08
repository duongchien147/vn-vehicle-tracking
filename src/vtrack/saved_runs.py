"""Nạp lại kết quả scripts/infer_video.py đã lưu (<clip>_tracked.mp4 + _events.csv + _run_config.json) thành
VideoResult — để web demo trình chiếu video xử lý sẵn trên GPU (notebooks/kaggle_demo_gpu) với đúng bảng đếm và biểu
đồ như khi chạy trực tiếp, không phải xử lý lại trên CPU."""

import json
from dataclasses import dataclass
from pathlib import Path

from vtrack.pipeline import VideoResult, read_events_csv

VIDEO_SUFFIX = "_tracked.mp4"
EVENTS_SUFFIX = "_events.csv"
CONFIG_SUFFIX = "_run_config.json"


@dataclass
class SavedRun:
    clip: str
    result: VideoResult
    config: dict  # cấu hình đã dùng khi xử lý, đọc nguyên từ _run_config.json
    events_csv: Path


def find_saved_runs(directory: str | Path) -> list[str]:
    """Tên các clip có đủ cả 3 file kết quả (thiếu file nào thì bỏ qua, tránh hiển thị kết quả dở dang)."""
    directory = Path(directory)
    clips = [p.name.removesuffix(VIDEO_SUFFIX) for p in directory.glob(f"*{VIDEO_SUFFIX}")]
    return sorted(c for c in clips
                  if (directory / f"{c}{EVENTS_SUFFIX}").exists() and (directory / f"{c}{CONFIG_SUFFIX}").exists())


def load_saved_run(directory: str | Path, clip: str) -> SavedRun:
    directory = Path(directory)
    config = json.loads((directory / f"{clip}{CONFIG_SUFFIX}").read_text(encoding="utf-8"))
    events_csv = directory / f"{clip}{EVENTS_SUFFIX}"
    result = VideoResult(
        output_path=directory / f"{clip}{VIDEO_SUFFIX}",
        events=read_events_csv(events_csv),
        frames=config["frames"],
        processing_fps=config["processing_fps"],
        imgsz=config["imgsz"],
    )
    return SavedRun(clip=clip, result=result, config=config, events_csv=events_csv)
