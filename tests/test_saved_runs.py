import json
from pathlib import Path

from vtrack.pipeline import CrossingEvent, VideoResult
from vtrack.saved_runs import find_saved_runs, load_saved_run

EVENT = CrossingEvent(frame=9, time_s=0.3, track_id=22, display_id=12, class_name="motorbike", direction="out")


def save_run(directory: Path, clip: str) -> None:
    """Ghi 3 file đúng như scripts/infer_video.py."""
    (directory / f"{clip}_tracked.mp4").write_bytes(b"")
    result = VideoResult(output_path=Path("x.mp4"), events=[EVENT], frames=1800, processing_fps=20.9, imgsz=960)
    result.write_events_csv(directory / f"{clip}_events.csv")
    config = {"weights": "weights/yolo26s_vn_v5_best.pt", "imgsz": 960, "frames": 1800, "processing_fps": 20.9}
    (directory / f"{clip}_run_config.json").write_text(json.dumps(config), encoding="utf-8")


def test_load_saved_run_restores_events_and_counts(tmp_path):
    save_run(tmp_path, "pvd_goc_b")

    run = load_saved_run(tmp_path, "pvd_goc_b")

    assert run.result.events == [EVENT]
    assert run.result.counts()["motorbike"] == {"in": 0, "out": 1}
    assert (run.result.frames, run.result.processing_fps, run.result.imgsz) == (1800, 20.9, 960)
    assert run.result.output_path == tmp_path / "pvd_goc_b_tracked.mp4"


def test_find_saved_runs_skips_clips_with_missing_files(tmp_path):
    save_run(tmp_path, "pvd_goc_b")
    save_run(tmp_path, "ham_chui_dem_1")
    (tmp_path / "pvd_goc_a_tracked.mp4").write_bytes(b"")  # chưa có CSV/cấu hình -> kết quả dở dang

    assert find_saved_runs(tmp_path) == ["ham_chui_dem_1", "pvd_goc_b"]
