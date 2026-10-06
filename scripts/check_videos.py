"""Kiểm tra video có dùng được cho đánh giá đếm xe không: độ phân giải, độ dài, và các đoạn camera đứng yên.
Lưu thêm ảnh frame giữa của mỗi video vào outputs/video_check/ để xem góc quay.

Ví dụ:
    python scripts/check_videos.py data/videos/nga_tu_hanoi_a.mp4
    python scripts/check_videos.py data/raw/probe/*.mp4 --min-length 60
"""

import argparse
from pathlib import Path

import cv2

from vtrack.camera_motion import measure_camera_motion, stable_segments

OUTPUT_DIR = Path("outputs/video_check")


def format_time(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes // 60}:{minutes % 60:02d}:{secs:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="+")
    parser.add_argument("--min-length", type=float, default=30.0, help="Độ dài tối thiểu (giây) của một đoạn ổn định")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for video in args.videos:
        capture = cv2.VideoCapture(video)
        if not capture.isOpened():
            print(f"\n== {video}: KHÔNG mở được (sai đường dẫn hoặc file hỏng) — bỏ qua")
            continue
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps else 0.0
        capture.set(cv2.CAP_PROP_POS_FRAMES, capture.get(cv2.CAP_PROP_FRAME_COUNT) // 2)
        ok, middle = capture.read()
        capture.release()
        if ok:
            cv2.imwrite(str(OUTPUT_DIR / f"{Path(video).stem}.jpg"), middle)

        samples = measure_camera_motion(video)
        segments = stable_segments(samples, min_length_s=args.min_length)
        stable_total = sum(end - start for start, end in segments)
        cuts = sum(s.scene_cut for s in samples)
        print(f"\n== {Path(video).name}: {width}x{height} {fps:.0f}fps, dài {format_time(duration)}")
        print(f"   ổn định {stable_total:.0f}s / {duration:.0f}s ({stable_total / max(duration, 1):.0%}), {cuts} lần cắt cảnh/mất khớp")
        for start, end in segments:
            print(f"   đoạn ổn định {format_time(start)} - {format_time(end)} ({end - start:.0f}s)")


if __name__ == "__main__":
    main()
