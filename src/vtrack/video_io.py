"""Đọc/ghi video. OpenCV ghi mp4 bằng codec mp4v mà trình duyệt không phát được,
nên video kết quả được re-encode sang H.264 bằng ffmpeg đi kèm imageio-ffmpeg."""

import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np


def to_h264(src: str | Path, dst: str | Path) -> Path:
    dst = Path(dst)
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-loglevel", "error",
        "-i", str(src),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        str(dst),
    ]
    subprocess.run(cmd, check=True)
    return dst


# Chất lượng gần như không mất (CRF 18 là mức "trông như bản gốc" thường dùng của x264)
CLIP_CRF = 18


def cut_clip(src: str | Path, dst: str | Path, start_s: float, num_frames: int) -> Path:
    """Cắt `num_frames` frame bắt đầu từ giây `start_s` ra file H.264 mới.

    Dùng cho clip gán nhãn tracking: CVAT và bộ chấm đều đọc đúng file đã cắt này, nên frame thứ i
    trong nhãn khớp chính xác frame thứ i khi chạy tracker (không lệch do tua video gốc).
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-nostdin", "-loglevel", "error",
        "-ss", f"{start_s}", "-i", str(src),
        "-frames:v", str(num_frames),
        "-c:v", "libx264", "-crf", str(CLIP_CRF), "-pix_fmt", "yuv420p", "-an",
        str(dst),
    ]
    subprocess.run(cmd, check=True)
    return dst


def read_frame(path: str | Path, index: int = 0) -> np.ndarray:
    """Một frame (BGR) của video — dùng để xem trước vị trí vạch đếm trước khi chạy cả video."""
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"Không đọc được frame {index} của {path}")
    return frame


def safe_upload_name(name: str, allowed_suffixes: tuple[str, ...]) -> str | None:
    """Tên file an toàn để lưu video người dùng tải lên, hoặc None nếu không hợp lệ.

    Tên do trình duyệt gửi lên có thể chứa thư mục ("../../x.mp4", dấu gạch ngược kiểu Windows, "C:/x.mp4") nhằm ghi file ra ngoài
    thư mục lưu (path traversal) -> chỉ giữ phần tên cuối, cấm tên ẩn/rỗng và đuôi ngoài danh sách cho phép.
    """
    base = Path(name.replace("\\", "/")).name
    if not base or base.startswith(".") or Path(base).suffix.lower().lstrip(".") not in allowed_suffixes:
        return None
    return base
