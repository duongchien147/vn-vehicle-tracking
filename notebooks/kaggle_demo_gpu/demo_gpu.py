"""Xử lý sẵn các clip demo trên GPU T4 của Kaggle bằng ĐÚNG pipeline của dự án (scripts/infer_video.py ->
vtrack.pipeline.process_video): video có vẽ box/ID/vạch đếm (H.264) + CSV sự kiện đếm + cấu hình đã dùng, để trình
chiếu lúc bảo vệ mà không phụ thuộc mạng. Dữ liệu lấy từ dataset private tạo bởi scripts/kaggle_push_benchmark_bundle.py.

Đẩy lên: python scripts/kaggle_train.py push --notebook-dir notebooks/kaggle_demo_gpu
Kết quả: /kaggle/working/demo/<clip>_tracked.mp4, _events.csv, _run_config.json

Mô hình = YOLO26s-v6, mô hình chính của dự án (configs/accept_yolo26s_v6.yaml): trên GPU nó nhanh ngang YOLO26n (cả
pipeline ~42 frame/s, outputs/kaggle/vtrack-gpu-benchmark-run/) mà chính xác hơn, và đã học thêm ảnh đêm. Tracker và bộ
lọc box trùng = mặc định của dự án (vtrack.tracking: botsort_tuned_nogmc, IoU 0.6).
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

# Cùng phiên bản với máy local (requirements.txt)
PACKAGES = ["ultralytics==8.4.163", "supervision==0.30.5", "transformers==5.18.0", "openvino", "lap",
            "pycocotools==2.0.11", "ensemble-boxes==1.0.9", "trackeval==1.3.0", "imageio-ffmpeg"]
WEIGHTS = "weights/yolo26s_vn_v6_best.pt"
TRACKER = "configs/botsort_tuned_nogmc.yaml"
# Clip và vạch đếm: configs/demo_clips.yaml (dùng chung với scripts/make_demo_videos.sh khi chạy trên máy)

# Mặc định chạy tất cả; khi chỉ đổi cấu hình vài clip thì liệt kê riêng chúng ở đây cho đỡ tốn quota
# (8/10/2026 đã chạy lại riêng ["pvd_goc_a"] sau khi đổi về vạch thẳng)
CLIPS_TO_RUN: list[str] | None = None  # None = mọi clip trong configs/demo_clips.yaml

subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PACKAGES], check=True)

bundle = next(Path("/kaggle/input").glob("**/scripts/infer_video.py")).parent.parent
project = Path("/kaggle/working/project")
shutil.copytree(bundle, project, dirs_exist_ok=True)
clip_lines = yaml.safe_load((project / "configs/demo_clips.yaml").read_text(encoding="utf-8"))["clips"]
out_dir = Path("/kaggle/working/demo")
environment = {**os.environ, "PYTHONPATH": str(project / "src"), "PYTHONUTF8": "1"}
try:
    for clip in CLIPS_TO_RUN or list(clip_lines):
        line = clip_lines[clip]
        print(f"== {clip} (vạch {line})", flush=True)
        subprocess.run(
            [sys.executable, "scripts/infer_video.py", "--source", f"data/videos/{clip}.mp4", "--weights", WEIGHTS,
             "--tracker", TRACKER, "--line", *map(str, line), "--out-dir", str(out_dir)],
            cwd=project, env=environment, check=True,
        )
finally:
    # Không đưa lại trọng số/video gốc vào output của kernel (nặng, và clip YouTube không được phát tán)
    shutil.rmtree(project, ignore_errors=True)
