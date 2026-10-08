"""Đo tốc độ các mô hình trên GPU T4 của Kaggle bằng ĐÚNG script đo trên máy local (scripts/benchmark_speed.py), để
báo cáo có bảng tốc độ trên cả CPU lẫn GPU. Dữ liệu (mã nguồn, trọng số, video VnTraffic 1080p) lấy từ dataset private
tạo bởi scripts/kaggle_push_benchmark_bundle.py.

Đẩy lên: python scripts/kaggle_train.py push --notebook-dir notebooks/kaggle_benchmark_gpu
Kết quả: /kaggle/working/outputs/benchmark_speed/e2_gpu_t4/ (results.csv, timings.csv, run_config.json)

Cả hai họ mô hình chạy PyTorch FP32 trên GPU (ultralytics tự chọn GPU; vtrack.rtdetr chọn GPU khi có), giống phép so
PyTorch–PyTorch trên CPU. Chưa dùng TensorRT/FP16, nên đây là tốc độ thận trọng, có thể nhanh hơn nữa.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# Cùng phiên bản với máy local (requirements.txt); thiếu gói nào thì import vtrack lỗi ngay từ đầu
PACKAGES = ["ultralytics==8.4.163", "supervision==0.30.5", "transformers==5.18.0", "openvino", "lap",
            "pycocotools==2.0.11", "ensemble-boxes==1.0.9", "trackeval==1.3.0", "imageio-ffmpeg"]
MODELS = [
    "weights/yolo26n_vn_v5_best.pt",
    "weights/yolo26s_vn_v5_best.pt",
    "weights/rtdetrv2_r18_vn_v5_best",
    "weights/yolo26s_vn_v5_best.pt@1280",  # cùng số điểm ảnh với RT-DETRv2 960x960
]
COOLDOWN_S = "2"  # GPU không bị hạ xung như CPU laptop: nghỉ ngắn giữa các đợt là đủ, đỡ tốn quota

subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PACKAGES], check=True)

bundle = next(Path("/kaggle/input").glob("**/scripts/benchmark_speed.py")).parent.parent
project = Path("/kaggle/working/project")
shutil.copytree(bundle, project, dirs_exist_ok=True)
print("Gói dữ liệu:", bundle, flush=True)

environment = {**os.environ, "PYTHONPATH": str(project / "src"), "PYTHONUTF8": "1"}
try:
    subprocess.run(
        [sys.executable, "scripts/benchmark_speed.py", "--name", "e2_gpu_t4", "--video", "data/vntraffic_1080p.mp4",
         "--tracker", "configs/botsort_tuned_nogmc.yaml", "--cooldown", COOLDOWN_S, "--models", *MODELS],
        cwd=project, env=environment, check=True,
    )
    shutil.copytree(project / "outputs", "/kaggle/working/outputs", dirs_exist_ok=True)
finally:
    # Kể cả khi lỗi: không để trọng số/video (~110 MB) lại trong output của kernel, tải về rất nặng
    shutil.rmtree(project, ignore_errors=True)
