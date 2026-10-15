#!/usr/bin/env bash
# Xử lý sẵn các clip demo (configs/demo_clips.yaml) bằng mô hình chính YOLO26s-v6 và tracker mặc định, đúng
# scripts/infer_video.py -> outputs/demo_precomputed/<clip>_tracked.mp4 + _events.csv + _run_config.json, để web demo
# trình chiếu mượt lúc bảo vệ. Chạy trên máy (CPU, ~1.5–2 giờ cho 6 clip 60 s); kết quả giống hệt khi chạy bằng kernel
# notebooks/kaggle_demo_gpu trên GPU, chỉ khác thời gian xử lý.
# Chạy:  bash scripts/make_demo_videos.sh [trọng số]
set -euo pipefail
export PYTHONUTF8=1
PY=.venv/Scripts/python.exe

WEIGHTS=${1:-weights/yolo26s_vn_v6_best.pt}
OUT=outputs/demo_precomputed

$PY - "$WEIGHTS" "$OUT" <<'EOF'
import subprocess
import sys
from pathlib import Path

import yaml

weights, out_dir = sys.argv[1], sys.argv[2]
clips = yaml.safe_load(Path("configs/demo_clips.yaml").read_text(encoding="utf-8"))["clips"]
for clip, line in clips.items():
    print(f"== {clip} (vạch {line})", flush=True)
    subprocess.run([sys.executable, "scripts/infer_video.py", "--source", f"data/videos/{clip}.mp4", "--weights", weights,
                    "--line", *map(str, line), "--out-dir", out_dir], check=True)
EOF
echo "Xong: $OUT"
