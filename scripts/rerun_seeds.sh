#!/usr/bin/env bash
# Đo độ dao động giữa các lần huấn luyện: v5 train lại với seed 1 và 2 (seed 42 là bản v5 gốc), chấm trên mọi tập
# phát hiện và hai tập theo dõi, kèm v3 làm mốc so sánh. Ghi vào thư mục con "seeds".
# Chạy khi máy rảnh (~1 giờ CPU):  bash scripts/rerun_seeds.sh
set -euo pipefail
export PYTHONUTF8=1
PY=.venv/Scripts/python.exe

V3=weights/yolo26n_vn_v3_best.pt
V5=weights/yolo26n_vn_v5_best.pt
S1=weights/yolo26n_vn_v5_seed1_best.pt
S2=weights/yolo26n_vn_v5_seed2_best.pt

echo "== val, test Hà Nội; vn_extra ngày, đêm"
$PY scripts/eval_hanoi_test.py --split-dir data/labeling/hanoi/val --name seeds --models $V3 $V5 $S1 $S2
$PY scripts/eval_hanoi_test.py --name seeds --models $V3 $V5 $S1 $S2
for split in day night; do
  $PY scripts/eval_hanoi_test.py --split-dir data/labeling/vn_extra/$split --name seeds --models $V3 $V5 $S1 $S2
done
echo "== 20 frame clip PVD, val UVH-26, UA-DETRAC, test Roboflow"
$PY scripts/eval_clip_detection.py --name seeds --models $V3 $V5 $S1 $S2
$PY scripts/eval_uvh26.py --name seeds --models $V3 $V5 $S1 $S2
$PY scripts/eval_ua_detrac.py --name seeds --models $V3 $V5 $S1 $S2
$PY scripts/eval_hanoi_test.py --split-dir data/processed/vn_vehicles/test --name seeds_roboflow --models $V5 $S1 $S2
echo "== theo dõi: dev VnTraffic, test PVD (tracker mặc định)"
$PY scripts/eval_tracking.py --sequences vntraffic --trackers configs/botsort_tuned_nogmc.yaml --name seeds_dev \
  --weights $V3 $V5 $S1 $S2
$PY scripts/eval_tracking.py --sequences pvd_goc_a --trackers configs/botsort_tuned_nogmc.yaml --name seeds_test \
  --weights $V5 $S1 $S2
echo "Xong"
