#!/usr/bin/env bash
# Độ dao động giữa các lần huấn luyện của YOLO26s (seed 42 = bản YOLO26s v5 gốc, seed 1 và 2 train 12–13/10/2026),
# cùng các tập như scripts/rerun_seeds.sh làm cho v5 (YOLO26n). Mục đích: kiểm kết luận "YOLO26s hơn YOLO26n ban đêm"
# vốn chỉ dựa trên một seed. Ghi vào thư mục con "seeds_yolo26s".
# Chạy khi máy rảnh (~2 giờ CPU):  bash scripts/rerun_seeds_yolo26s.sh
set -euo pipefail
export PYTHONUTF8=1
PY=.venv/Scripts/python.exe

S42=weights/yolo26s_vn_v5_best.pt
S1=weights/yolo26s_vn_v5_seed1_best.pt
S2=weights/yolo26s_vn_v5_seed2_best.pt
NAME=seeds_yolo26s

echo "== val, test Hà Nội; vn_extra ngày, đêm; dev đêm night_extra"
$PY scripts/eval_hanoi_test.py --split-dir data/labeling/hanoi/val --name $NAME --models $S42 $S1 $S2
$PY scripts/eval_hanoi_test.py --name $NAME --models $S42 $S1 $S2
for split in day night; do
  $PY scripts/eval_hanoi_test.py --split-dir data/labeling/vn_extra/$split --name $NAME --models $S42 $S1 $S2
done
$PY scripts/eval_hanoi_test.py --split-dir data/labeling/night_extra/val --name ${NAME}_nightdev --models $S42 $S1 $S2
echo "== 20 frame clip PVD, val UVH-26, UA-DETRAC, test Roboflow"
$PY scripts/eval_clip_detection.py --name $NAME --models $S42 $S1 $S2
$PY scripts/eval_uvh26.py --name $NAME --models $S42 $S1 $S2
$PY scripts/eval_ua_detrac.py --name $NAME --models $S42 $S1 $S2
$PY scripts/eval_hanoi_test.py --split-dir data/processed/vn_vehicles/test --name ${NAME}_roboflow --models $S42 $S1 $S2
echo "== theo dõi: dev VnTraffic, test PVD (tracker mặc định)"
$PY scripts/eval_tracking.py --sequences vntraffic --trackers configs/botsort_tuned_nogmc.yaml --name ${NAME}_dev \
  --weights $S42 $S1 $S2
$PY scripts/eval_tracking.py --sequences pvd_goc_a --trackers configs/botsort_tuned_nogmc.yaml --name ${NAME}_test \
  --weights $S42 $S1 $S2
echo "Xong"
