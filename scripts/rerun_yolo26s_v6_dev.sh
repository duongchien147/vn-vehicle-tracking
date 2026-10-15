#!/usr/bin/env bash
# Chấm YOLO26s-v6 trên các tập DEV theo configs/accept_yolo26s_v6.yaml, từng seed một (hai kernel Kaggle xong lệch giờ):
# val Hà Nội, dev đêm, và theo dõi trên VnTraffic + clip dev PVD góc C với bộ lọc box trùng none / 0.7 / 0.6 / 0.5 (luật
# chọn của configs/tracker_tuning_v6_dev2.yaml). Không chạm tập test. Đủ hai seed thì kiểm tiêu chí và chọn tracker:
#   bash scripts/rerun_yolo26s_v6_dev.sh weights/yolo26s_vn_v6_seed1_best.pt s1
#   bash scripts/rerun_yolo26s_v6_dev.sh weights/yolo26s_vn_v6_best.pt s42
#   python scripts/check_accept_yolo26s_v6.py
#   python scripts/select_tracker_v6.py --dev yolo26s_v6_tracker_pvd_goc_c_dev_s42 yolo26s_v6_tracker_pvd_goc_c_dev_s1 \
#     --vntraffic yolo26s_v6_tracker_vntraffic_s42 yolo26s_v6_tracker_vntraffic_s1
set -euo pipefail
export PYTHONUTF8=1 KMP_BLOCKTIME=0 OMP_WAIT_POLICY=PASSIVE
PY=.venv/Scripts/python.exe

WEIGHTS=$1
TAG=$2

echo "== phát hiện: val Hà Nội, dev đêm"
$PY scripts/eval_hanoi_test.py --split-dir data/labeling/hanoi/val --name yolo26s_v6_dev_$TAG --models $WEIGHTS
$PY scripts/eval_hanoi_test.py --split-dir data/labeling/night_extra/val --name yolo26s_v6_dev_nightdev_$TAG \
  --models $WEIGHTS
echo "== theo dõi: VnTraffic và clip dev PVD góc C, bộ lọc box trùng none / 0.7 / 0.6 / 0.5"
for seq in vntraffic pvd_goc_c_dev; do
  $PY scripts/eval_tracking.py --sequences $seq --trackers configs/botsort_tuned_nogmc.yaml \
    --agnostic-nms-ious none 0.7 0.6 0.5 --weights $WEIGHTS --name yolo26s_v6_tracker_${seq}_$TAG
done
echo "Xong $TAG"
