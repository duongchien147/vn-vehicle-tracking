#!/usr/bin/env bash
# Chọn ngưỡng bộ lọc box trùng trước tracker cho v6 trên các tập DEV theo configs/tracker_tuning_v6_nms.yaml:
# theo dõi trên VnTraffic, phát hiện trên val Hà Nội và dev đêm (night_extra/val). Không chạm tập test.
# Chạy:  bash scripts/rerun_tracker_v6_nms.sh
set -euo pipefail
export PYTHONUTF8=1
PY=.venv/Scripts/python.exe

V6=weights/yolo26n_vn_v6_best.pt
V6S1=weights/yolo26n_vn_v6_seed1_best.pt

echo "== dev VnTraffic: theo dõi"
$PY scripts/eval_tracking.py --sequences vntraffic --trackers configs/botsort_tuned_nogmc.yaml \
  --agnostic-nms-ious none 0.7 0.6 0.5 --weights $V6 $V6S1 --name tuning_v6_nms
echo "== dev phát hiện: val Hà Nội, dev đêm (tắt bộ lọc = thư mục v6_dev, v6_dev_nightdev đã có)"
for iou in 0.7 0.6 0.5; do
  $PY scripts/eval_hanoi_test.py --split-dir data/labeling/hanoi/val --agnostic-nms-iou $iou --name dev_v6_nms$iou \
    --models $V6 $V6S1
  $PY scripts/eval_hanoi_test.py --split-dir data/labeling/night_extra/val --agnostic-nms-iou $iou \
    --name dev_v6_nms${iou}_nightdev --models $V6 $V6S1
done
echo "Xong"
