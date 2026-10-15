#!/usr/bin/env bash
# Chấm YOLO26s-v6 (seed 42 và 1) trên các tập DEV theo configs/accept_yolo26s_v6.yaml: val Hà Nội, dev đêm, VnTraffic
# (IDR, tracker botsort_tuned_nogmc chưa lọc box trùng), rồi kiểm tiêu chí nhận. Cùng lúc chấm các ứng viên bộ lọc box
# trùng trên hai tập dev theo luật của configs/tracker_tuning_v6_dev2.yaml. Không chạm tập test.
# Chạy:  bash scripts/rerun_yolo26s_v6_dev.sh
set -euo pipefail
export PYTHONUTF8=1 KMP_BLOCKTIME=0 OMP_WAIT_POLICY=PASSIVE
PY=.venv/Scripts/python.exe

S42=weights/yolo26s_vn_v6_best.pt
S1=weights/yolo26s_vn_v6_seed1_best.pt

echo "== phát hiện: val Hà Nội, dev đêm"
$PY scripts/eval_hanoi_test.py --split-dir data/labeling/hanoi/val --name yolo26s_v6_dev --models $S42 $S1
$PY scripts/eval_hanoi_test.py --split-dir data/labeling/night_extra/val --name yolo26s_v6_dev_nightdev --models $S42 $S1
echo "== theo dõi: VnTraffic và clip dev PVD góc C, bộ lọc box trùng none / 0.7 / 0.6 / 0.5"
for seq in vntraffic pvd_goc_c_dev; do
  $PY scripts/eval_tracking.py --sequences $seq --trackers configs/botsort_tuned_nogmc.yaml \
    --agnostic-nms-ious none 0.7 0.6 0.5 --weights $S42 $S1 --name yolo26s_v6_tracker_$seq
done
echo "== tiêu chí nhận"
$PY scripts/check_accept_yolo26s_v6.py
echo "== chọn tracker (chỉ có ý nghĩa nếu được nhận)"
$PY scripts/select_tracker_v6.py --dev yolo26s_v6_tracker_pvd_goc_c_dev --vntraffic yolo26s_v6_tracker_vntraffic
echo "Xong"
