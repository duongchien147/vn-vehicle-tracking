#!/usr/bin/env bash
# Chấm mọi ứng viên tracker của v6 trên clip dev mới pvd_goc_c_dev theo configs/tracker_tuning_v6_dev2.yaml
# (lưới ngưỡng + bộ lọc box trùng). Kết quả VnTraffic của cùng ứng viên đã có ở tuning_v6 và tuning_v6_nms.
# Không chạm tập test. Chạy:  bash scripts/rerun_tracker_v6_dev2.sh  rồi  python scripts/select_tracker_v6.py
set -euo pipefail
export PYTHONUTF8=1
PY=.venv/Scripts/python.exe

V6=weights/yolo26n_vn_v6_best.pt
V6S1=weights/yolo26n_vn_v6_seed1_best.pt

echo "== lưới ngưỡng BoT-SORT"
$PY scripts/eval_tracking.py --grid configs/tracker_tuning_v6_dev2.yaml --sequences pvd_goc_c_dev \
  --weights $V6 $V6S1 --name tuning_v6_dev2
echo "== bộ lọc box trùng (cấu hình hiện tại)"
$PY scripts/eval_tracking.py --sequences pvd_goc_c_dev --trackers configs/botsort_tuned_nogmc.yaml \
  --agnostic-nms-ious none 0.7 0.6 0.5 --weights $V6 $V6S1 --name tuning_v6_dev2_nms
echo "Xong"
