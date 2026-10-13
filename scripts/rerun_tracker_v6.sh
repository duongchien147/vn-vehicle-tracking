#!/usr/bin/env bash
# Chấm tracker mới configs/botsort_v6.yaml (chọn trên dev bằng configs/tracker_tuning_v6.yaml) trên tập TEST, một lần,
# kèm v5 để so sánh, và chạy lại đếm xe ngày / đêm với cấu hình mặc định mới. Ghi vào các thư mục "tracker_v6*".
# Chạy:  bash scripts/rerun_tracker_v6.sh
set -euo pipefail
export PYTHONUTF8=1 KMP_BLOCKTIME=0 OMP_WAIT_POLICY=PASSIVE
PY=.venv/Scripts/python.exe

V5=weights/yolo26n_vn_v5_best.pt
V6=weights/yolo26n_vn_v6_best.pt
V6S1=weights/yolo26n_vn_v6_seed1_best.pt
V6_OV="weights/yolo26n_vn_v6_best_544x960_openvino_model#gpu"
TRACKER=configs/botsort_v6.yaml

echo "== dev VnTraffic: v5 và bản OpenVINO của v6 với tracker mới (để báo cáo)"
$PY scripts/eval_tracking.py --sequences vntraffic --trackers $TRACKER --name tracker_v6_dev --weights $V5 "$V6_OV"
echo "== test PVD: tracking"
$PY scripts/eval_tracking.py --sequences pvd_goc_a --trackers $TRACKER --min-track-ages 1 3 --name tracker_v6_test \
  --weights $V6 $V6S1 $V5
echo "== đếm xe ban ngày (nhãn CVAT PVD) và ban đêm (người đếm tay)"
for weights in $V6 $V6S1; do
  run=$(basename "$weights" .pt)
  $PY scripts/eval_counting.py --sequence pvd_goc_a --weights "$weights" --tracker $TRACKER --name "tracker_v6_$run"
  for clip in ham_chui_dem_2 ham_chui_dem_3; do
    $PY scripts/infer_video.py --source data/videos/$clip.mp4 --weights "$weights" --tracker $TRACKER \
      --out-dir "outputs/manual_count_runs/tracker_v6_$run"
  done
done
$PY scripts/eval_manual_counts.py --runs outputs/manual_count_runs/v5 outputs/manual_count_runs/v6_yolo26n_vn_v6_best \
  outputs/manual_count_runs/tracker_v6_yolo26n_vn_v6_best outputs/manual_count_runs/tracker_v6_yolo26n_vn_v6_seed1_best \
  --name night_human_tracker_v6
echo "Xong"
