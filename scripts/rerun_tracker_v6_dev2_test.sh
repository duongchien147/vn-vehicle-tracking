#!/usr/bin/env bash
# Chấm cấu hình chọn trên hai tập dev theo configs/tracker_tuning_v6_dev2.yaml (botsort_tuned_nogmc + bộ lọc box trùng
# IoU 0.6) trên tập TEST, một lần: tracking clip PVD, đếm xe ban ngày (nhãn CVAT) và ban đêm (người đếm tay).
# Thêm bản OpenVINO của web demo trên hai tập dev (để báo cáo). Ghi vào các thư mục "*nms06*".
# Chạy:  bash scripts/rerun_tracker_v6_dev2_test.sh
set -euo pipefail
export PYTHONUTF8=1 KMP_BLOCKTIME=0 OMP_WAIT_POLICY=PASSIVE
PY=.venv/Scripts/python.exe

V6=weights/yolo26n_vn_v6_best.pt
V6S1=weights/yolo26n_vn_v6_seed1_best.pt
V6_OV="weights/yolo26n_vn_v6_best_544x960_openvino_model#gpu"
TRACKER=configs/botsort_tuned_nogmc.yaml
IOU=0.6

echo "== dev: bản OpenVINO của web demo, tắt / bật bộ lọc"
for seq in vntraffic pvd_goc_c_dev; do
  $PY scripts/eval_tracking.py --sequences $seq --trackers $TRACKER --agnostic-nms-ious none $IOU \
    --weights "$V6_OV" --name nms06_openvino_dev_$seq
done
echo "== test PVD: tracking"
$PY scripts/eval_tracking.py --sequences pvd_goc_a --trackers $TRACKER --agnostic-nms-ious $IOU --min-track-ages 1 3 \
  --weights $V6 $V6S1 --name nms06_v6_test
echo "== đếm xe ban ngày (nhãn CVAT PVD) và ban đêm (người đếm tay)"
for weights in $V6 $V6S1; do
  run=$(basename "$weights" .pt)
  $PY scripts/eval_counting.py --sequence pvd_goc_a --weights "$weights" --agnostic-nms-iou $IOU --name "nms06_$run"
  for clip in ham_chui_dem_2 ham_chui_dem_3; do
    $PY scripts/infer_video.py --source data/videos/$clip.mp4 --weights "$weights" --agnostic-nms-iou $IOU \
      --out-dir "outputs/manual_count_runs/nms06_$run"
  done
done
$PY scripts/eval_manual_counts.py --runs outputs/manual_count_runs/v5 outputs/manual_count_runs/v6_yolo26n_vn_v6_best \
  outputs/manual_count_runs/nms06_yolo26n_vn_v6_best outputs/manual_count_runs/nms06_yolo26n_vn_v6_seed1_best \
  --name night_human_nms06
echo "Xong"
