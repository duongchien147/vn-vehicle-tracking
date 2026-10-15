#!/usr/bin/env bash
# Chấm YOLO26s-v6 (seed 42 và 1) trên mọi tập TEST, một lần, sau khi đã được nhận theo configs/accept_yolo26s_v6.yaml và
# đã chọn tracker trên hai tập dev (bộ lọc box trùng IoU 0.6, như v6). Cùng các tập và cách chấm với scripts/rerun_v6.sh.
# Ghi vào các thư mục "yolo26s_v6*". Chạy khi máy rảnh (~1.5–2 giờ CPU):  bash scripts/rerun_yolo26s_v6_test.sh
set -euo pipefail
export PYTHONUTF8=1 KMP_BLOCKTIME=0 OMP_WAIT_POLICY=PASSIVE
PY=.venv/Scripts/python.exe

S42=weights/yolo26s_vn_v6_best.pt
S1=weights/yolo26s_vn_v6_seed1_best.pt
NAME=yolo26s_v6
IOU=0.6

echo "== test Hà Nội; vn_extra ngày, đêm"
$PY scripts/eval_hanoi_test.py --name $NAME --models $S42 $S1
for split in day night; do
  $PY scripts/eval_hanoi_test.py --split-dir data/labeling/vn_extra/$split --name $NAME --models $S42 $S1
done
echo "== 20 frame clip PVD, val UVH-26, UA-DETRAC, test Roboflow"
$PY scripts/eval_clip_detection.py --name $NAME --models $S42 $S1
$PY scripts/eval_uvh26.py --name $NAME --models $S42 $S1
$PY scripts/eval_ua_detrac.py --name $NAME --models $S42 $S1
$PY scripts/eval_hanoi_test.py --split-dir data/processed/vn_vehicles/test --name ${NAME}_roboflow --models $S42 $S1
echo "== theo dõi: test PVD (tracker mặc định + bộ lọc box trùng)"
$PY scripts/eval_tracking.py --sequences pvd_goc_a --trackers configs/botsort_tuned_nogmc.yaml --agnostic-nms-ious $IOU \
  --min-track-ages 1 3 --name ${NAME}_test --weights $S42 $S1
echo "== đếm xe: ban ngày (nhãn CVAT PVD), ban đêm (người đếm tay, clip hầm chui 2 và 3)"
for weights in $S42 $S1; do
  run=$(basename "$weights" .pt)
  $PY scripts/eval_counting.py --sequence pvd_goc_a --weights "$weights" --agnostic-nms-iou $IOU --name "${NAME}_$run"
  for clip in ham_chui_dem_2 ham_chui_dem_3; do
    $PY scripts/infer_video.py --source data/videos/$clip.mp4 --weights "$weights" --agnostic-nms-iou $IOU \
      --out-dir "outputs/manual_count_runs/${NAME}_$run"
  done
done
$PY scripts/eval_manual_counts.py --runs outputs/manual_count_runs/v5 outputs/kaggle/vtrack-demo-gpu/demo \
  outputs/manual_count_runs/nms06_yolo26n_vn_v6_best outputs/manual_count_runs/${NAME}_yolo26s_vn_v6_best \
  outputs/manual_count_runs/${NAME}_yolo26s_vn_v6_seed1_best --name night_human_yolo26s_v6
echo "Xong"
