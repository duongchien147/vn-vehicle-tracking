#!/usr/bin/env bash
# Chấm lại các tập frame cắt từ video (test/val Hà Nội, vn_extra ngày/đêm) và UA-DETRAC với khoảng tin cậy bootstrap
# theo khối (vtrack.coco_eval: khối BLOCK_SECONDS giây trong mỗi clip; UA-DETRAC: mỗi video một cụm), bên cạnh khoảng
# theo ảnh như cũ. mAP không đổi; cột main_mAP50_block_ci_* là mới. Ghi vào thư mục con "ci_block".
# Không có khoảng theo khối: clip PVD (20 s = 2 khối, quá ít), val UVH-26 (không có thông tin camera), test Roboflow.
# Chạy khi máy rảnh (~1.5 giờ CPU):  bash scripts/rerun_block_ci.sh
set -euo pipefail
export PYTHONUTF8=1
PY=.venv/Scripts/python.exe

COCO=weights/yolo26n.pt@1280
V1=weights/yolo26n_vn_best.pt
V2=weights/yolo26n_vn_v2_best.pt
V3=weights/yolo26n_vn_v3_best.pt
V4=weights/yolo26n_vn_v4_best.pt
V5=weights/yolo26n_vn_v5_best.pt
V5_1280=weights/yolo26n_vn_v5_best.pt@1280
S=weights/yolo26s_vn_v5_best.pt
S_1280=weights/yolo26s_vn_v5_best.pt@1280
RTDETR=weights/rtdetrv2_r18_vn_v5_best

echo "== val Hà Nội (dev)"
$PY scripts/eval_hanoi_test.py --split-dir data/labeling/hanoi/val --name ci_block --models $V3 $V4 $V5 $V5_1280
echo "== test Hà Nội"
$PY scripts/eval_hanoi_test.py --name ci_block --models $COCO $V1 $V2 $V3 $V4 $V5 $V5_1280 $S $S_1280 $RTDETR
echo "== vn_extra ngày, đêm"
for split in day night; do
  $PY scripts/eval_hanoi_test.py --split-dir data/labeling/vn_extra/$split --name ci_block \
    --models $COCO $V1 $V2 $V3 $V4 $V5 $V5_1280 $S $S_1280 $RTDETR
done
echo "== UA-DETRAC đêm, ngày"
$PY scripts/eval_ua_detrac.py --name ci_block --models $COCO $V3 $V5 $V5_1280 $S $S_1280 $RTDETR
echo "Xong"
