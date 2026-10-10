#!/usr/bin/env bash
# Chấm lại mọi bảng phát hiện của báo cáo với khoảng tin cậy bootstrap 1.000 lần (vtrack.coco_eval.BOOTSTRAP_ROUNDS),
# thay cho bản cũ 20 lần. Điểm mAP không đổi (dự đoán tất định); chỉ khoảng tin cậy thay đổi.
# Mỗi tập gộp mọi mô hình từng chấm trên tập đó vào một lần chạy, ghi vào thư mục con "ci1000".
# Chạy khi máy rảnh (~1.5-2 giờ CPU, RT-DETRv2 chiếm phần lớn):  bash scripts/rerun_detection_ci.sh
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
$PY scripts/eval_hanoi_test.py --split-dir data/labeling/hanoi/val --name ci1000 --models $V3 $V4 $V5 $V5_1280
echo "== test Hà Nội"
$PY scripts/eval_hanoi_test.py --name ci1000 --models $COCO $V1 $V2 $V3 $V4 $V5 $V5_1280 $S $S_1280 $RTDETR
echo "== vn_extra ngày, đêm"
for split in day night; do
  $PY scripts/eval_hanoi_test.py --split-dir data/labeling/vn_extra/$split --name ci1000 \
    --models $COCO $V1 $V2 $V3 $V4 $V5 $V5_1280 $S $S_1280 $RTDETR
done
echo "== 20 frame clip PVD"
$PY scripts/eval_clip_detection.py --name ci1000 --models $COCO $V2 $V3 $V4 $V5 $V5_1280 $S $S_1280 $RTDETR
echo "== val UVH-26"
$PY scripts/eval_uvh26.py --name ci1000 --models $COCO $V3 $V5 $V5_1280 $S $S_1280 $RTDETR
echo "== UA-DETRAC đêm, ngày"
$PY scripts/eval_ua_detrac.py --name ci1000 --models $COCO $V3 $V5 $V5_1280 $S $S_1280 $RTDETR
echo "== test Roboflow"
$PY scripts/eval_hanoi_test.py --split-dir data/processed/vn_vehicles/test --name ci1000_roboflow \
  --models $COCO $V5 $S $RTDETR
echo "Xong"
