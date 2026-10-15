#!/usr/bin/env bash
# Chấm bản OpenVINO 544x960 của mô hình chính YOLO26s-v6 (chạy trên GPU Intel tích hợp, như lựa chọn trong web demo)
# trên các tập DEV, cùng cách chấm bản PyTorch trong scripts/rerun_yolo26s_v6_dev.sh để so trực tiếp: val Hà Nội, dev
# đêm, theo dõi trên VnTraffic + clip dev PVD góc C (không lọc / lọc box trùng 0.6). Không chạm tập test.
# Kết quả: outputs/eval_hanoi/openvino_yolo26s_v6{,_nightdev}/, outputs/eval_tracking/openvino_yolo26s_v6_<sequence>/
set -euo pipefail
export PYTHONUTF8=1 KMP_BLOCKTIME=0 OMP_WAIT_POLICY=PASSIVE
PY=.venv/Scripts/python.exe

WEIGHTS="weights/yolo26s_vn_v6_best_544x960_openvino_model#gpu"

echo "== phát hiện: val Hà Nội, dev đêm"
$PY scripts/eval_hanoi_test.py --split-dir data/labeling/hanoi/val --name openvino_yolo26s_v6 --models "$WEIGHTS"
$PY scripts/eval_hanoi_test.py --split-dir data/labeling/night_extra/val --name openvino_yolo26s_v6_nightdev \
  --models "$WEIGHTS"
echo "== theo dõi: VnTraffic và clip dev PVD góc C, không lọc / lọc box trùng 0.6"
for seq in vntraffic pvd_goc_c_dev; do
  $PY scripts/eval_tracking.py --sequences $seq --trackers configs/botsort_tuned_nogmc.yaml \
    --agnostic-nms-ious none 0.6 --weights "$WEIGHTS" --name openvino_yolo26s_v6_${seq}
done
echo "Xong"
