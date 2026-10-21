"""YOLO26x COCO @1280 trên clip dev (mỗi STEP frame) -> JSON box pixel; chỉ để soát chéo nhãn, không tạo nhãn."""
import json
import sys
import time

import cv2
from ultralytics import YOLO

from vtrack.pseudo_label import predict_teacher

STEP = int(sys.argv[2]) if len(sys.argv) > 2 else 3
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
model = YOLO("weights/yolo26x.pt")
cap = cv2.VideoCapture("data/tracking_gt/pvd_goc_c_dev/clip.mp4")
out = {}
frame_idx = 0
t0 = time.time()
while True:
    ok, image = cap.read()
    if not ok or len(out) >= LIMIT:
        break
    if frame_idx % STEP == 0:
        h, w = image.shape[:2]
        boxes = predict_teacher(model, image, 1280, 0.1)
        out[frame_idx] = [
            [b.class_id, (b.center_x - b.width / 2) * w, (b.center_y - b.height / 2) * h,
             (b.center_x + b.width / 2) * w, (b.center_y + b.height / 2) * h, b.confidence]
            for b in boxes
        ]
        if len(out) % 20 == 0:
            print(frame_idx, f"{(time.time() - t0) / len(out):.2f} s/frame", flush=True)
    frame_idx += 1
json.dump(out, open(sys.argv[1], "w"))
print("xong", len(out), f"{time.time() - t0:.0f} s")
