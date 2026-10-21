"""YOLO26x COCO trên vùng cắt của clip dev, từng frame -> JSON box pixel (toạ độ ảnh gốc) kèm lớp COCO thô.
Chỉ để đề xuất box cho người/trợ lý AI duyệt, không phải mô hình của dự án."""
import json
import sys
import time

import cv2
from ultralytics import YOLO

# (tên, x1, y1, x2, y2, frame đầu, frame cuối)
REGIONS = [
    ("left_road", 0, 600, 1000, 1080, 0, 599),
    ("pair_267_274", 1250, 660, 1650, 960, 84, 134),
    ("post_461", 1400, 650, 1600, 800, 0, 599),
]
KEEP = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}

model = YOLO("weights/yolo26x.pt")
cap = cv2.VideoCapture("data/tracking_gt/pvd_goc_c_dev/clip.mp4")
out = {name: {} for name, *_ in REGIONS}
t0 = time.time()
frame_idx = 0
while True:
    ok, image = cap.read()
    if not ok:
        break
    for name, x1, y1, x2, y2, f0, f1 in REGIONS:
        if not f0 <= frame_idx <= f1 or (name == "post_461" and frame_idx % 30):
            continue
        crop = image[y1:y2, x1:x2]
        result = model.predict(crop, imgsz=1024, conf=0.1, verbose=False)[0]
        boxes = []
        for cls, conf, (a, b, c, d) in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(),
                                           result.boxes.xyxy.tolist()):
            label = model.names[int(cls)]
            if label in KEEP:
                boxes.append([label, round(a + x1, 1), round(b + y1, 1), round(c + x1, 1), round(d + y1, 1), round(conf, 3)])
        out[name][frame_idx] = boxes
    if frame_idx % 50 == 0:
        print(frame_idx, f"{time.time() - t0:.0f} s", flush=True)
    frame_idx += 1
json.dump(out, open(sys.argv[1], "w"))
print("xong", f"{time.time() - t0:.0f} s")
