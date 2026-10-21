"""Nối phát hiện YOLO26x trên đường bên trái (chiều ngược lại, dưới hàng cây) thành ứng viên track để duyệt bằng mắt."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from review_tracks import iou  # noqa: E402

from vtrack.dataset import merge_riders_into_motorbikes

LINE_Y = 637
# Mép trái dải phân cách (đo trên ảnh): đường bên trái nằm bên trái đường thẳng này
MEDIAN = ((867.0, 667.0), (300.0, 1075.0))
CLASS_ID = {"person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "bus": 5, "truck": 7}
NAME = {v: k for k, v in CLASS_ID.items()}


def left_of_median(x: float, y: float) -> bool:
    (x1, y1), (x2, y2) = MEDIAN
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1) > 0


def frame_boxes(raw: list) -> list[tuple[str, list[float], float]]:
    boxes = [(CLASS_ID[label], x1, y1, x2, y2) for label, x1, y1, x2, y2, _ in raw]
    scores = [conf for *_, conf in raw]
    merged = merge_riders_into_motorbikes(boxes, motor_class=3, rider_class=0)
    out = []
    for (cls, x1, y1, x2, y2), conf in zip(merged, scores):
        if cls in (0, 1):  # người đi bộ / xe đạp không phải lớp của dự án
            continue
        h = y2 - y1
        below = max(0.0, y2 - max(y1, LINE_Y)) / h if h > 0 else 0
        if below >= 0.5 and left_of_median((x1 + x2) / 2, y2):
            out.append((NAME[cls], [x1, y1, x2, y2], conf))
    return out


def link(dets: dict[int, list], min_iou: float = 0.2, max_gap: int = 4) -> list[dict]:
    tracks: list[dict] = []
    for f in sorted(dets):
        for label, box, conf in sorted(dets[f], key=lambda d: -d[2]):
            best, best_iou = None, min_iou
            for t in tracks:
                last_f = max(t["boxes"])
                if 0 < f - last_f <= max_gap and f not in t["boxes"]:
                    v = iou(box, t["boxes"][last_f])
                    if v > best_iou:
                        best, best_iou = t, v
            if best is None:
                best = {"boxes": {}, "conf": {}, "labels": {}}
                tracks.append(best)
            best["boxes"][f] = box
            best["conf"][f] = conf
            best["labels"][f] = label
    return tracks


def main(det_path: str, out_path: str) -> None:
    raw = json.load(open(det_path))["left_road"]
    dets = {int(f): frame_boxes(v) for f, v in raw.items()}
    tracks = link(dets)
    keep = [t for t in tracks if len(t["boxes"]) >= 3 and max(t["conf"].values()) >= 0.4]
    for i, t in enumerate(keep):
        fs = sorted(t["boxes"])
        labels = sorted(set(t["labels"].values()))
        print(f"#{i} frame {fs[0]}->{fs[-1]} ({len(fs)} frame có box), conf max {max(t['conf'].values()):.2f}, "
              f"lớp {labels}, box đầu {[round(v) for v in t['boxes'][fs[0]]]}, cuối {[round(v) for v in t['boxes'][fs[-1]]]}")
    json.dump([{"boxes": {str(k): v for k, v in t["boxes"].items()}, "conf": {str(k): v for k, v in t["conf"].items()},
                "labels": {str(k): v for k, v in t["labels"].items()}} for t in keep], open(out_path, "w"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
