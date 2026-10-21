"""Tách box hai xe máy chạy song song (267 mũ xanh lá, 274 mũ trắng) ở frame 84–134 từ phát hiện YOLO26x từng frame:
mỗi frame ghép phát hiện với box frame trước của từng xe (IoU lớn nhất, không dùng chung)."""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from review_tracks import frames_of, iou  # noqa: E402

from vtrack.dataset import merge_riders_into_motorbikes

CLASS_ID = {"person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "bus": 5, "truck": 7}


def motorbikes(raw: list) -> list[tuple[list[float], float]]:
    boxes = [(CLASS_ID[label], x1, y1, x2, y2) for label, x1, y1, x2, y2, _ in raw]
    merged = merge_riders_into_motorbikes(boxes, motor_class=3, rider_class=0)
    return [([x1, y1, x2, y2], conf) for (cls, x1, y1, x2, y2), (*_, conf) in zip(merged, raw) if cls == 3]


def main(det_path: str, ann_path: str, out_path: str) -> None:
    dets = {int(f): motorbikes(v) for f, v in json.load(open(det_path))["pair_267_274"].items()}
    ann = json.load(open(ann_path, encoding="utf-8"))
    per = {str(t["id"])[-3:]: frames_of(t) for t in ann["tracks"]}
    start = 84
    last = {"267": per["267"][start], "274": per["274"][start]}
    result = {"267": {}, "274": {}}
    for f in range(start, 135):
        cands = dets.get(f, [])
        pairs = sorted(((iou(last[k], b), k, i) for k in last for i, (b, _) in enumerate(cands)), reverse=True)
        used_k, used_i = set(), set()
        for v, k, i in pairs:
            if v < 0.3 or k in used_k or i in used_i:
                continue
            used_k.add(k)
            used_i.add(i)
            result[k][f] = cands[i][0]
            last[k] = cands[i][0]
        print(f, {k: ([round(x) for x in result[k][f]] if f in result[k] else None) for k in result})
    json.dump(result, open(out_path, "w"))


if __name__ == "__main__":
    main(*sys.argv[1:4])
