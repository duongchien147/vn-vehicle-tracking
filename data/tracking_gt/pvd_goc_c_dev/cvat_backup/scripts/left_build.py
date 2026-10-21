"""Dựng track mới cho xe ở đường bên trái từ ứng viên đã duyệt bằng mắt (left_cands.json):
gộp các ứng viên cùng một xe, lớp do người duyệt quyết định, bỏ box phình to vì lá che (> 1.6 lần diện tích trung vị)."""
import json
import os
import statistics
import sys

SP = os.path.dirname(__file__)
# (nhãn dự án, các ứng viên cùng một xe, chọn box lớn hơn khi trùng frame)
VEHICLES = [
    ("motorbike", [0], False),
    ("motorbike", [1], False),
    ("motorbike", [2, 3], False),
    ("motorbike", [4, 5], False),
    ("motorbike", [11], False),
    ("car", [12], True),  # sedan trắng
    ("motorbike", [15], False),
    ("car", [16, 17], True),  # ô tô xanh ngọc: box lớn hơn = phần xe thấy được qua lá
]
EXTEND_308 = (8, [392, 393, 394, 395])  # ô tô tối màu: track 308 có từ 396, thêm 392–395 từ ứng viên #6 (chỉ số 6)


def area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def build(cands, idxs, prefer_large):
    per_frame = {}
    for i in idxs:
        for f, b in cands[i]["boxes"].items():
            f = int(f)
            if f not in per_frame:
                per_frame[f] = b
            elif (area(b) > area(per_frame[f])) == prefer_large:
                per_frame[f] = b
    if not prefer_large:
        med = statistics.median(area(b) for b in per_frame.values())
        per_frame = {f: b for f, b in per_frame.items() if area(b) <= 1.6 * med}
    return dict(sorted(per_frame.items()))


def main():
    cands = json.load(open(os.path.join(SP, "left_cands.json")))
    out = []
    for label, idxs, prefer_large in VEHICLES:
        boxes = build(cands, idxs, prefer_large)
        fs = list(boxes)
        print(label, idxs, f"{fs[0]}->{fs[-1]}", len(fs), "keyframe", "bỏ" if False else "")
        out.append({"label": label, "boxes": {str(f): [round(v, 1) for v in b] for f, b in boxes.items()}})
    ext = {str(f): [round(v, 1) for v in cands[6]["boxes"][str(f)]] for f in EXTEND_308[1]}
    json.dump({"new": out, "extend_308": ext}, open(os.path.join(SP, "left_new.json"), "w"))


if __name__ == "__main__":
    main()
