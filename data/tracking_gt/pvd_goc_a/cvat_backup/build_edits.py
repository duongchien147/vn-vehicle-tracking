"""Áp các chỉnh sửa đã kiểm bằng mắt lên annotation CVAT (bản sao cục bộ) -> edited.json + edit_log.json.
Frame CVAT đánh từ 0; frame MOT (bản xuất) đánh từ 1."""
import json
import numpy as np

S = r"outputs/cvat_edit_tmp"
BACKUP = r"E:/cv and dl/data/tracking_gt/pvd_goc_a/cvat_backup/job4541067_20261005_before_ai_edit.json"
IGNORE_LABEL_ID = 8538627
IGNORE_LINE_Y = 462       # xe máy cao trung vị 25 px (ngưỡng KITTI) khi đáy box ở y≈475 -> tâm ở 462
KEYFRAME_EVERY = 5

ann = json.load(open(BACKUP, encoding="utf-8"))
gt = np.loadtxt(S + "/mot_export/gt/gt.txt", delimiter=",")
ext = json.load(open(S + "/ext.json"))
log = []


def _match_original(mot_id: int) -> int:
    """Id track CVAT ứng với id MOT của bản xuất (khớp theo frame + box đầu tiên, trên annotation gốc)."""
    t = gt[gt[:, 1] == mot_id]
    f0, x, y = int(t[0, 0]) - 1, t[0, 2], t[0, 3]
    found = [tr["id"] for tr in ann["tracks"]
             if tr["shapes"][0]["frame"] == f0 and abs(tr["shapes"][0]["points"][0] - x) < 0.02
             and abs(tr["shapes"][0]["points"][1] - y) < 0.02]
    assert len(found) == 1, (mot_id, len(found))
    return found[0]


# ánh xạ tính một lần trước khi sửa (sửa xong box đầu của track đổi, không khớp lại được)
MOT_TO_CVAT = {int(i): _match_original(int(i)) for i in np.unique(gt[:, 1])}
CVAT_BY_ID = {tr["id"]: tr for tr in ann["tracks"]}


def cvat_track_of(mot_id: int) -> dict:
    return CVAT_BY_ID[MOT_TO_CVAT[mot_id]]


def mot_box(mot_id: int, mot_frame: int) -> list[float]:
    r = gt[(gt[:, 1] == mot_id) & (gt[:, 0] == mot_frame)]
    assert len(r) == 1, (mot_id, mot_frame)
    x, y, w, h = r[0, 2:6]
    return [float(x), float(y), float(x + w), float(y + h)]


def shape(frame: int, points: list[float], outside: bool = False) -> dict:
    return {"type": "rectangle", "frame": frame, "occluded": False, "outside": outside, "z_order": 0,
            "rotation": 0.0, "points": [round(p, 2) for p in points], "attributes": []}


def set_keyframe(track: dict, frame: int, points: list[float]) -> None:
    track["shapes"] = [s for s in track["shapes"] if s["frame"] != frame] + [shape(frame, points)]
    track["shapes"].sort(key=lambda s: s["frame"])
    track["frame"] = track["shapes"][0]["frame"]


# 1. Vùng ignore: thay box nhỏ cuối đại lộ bằng dải phía trên dòng y = 462 (cả khung hình)
ignore = [tr for tr in ann["tracks"] if tr["label_id"] == IGNORE_LABEL_ID]
assert len(ignore) == 1
old = [s["points"] for s in ignore[0]["shapes"]]
for s in ignore[0]["shapes"]:
    s["points"] = [0.0, 0.0, 1920.0, float(IGNORE_LINE_Y)]
log.append({"sửa": "vùng ignore cuối đại lộ -> dải y < 462", "trước": old})

# 2. Kéo dài ngược 20 track bắt đầu muộn (so khớp mẫu NCC, đã kiểm bằng mắt), làm mượt rồi lấy keyframe mỗi 5 frame
for mot_id, e in ext.items():
    boxes = np.array(e["boxes"])  # frame MOT, x, y, w, h, score
    if len(boxes) == 0:
        continue
    k = 7
    pad = lambda v: np.convolve(np.pad(v, k // 2, mode="edge"), np.ones(k) / k, mode="valid")
    cx, cy = pad(boxes[:, 1] + boxes[:, 3] / 2), pad(boxes[:, 2] + boxes[:, 4] / 2)
    w, h = pad(boxes[:, 3]), pad(boxes[:, 4])
    track = cvat_track_of(int(mot_id))
    first_existing = track["shapes"][0]["frame"]
    picks = list(range(0, len(boxes), KEYFRAME_EVERY))
    picks = [i for i in picks if first_existing - (int(boxes[i, 0]) - 1) >= 3]
    for i in picks:
        set_keyframe(track, int(boxes[i, 0]) - 1, [cx[i] - w[i] / 2, cy[i] - h[i] / 2, cx[i] + w[i] / 2, cy[i] + h[i] / 2])
    log.append({"sửa": f"kéo dài ngược track MOT {mot_id}", "từ frame MOT": int(boxes[0, 0]),
                "đến": e["start"], "keyframe": len(picks), "dừng vì": e["stop"]})

# 3. Nối chỗ đứt mà xe vẫn nhìn thấy (đã kiểm bằng mắt); giữ 6 chỗ che khuất thật: 28, 36, 78, 99, 131, 185
GAPS_TO_FILL = [(20, 15, 18), (21, 32, 37), (21, 149, 158), (34, 57, 64), (42, 218, 220), (49, 195, 197),
                (52, 88, 90), (54, 237, 239), (55, 326, 328), (71, 271, 275), (93, 241, 244), (95, 343, 345),
                (98, 342, 345), (141, 578, 580)]
for mot_id, a, b in GAPS_TO_FILL:
    track = cvat_track_of(mot_id)
    removed = [s["frame"] for s in track["shapes"] if a - 1 < s["frame"] < b - 1]
    assert removed and all(s["outside"] for s in track["shapes"] if s["frame"] in removed), (mot_id, removed)
    track["shapes"] = [s for s in track["shapes"] if not (a - 1 < s["frame"] < b - 1)]
    set_keyframe(track, a - 1, mot_box(mot_id, a))  # giữ nguyên box trước chỗ đứt, nội suy thẳng tới frame b
    assert any(s["frame"] == b - 1 and not s["outside"] for s in track["shapes"]), mot_id
    log.append({"sửa": f"nối chỗ đứt track MOT {mot_id}", "frame MOT trống": [a + 1, b - 1]})

# 4. Track 180: từ f585 box cao dần tới 42 px (ôm thêm xe phía sau); giữ đáy, đặt chiều cao 28 px như đoạn trước
track = cvat_track_of(180)
for s in track["shapes"]:
    if s["frame"] >= 585:
        x1, _, x2, y2 = s["points"]
        before = list(s["points"])
        s["points"] = [x1, y2 - 28.0, x2, y2]
        log.append({"sửa": "track MOT 180 chiều cao box", "frame CVAT": s["frame"], "trước": before, "sau": s["points"]})

# 5. Track 99 (ô tô vào từ mép phải): keyframe đo trên ảnh, thay đoạn nội suy lệch f242–f250
track = cvat_track_of(99)
CAR99 = {239: [1898, 812, 1920, 878], 241: [1855, 789, 1920, 872], 243: [1812, 765, 1920, 870],
         245: [1771, 742, 1920, 862], 247: [1741, 722, 1920, 838], 249: [1729, 708, 1901, 808]}
for frame, points in CAR99.items():
    set_keyframe(track, frame, [float(p) for p in points])
log.append({"sửa": "track MOT 99 keyframe đo tay", "frame CVAT": list(CAR99)})

# 6. Vùng ignore nhỏ cho chỗ mơ hồ (không chỉ ra được xe nằm từ đâu đến đâu)
# mỗi vùng: (lý do, {frame CVAT: box keyframe}); keyframe cuối < 599 thì vùng kết thúc ngay sau đó
NEW_IGNORE = [
    ("2 xe máy trong cụm dày sát dòng 462 cuối clip (bám theo xe, tránh track 175/180 và ô tô 111)",
     {565: [884, 448, 903, 476], 585: [868, 450, 896, 492], 599: [866, 455, 890, 505]}),
    ("xe máy sát dòng 462 cuối clip, bị xe khác che", {583: [836, 452, 850, 486], 599: [836, 452, 850, 486]}),
    ("xe máy chiều đối diện lộ từng mảnh sau tán cây, mép phải", {142: [1885, 612, 1920, 660], 158: [1885, 612, 1920, 660]}),
]
for reason, keyframes in NEW_IGNORE:
    frames = sorted(keyframes)
    shapes = [shape(f, [float(p) for p in keyframes[f]]) for f in frames]
    if frames[-1] != 599:
        shapes.append(shape(frames[-1] + 1, [float(p) for p in keyframes[frames[-1]]], outside=True))
    ann["tracks"].append({"label_id": IGNORE_LABEL_ID, "frame": frames[0], "group": 0, "source": "manual",
                          "shapes": shapes, "attributes": [], "elements": []})
    log.append({"sửa": "thêm vùng ignore", "lý do": reason, "keyframe CVAT": keyframes})

# 7. Box "mảnh vụn": xe gần như bị che hẳn (sau cây/bụi), box không còn chỉ ra được xe -> outside, giữ ID khi xe hiện lại.
# Ứng viên lấy bằng luật kích thước (căn diện tích < 0,45 x lớn nhất trong ±20 frame), rồi XEM TỪNG ĐOẠN BẰNG MẮT:
# chỉ ẩn đoạn không nhận ra được xe; đoạn xe vẫn nhìn thấy (dù bị che một phần) thì giữ.
# Khoá = thứ tự track trong annotation (đánh từ 1, như interp.py), giá trị = frame MOT (đánh từ 1).
R = lambda a, b: list(range(a, b + 1))
HIDE = {9: [34], 23: [248], 25: [1, *R(48, 56)], 29: [*R(52, 57), 78], 33: [*R(90, 96), *R(136, 143)], 35: [202],
        65: R(262, 265), 79: [*R(385, 391), *R(441, 448)], 105: R(93, 95), 106: [126], 110: [193], 111: [201, 206, 207],
        112: [205], 115: [223], 116: [263], 119: [301], 120: [311], 121: [337], 122: [371], 125: R(428, 430),
        126: [431], 128: [443], 130: [*R(457, 459), 462], 131: [467], 135: [495, 496], 136: [497], 138: [509],
        140: R(512, 514), 141: [521], 142: [527, 528], 143: [*R(553, 556), 559], 144: [587, 588], 145: [595, 596],
        154: [131], 156: [206], 160: [258, 259], 161: [261, 262], 162: [316, 317], 163: [325, 326], 167: [289, 290],
        168: [307, 308], 173: [438, 439], 174: [449], 176: [482], 180: [520], 182: [533, 534], 183: [542, 543]}


def track_boxes(track: dict) -> dict[int, list[float]]:
    """Box từng frame CVAT của track, nội suy tuyến tính như CVAT (đã kiểm khớp bản xuất tới 0,01 px)."""
    sh = sorted(track["shapes"], key=lambda s: s["frame"])
    out = {}
    for i, s in enumerate(sh):
        if s["outside"]:
            continue
        nxt = sh[i + 1] if i + 1 < len(sh) else None
        for f in range(s["frame"], nxt["frame"] if nxt else 600):
            if nxt and not nxt["outside"]:
                r = (f - s["frame"]) / (nxt["frame"] - s["frame"])
                out[f] = [a + (b - a) * r for a, b in zip(s["points"], nxt["points"])]
            else:
                out[f] = list(s["points"])
    return out


def hide_frames(track: dict, hidden: set[int]) -> None:
    """Đánh dấu outside các frame CVAT trong `hidden`, không đổi box ở các frame còn lại."""
    boxes = track_boxes(track)
    visible = sorted(f for f in boxes if f not in hidden)
    keep = [s for s in track["shapes"] if s["frame"] not in hidden]
    keyed = {s["frame"] for s in keep}
    new = []
    for f in visible:
        prev_hidden, next_hidden = (f - 1) in hidden, (f + 1) in hidden
        if (prev_hidden or next_hidden) and f not in keyed:
            new.append(shape(f, boxes[f]))
        if next_hidden:
            new.append(shape(f + 1, boxes[f], outside=True))
    # đoạn ẩn ở đầu track: frame nhìn thấy đầu tiên phải là keyframe (đã thêm ở trên nếu kề frame ẩn)
    shapes = {s["frame"]: s for s in keep}
    for s in new:
        if s["frame"] not in shapes or s["outside"]:
            shapes[s["frame"]] = s
    track["shapes"] = [shapes[f] for f in sorted(shapes)]
    # bỏ outside thừa đứng trước mọi box nhìn thấy, rồi cập nhật frame bắt đầu
    while track["shapes"] and track["shapes"][0]["outside"]:
        track["shapes"].pop(0)
    track["frame"] = track["shapes"][0]["frame"]
    after = track_boxes(track)
    assert set(after) == set(visible) and all(np.allclose(after[f], boxes[f], atol=0.02) for f in visible)


for tid, mot_frames in HIDE.items():
    track = ann["tracks"][tid - 1]
    hide_frames(track, {f - 1 for f in mot_frames})
    log.append({"sửa": "box mảnh vụn -> outside (kiểm bằng mắt)", "track CVAT": track.get("id"), "frame MOT": mot_frames})

# 8. Box lệch phát hiện qua bước nhảy giữa 2 frame liền nhau (đã xem bằng mắt):
#  - track 44 (xe máy áo trắng): f117–118 box cao hơn xe ~9 px rồi nhảy về đúng ở f119 -> keyframe lấy bằng so khớp
#    mẫu lùi từ box đúng ở f119 (NCC 0,79–0,96);
#  - track 45: keyframe f203 cao 76 px, ôm cả xe 51 phía sau -> box đúng nối liền với f204.
BOX_FIXES = {
    44: {"drop": [116, 117], "set": {110: [855, 491, 870, 522], 113: [853, 495, 868, 526],
                                     115: [851, 497, 866, 528], 117: [850, 500, 865, 531]}},
    45: {"drop": [202], "set": {202: [791, 568, 816, 614]}},
}
for idx, fix in BOX_FIXES.items():
    track = ann["tracks"][idx - 1]
    track["shapes"] = [s for s in track["shapes"] if s["frame"] not in fix["drop"]]
    for frame, points in fix["set"].items():
        set_keyframe(track, frame, [float(p) for p in points])
    log.append({"sửa": "box lệch -> keyframe mới", "track CVAT": track.get("id"), "frame CVAT": fix})

json.dump(ann, open(S + "/edited.json", "w", encoding="utf-8"), ensure_ascii=False)
json.dump(log, open(S + "/edit_log.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(log), "chỉnh sửa;", len(ann["tracks"]), "track")
