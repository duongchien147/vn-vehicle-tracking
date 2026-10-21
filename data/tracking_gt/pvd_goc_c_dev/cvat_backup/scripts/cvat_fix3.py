"""Đợt sửa 3 trên job CVAT clip dev (trợ lý AI làm theo yêu cầu người gán, đã xem ảnh từng box). Chạy: cvat_fix3.py [--apply].

- 267 (mũ xanh lá) và 274 (mũ trắng), frame 86–128: hai xe chạy song song rồi nối đuôi; box cũ của 267 ôm cả hai xe,
  274 chỉ có box nội suy. Box mới đọc bằng mắt trên ảnh phóng to có lưới toạ độ (keyframe mỗi 6 frame).
- 8 xe ở đường bên trái (chiều ngược lại, dưới hàng cây) chưa có nhãn: box lấy từ YOLO26x COCO từng frame
  (không phải mô hình của dự án), lớp do người duyệt quyết định, bỏ box phình to vì lá che; source = semi-auto.
- 308 (ô tô tối màu đường bên trái): thêm frame 392–395 (xe đã ra khỏi tán cây nhưng track bắt đầu ở 396).
"""
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cvat_api import HOST, JOB_ID, session  # noqa: E402

SP = Path(__file__).parent
PREFIX = 28362000
LABEL_ID = {"motorbike": 8538623, "car": 8538624}
LAST_FRAME = 599


def shape(frame: int, box: list[float], outside: bool = False) -> dict:
    return {"type": "rectangle", "frame": frame, "outside": outside, "occluded": False, "z_order": 0,
            "rotation": 0.0, "points": [float(v) for v in box], "attributes": []}


def replace_range(track: dict, lo: int, hi: int, boxes: dict[int, list[float]]) -> None:
    """Bỏ keyframe trong [lo, hi], chèn keyframe mới."""
    kept = [s for s in track["shapes"] if not lo <= s["frame"] <= hi]
    track["shapes"] = sorted(kept + [shape(f, b) for f, b in boxes.items()], key=lambda s: s["frame"])
    track["frame"] = track["shapes"][0]["frame"]


def main(apply: bool) -> None:
    s = session()
    url = f"{HOST}/api/jobs/{JOB_ID}/annotations"
    ann = s.get(url, timeout=60).json()
    Path(f"data/tracking_gt/pvd_goc_c_dev/cvat_backup/before_fix3_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(
        json.dumps(ann, ensure_ascii=False), encoding="utf-8")
    tracks = {t["id"]: t for t in ann["tracks"]}

    pair = json.load(open(SP / "pair_manual.json"))
    updated = []
    for key, tid in (("green_267", PREFIX + 267), ("white_274", PREFIX + 274)):
        t = copy.deepcopy(tracks[tid])
        boxes = {int(f): b for f, b in pair[key].items()}
        replace_range(t, 86, 128, boxes)
        updated.append(t)
        print(f"{tid}: keyframe 86–128 thay bằng {sorted(boxes)}")

    left = json.load(open(SP / "left_new.json"))
    t308 = copy.deepcopy(tracks[PREFIX + 308])
    assert min(x["frame"] for x in t308["shapes"]) == 396, "308 đã đổi"
    t308["shapes"] = [shape(int(f), b) for f, b in left["extend_308"].items()] + t308["shapes"]
    t308["frame"] = 392
    updated.append(t308)
    print("308: thêm 392–395")

    created = []
    for v in left["new"]:
        boxes = {int(f): b for f, b in v["boxes"].items()}
        last = max(boxes)
        shapes = [shape(f, b) for f, b in sorted(boxes.items())]
        if last < LAST_FRAME:
            shapes.append(shape(last + 1, boxes[last], outside=True))
        created.append({"frame": min(boxes), "label_id": LABEL_ID[v["label"]], "group": 0, "source": "semi-auto",
                        "attributes": [], "shapes": shapes, "elements": []})
        print(f"tạo {v['label']} {min(boxes)}–{last} ({len(boxes)} keyframe)")

    if not apply:
        return
    for action, tracks_payload in (("update", updated), ("create", created)):
        r = s.patch(url, params={"action": action},
                    json={"version": ann["version"], "tags": [], "shapes": [], "tracks": tracks_payload}, timeout=120)
        print("PATCH", action, r.status_code)
        r.raise_for_status()


if __name__ == "__main__":
    main("--apply" in sys.argv)
