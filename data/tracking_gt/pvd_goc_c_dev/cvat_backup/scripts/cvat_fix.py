"""Lấp chỗ hở và sửa 3 track trên job CVAT clip dev (đã kiểm từng chỗ bằng mắt). Chạy: cvat_fix.py [--apply]."""
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cvat_api import HOST, JOB_ID, session  # noqa: E402

# (track, frame bắt đầu hở): xe vẫn thấy rõ suốt đoạn hở -> bỏ keyframe outside để CVAT nội suy
FILL = [
    (28362242, 157), (28362246, 73), (28362255, 6), (28362255, 61), (28362263, 155), (28362267, 88),
    (28362267, 227), (28362274, 204), (28362277, 183), (28362277, 196), (28362277, 333), (28362280, 221),
    (28362281, 262), (28362287, 345), (28362287, 351), (28362292, 239), (28362293, 393), (28362294, 246),
    (28362294, 250), (28362294, 254), (28362295, 427), (28362296, 262), (28362301, 418), (28362310, 527),
    (28362311, 426), (28362326, 538),
]
TRACK_275_END = (28362275, 149)  # 275 nhảy từ xe chở hàng đỏ sang xe áo trắng (đã có 263) -> kết thúc ở 149
MOVE_280_TO_281 = (28362280, 149, 150, 28362281)  # box ở 149 là xe mũ xanh ngọc = track 281
ACCIDENTAL = (28373613, (148, 149))  # keyframe bị kéo lệch 60 px rồi outside 1 frame


def by_frame(track):
    return {s["frame"]: s for s in track["shapes"]}


def fill_gap(track, a) -> str:
    sh = sorted(track["shapes"], key=lambda s: s["frame"])
    idx = [i for i, s in enumerate(sh) if s["frame"] == a]
    if not idx or not sh[idx[0]]["outside"] or idx[0] == 0 or sh[idx[0] - 1]["outside"]:
        return "bỏ qua (đã đổi)"
    i = idx[0]
    j = i
    while j < len(sh) and sh[j]["outside"]:
        j += 1
    if j == len(sh):
        return "bỏ qua (không hiện lại)"
    removed = [s["frame"] for s in sh[i:j]]
    track["shapes"] = sh[:i] + sh[j:]
    return f"xoá outside {removed}, nội suy {sh[i - 1]['frame']} -> {sh[j]['frame']}"


def end_track(track, frame) -> str:
    sh = sorted(track["shapes"], key=lambda s: s["frame"])
    keep = [s for s in sh if s["frame"] < frame]
    end = next((s for s in sh if s["frame"] == frame), None)
    if end is None or not end["outside"]:
        return "bỏ qua (đã đổi)"
    track["shapes"] = keep + [end]
    return f"kết thúc ở {frame}, bỏ {[s['frame'] for s in sh if s['frame'] > frame]}"


def main(apply: bool) -> None:
    s = session()
    url = f"{HOST}/api/jobs/{JOB_ID}/annotations"
    ann = s.get(url, timeout=60).json()
    backup = Path("data/tracking_gt/pvd_goc_c_dev/cvat_backup") / f"before_fix_{time.strftime('%Y%m%d_%H%M%S')}.json"
    backup.write_text(json.dumps(ann, ensure_ascii=False), encoding="utf-8")
    tracks = {t["id"]: t for t in ann["tracks"]}
    changed = {}

    def get(tid):
        if tid not in changed:
            changed[tid] = copy.deepcopy(tracks[tid])
        return changed[tid]

    for tid, a in FILL:
        print("lấp", tid, a, fill_gap(get(tid), a))
    print("kết thúc 275", end_track(get(TRACK_275_END[0]), TRACK_275_END[1]))

    src_id, f_box, f_out, dst_id = MOVE_280_TO_281
    src, dst = get(src_id), get(dst_id)
    fs = by_frame(src)
    if f_box in fs and not fs[f_box]["outside"] and fs.get(f_out, {}).get("outside") and dst["frame"] > f_box:
        box = copy.deepcopy(fs[f_box])
        box.pop("id", None)
        dst["shapes"] = [box] + dst["shapes"]
        dst["frame"] = f_box
        src["shapes"] = [x for x in src["shapes"] if x["frame"] not in (f_box, f_out)]
        src["frame"] = min(x["frame"] for x in src["shapes"])
        print(f"chuyển box frame {f_box} từ {src_id} sang {dst_id}; {src_id} giờ bắt đầu ở {src['frame']}")
    else:
        print("bỏ qua chuyển 280 -> 281 (đã đổi)")

    tid, frames = ACCIDENTAL
    t = get(tid)
    fs = by_frame(t)
    if all(f in fs for f in frames) and fs[frames[1]]["outside"]:
        t["shapes"] = [x for x in t["shapes"] if x["frame"] not in frames]
        print("bỏ keyframe kéo nhầm", tid, frames)
    else:
        print("bỏ qua keyframe kéo nhầm (đã đổi)")

    payload = {"version": ann["version"], "tags": [], "shapes": [], "tracks": list(changed.values())}
    print("số track thay đổi:", len(changed))
    if not apply:
        return
    r = s.patch(url, params={"action": "update"}, json=payload, timeout=120)
    print("PATCH", r.status_code)
    r.raise_for_status()


if __name__ == "__main__":
    main("--apply" in sys.argv)
