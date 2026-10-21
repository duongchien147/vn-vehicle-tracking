"""Đợt sửa 2 trên job CVAT clip dev, sau khi rà soát (đã xem ảnh từng chỗ). Chạy: cvat_fix2.py [--apply].

- 277: trả lại outside ở 333 (xe khuất sau biển CẤM; lần lấp trước làm box đè lên xe của 287) và bỏ đoạn 350–351
  (box trùng trên xe của 287).
- 301 + 311: cùng một ô tô (đổi ID ở 420, hai box trùng 420–430) -> giữ 301 tới 419, nối phần 311 từ 420, xoá 311.
- 248 + 306: cùng một xe tay ga đỗ trên vỉa hè (359–377 chỉ bị người đi bộ che một phần) -> nối 306 vào 248, xoá 306.
- 304: box trùng với xe đỗ 460 (người gán đã vẽ từ frame 0) -> xoá.
- 613 (xe mũ đen, người gán vẽ lại từ 147): nối lại đoạn 0–138 của track cũ 28373612 đã bị xoá nhầm.
- 278 (xe áo đỏ sẫm, người gán vẽ lại từ 138): nối lại đoạn 46–136 của track cũ 28362264 đã bị xoá nhầm.
- 268 + 274: cùng xe mũ trắng (268 tới 90, 274 từ 129) -> gộp; đoạn 91–128 CVAT nội suy, người gán chỉnh lại.
"""
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cvat_api import HOST, JOB_ID, session  # noqa: E402

OLD_BACKUP = "data/tracking_gt/pvd_goc_c_dev/cvat_backup/before_fix_20261014_222314.json"  # còn 2 track bị xoá
PREFIX = 28362000  # id track trên server = PREFIX + 3 số cuối (với các track của nhãn sơ bộ)


def sorted_shapes(track: dict) -> list[dict]:
    return sorted(track["shapes"], key=lambda s: s["frame"])


def strip_id(shape: dict) -> dict:
    shape = copy.deepcopy(shape)
    shape.pop("id", None)
    return shape


def main(apply: bool) -> None:
    s = session()
    url = f"{HOST}/api/jobs/{JOB_ID}/annotations"
    ann = s.get(url, timeout=60).json()
    Path(f"data/tracking_gt/pvd_goc_c_dev/cvat_backup/before_fix2_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(
        json.dumps(ann, ensure_ascii=False), encoding="utf-8")
    tracks = {t["id"]: t for t in ann["tracks"]}
    t = {k: copy.deepcopy(tracks.get(PREFIX + k)) for k in (277, 301, 311, 248, 306, 304)}
    if any(v is None for v in t.values()):
        sys.exit(f"thiếu track: {[k for k, v in t.items() if v is None]}")
    updated, deleted = [], []

    # 277: kết thúc ở 333
    sh = sorted_shapes(t[277])
    assert [x["frame"] for x in sh if x["frame"] >= 332] == [332, 350, 351, 352], "277 đã đổi"
    end = strip_id(sh[[x["frame"] for x in sh].index(332)])
    end.update(frame=333, outside=True)
    t[277]["shapes"] = [x for x in sh if x["frame"] <= 332] + [end]
    updated.append(t[277])
    print("277: outside ở 333, bỏ 350–352")

    # 301 + 311
    sh301, sh311 = sorted_shapes(t[301]), sorted_shapes(t[311])
    assert sh311[0]["frame"] == 420 and any(x["frame"] == 419 and not x["outside"] for x in sh301), "301/311 đã đổi"
    t[301]["shapes"] = [x for x in sh301 if x["frame"] <= 419] + [strip_id(x) for x in sh311]
    updated.append(t[301])
    deleted.append(t[311])
    print(f"301: giữ tới 419 + {len(sh311)} keyframe của 311 (420–{sh311[-1]['frame']}); xoá 311")

    # 248 + 306
    sh248, sh306 = sorted_shapes(t[248]), sorted_shapes(t[306])
    assert sh248[-1]["frame"] == 359 and sh248[-1]["outside"] and sh306[0]["frame"] == 378, "248/306 đã đổi"
    t[248]["shapes"] = sh248[:-1] + [strip_id(x) for x in sh306]
    updated.append(t[248])
    deleted.append(t[306])
    print(f"248: bỏ outside 359, nối {len(sh306)} keyframe của 306 (378–{sh306[-1]['frame']}); xoá 306")

    deleted.append(t[304])
    print("xoá 304 (trùng xe đỗ 460)")

    old = {x["id"]: x for x in json.load(open(OLD_BACKUP, encoding="utf-8"))["tracks"]}
    for keep_id, old_id, last_old in ((28373613, 28373612, 138), (PREFIX + 278, 28362264, 136)):
        keep = copy.deepcopy(tracks[keep_id])
        first = sorted_shapes(keep)[0]["frame"]
        assert first > last_old and old_id not in tracks, f"{keep_id} đã đổi"
        head = [strip_id(x) for x in sorted_shapes(old[old_id]) if not x["outside"] and x["frame"] <= last_old]
        keep["shapes"] = head + sorted_shapes(keep)
        keep["frame"] = head[0]["frame"]
        updated.append(keep)
        print(f"{keep_id}: nối {len(head)} keyframe {head[0]['frame']}–{head[-1]['frame']} của track cũ {old_id}")

    t268, t274 = copy.deepcopy(tracks[PREFIX + 268]), copy.deepcopy(tracks[PREFIX + 274])
    sh268, sh274 = sorted_shapes(t268), sorted_shapes(t274)
    assert sh268[-1]["frame"] == 91 and sh268[-1]["outside"] and sh274[0]["frame"] == 129, "268/274 đã đổi"
    t274["shapes"] = [strip_id(x) for x in sh268[:-1]] + sh274
    t274["frame"] = sh268[0]["frame"]
    updated.append(t274)
    deleted.append(t268)
    print("274: nối 268 (62–90) vào trước, xoá 268")

    if not apply:
        return
    r = s.patch(url, params={"action": "update"},
                json={"version": ann["version"], "tags": [], "shapes": [], "tracks": updated}, timeout=120)
    print("PATCH update", r.status_code)
    r.raise_for_status()
    r = s.patch(url, params={"action": "delete"},
                json={"version": ann["version"], "tags": [], "shapes": [], "tracks": deleted}, timeout=120)
    print("PATCH delete", r.status_code)
    r.raise_for_status()


if __name__ == "__main__":
    main("--apply" in sys.argv)
