"""Rà soát nhãn quỹ đạo CVAT (JSON của API) cho clip dev: box đứng yên, box nhảy, track trùng, chỗ hở;
nếu có file dự đoán YOLO26x thì thêm: xe có thể bị sót, box có thể lệch/thừa."""
import json
import sys
from collections import defaultdict

LINE_Y = 637
LAST_FRAME = 599
IGNORE_LABEL_ID = 8538627
NAMES = {8538623: "motorbike", 8538624: "car", 8538625: "bus", 8538626: "truck", 8538627: "ignore"}


def frames_of(track: dict) -> dict[int, list[float]]:
    """Box từng frame theo cách CVAT nội suy: tuyến tính giữa keyframe, ẩn sau keyframe outside,
    keyframe hiện cuối cùng thì giữ nguyên tới hết clip."""
    shapes = sorted(track["shapes"], key=lambda s: s["frame"])
    boxes = {}
    for cur, nxt in zip(shapes, shapes[1:] + [None]):
        if cur["outside"]:
            continue
        end = nxt["frame"] if nxt else LAST_FRAME + 1
        for f in range(cur["frame"], end):
            if nxt is None:
                boxes[f] = cur["points"]
            else:
                w = (f - cur["frame"]) / (nxt["frame"] - cur["frame"])
                boxes[f] = [a + (b - a) * w for a, b in zip(cur["points"], nxt["points"])]
    return boxes


def frac_below(b) -> float:
    h = b[3] - b[1]
    return max(0.0, b[3] - max(b[1], LINE_Y)) / h if h > 0 else 0.0


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def inside_frac(a, region) -> float:
    ix = max(0.0, min(a[2], region[2]) - max(a[0], region[0]))
    iy = max(0.0, min(a[3], region[3]) - max(a[1], region[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return ix * iy / area if area > 0 else 0.0


def spans(frames: list[int], max_step: int) -> list[tuple[int, int]]:
    out = []
    for f in sorted(frames):
        if out and f - out[-1][1] <= max_step:
            out[-1][1] = f
        else:
            out.append([f, f])
    return [tuple(s) for s in out]


def main(ann_path: str, det_path: str | None) -> None:
    ann = json.load(open(ann_path, encoding="utf-8"))
    tracks = [t for t in ann["tracks"] if t["label_id"] != IGNORE_LABEL_ID]
    ignore_tracks = [t for t in ann["tracks"] if t["label_id"] == IGNORE_LABEL_ID]
    per_track = {t["id"]: frames_of(t) for t in tracks}
    ignore_boxes = defaultdict(list)
    for t in ignore_tracks:
        for f, b in frames_of(t).items():
            if b[3] > LINE_Y + 5:  # vùng ignore người gán tự vẽ (không tính dải trên vạch)
                ignore_boxes[f].append(b)
    short = {t["id"]: str(t["id"])[-3:] for t in tracks}
    print(f"{len(tracks)} track xe, {len(ignore_tracks)} track ignore")

    print("\n[A] Box giữ nguyên tới cuối clip (keyframe cuối không outside) và nằm dưới vạch:")
    for t in tracks:
        shapes = sorted(t["shapes"], key=lambda s: s["frame"])
        last = shapes[-1]
        if not last["outside"] and last["frame"] < LAST_FRAME - 3 and frac_below(last["points"]) >= 0.5:
            print(f"  {short[t['id']]} {NAMES[t['label_id']]} keyframe cuối {last['frame']}, "
                  f"{len(shapes)} keyframe, box {[round(v) for v in last['points']]}")

    print("\n[B] Box nhảy giữa hai keyframe liên tiếp (dưới vạch):")
    for t in tracks:
        shapes = [s for s in sorted(t["shapes"], key=lambda s: s["frame"])]
        for a, b in zip(shapes, shapes[1:]):
            if a["outside"] or b["outside"]:
                continue
            pa, pb = a["points"], b["points"]
            if max(frac_below(pa), frac_below(pb)) < 0.5:
                continue
            h = max(pa[3] - pa[1], pb[3] - pb[1])
            dx = ((pb[0] + pb[2]) - (pa[0] + pa[2])) / 2
            dy = ((pb[1] + pb[3]) - (pa[1] + pa[3])) / 2
            step = (dx * dx + dy * dy) ** 0.5 / (b["frame"] - a["frame"])
            size_ratio = max((pb[3] - pb[1]) / (pa[3] - pa[1]), (pa[3] - pa[1]) / (pb[3] - pb[1]))
            if step > 0.3 * h or (size_ratio > 1.5 and b["frame"] - a["frame"] <= 5):
                print(f"  {short[t['id']]} {a['frame']}->{b['frame']}: dời {step:.1f} px/frame "
                      f"(box cao {h:.0f}), tỉ lệ cỡ {size_ratio:.2f}")

    print("\n[C] Hai track trùng nhau (IoU > 0.5, dưới vạch) >= 3 frame:")
    pair_frames = defaultdict(list)
    by_frame = defaultdict(list)
    for tid, fb in per_track.items():
        for f, b in fb.items():
            if frac_below(b) >= 0.5:
                by_frame[f].append((tid, b))
    for f, items in by_frame.items():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if iou(items[i][1], items[j][1]) > 0.5:
                    pair_frames[(items[i][0], items[j][0])].append(f)
    for (a, b), fs in sorted(pair_frames.items(), key=lambda kv: -len(kv[1])):
        if len(fs) >= 3:
            print(f"  {short[a]} & {short[b]}: {spans(fs, 1)}")

    print("\n[D] Chỗ hở (outside rồi hiện lại) dưới vạch:")
    for t in tracks:
        shapes = sorted(t["shapes"], key=lambda s: s["frame"])
        for i in range(1, len(shapes) - 1):
            if shapes[i]["outside"] and not shapes[i - 1]["outside"]:
                j = i + 1
                while j < len(shapes) and shapes[j]["outside"]:
                    j += 1
                if j < len(shapes) and max(frac_below(shapes[i - 1]["points"]), frac_below(shapes[j]["points"])) >= 0.5:
                    print(f"  {short[t['id']]} hở {shapes[i]['frame']}->{shapes[j]['frame']}")

    print("\n[E] Track rất ngắn (< 8 frame có box dưới vạch):")
    for t in tracks:
        n = sum(1 for b in per_track[t["id"]].values() if frac_below(b) >= 0.5)
        if 0 < n < 8:
            print(f"  {short[t['id']]} {n} frame")

    if not det_path:
        return
    dets = {int(k): v for k, v in json.load(open(det_path)).items()}
    print(f"\n[F] Xe có thể bị sót (YOLO26x conf >= 0.5, dưới vạch, IoU < 0.3 với mọi box nhãn) — {len(dets)} frame soát:")
    missing = []
    for f, ds in dets.items():
        gts = [b for fb in per_track.values() if (b := fb.get(f)) is not None]
        for cls, x1, y1, x2, y2, conf in ds:
            d = [x1, y1, x2, y2]
            if conf < 0.5 or frac_below(d) < 0.5:
                continue
            if any(inside_frac(d, r) >= 0.5 for r in ignore_boxes[f]):
                continue
            if max((iou(d, g) for g in gts), default=0) < 0.3:
                missing.append((f, d, conf, cls))
    # nối các phát hiện liền nhau thành "xe" để báo theo đoạn
    groups = []
    for f, d, conf, cls in sorted(missing, key=lambda m: m[0]):
        for g in groups:
            lf, ld = g[-1][0], g[-1][1]
            if 0 < f - lf <= 9 and iou(d, ld) > 0.2:
                g.append((f, d, conf, cls))
                break
        else:
            groups.append([(f, d, conf, cls)])
    for g in sorted(groups, key=lambda g: -len(g)):
        f0, d0 = g[0][0], g[0][1]
        print(f"  frame {f0}->{g[-1][0]} ({len(g)} lần), box đầu {[round(v) for v in d0]}, "
              f"conf max {max(x[2] for x in g):.2f}, lớp {sorted({x[3] for x in g})}")
    json.dump([[(x[0], x[1], x[2], x[3]) for x in g] for g in groups], open(det_path + ".missing.json", "w"))

    print("\n[G] Box nhãn không khớp phát hiện nào (IoU < 0.3 với mọi dự đoán conf >= 0.1), dưới vạch:")
    flagged = defaultdict(list)
    for f, ds in dets.items():
        preds = [d[1:5] for d in ds]
        for tid, fb in per_track.items():
            b = fb.get(f)
            if b is None or frac_below(b) < 0.5:
                continue
            if max((iou(b, p) for p in preds), default=0) < 0.3:
                flagged[tid].append(f)
    for tid, fs in sorted(flagged.items(), key=lambda kv: -len(kv[1])):
        total = sum(1 for f in dets if (b := per_track[tid].get(f)) is not None and frac_below(b) >= 0.5)
        print(f"  {short[tid]}: {len(fs)}/{total} frame soát, đoạn {spans(fs, 3)}")
    json.dump({str(k): v for k, v in flagged.items()}, open(det_path + ".flagged.json", "w"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
