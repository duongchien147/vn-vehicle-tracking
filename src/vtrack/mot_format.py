"""Đọc/ghi file tracking định dạng MOTChallenge (frame,id,x,y,w,h,...) và lấp chỗ đứt của track nhãn chuẩn.

Định dạng chuẩn (MOT17): frame đánh số từ 1; file nhãn chuẩn có 9 cột
`frame,id,x,y,w,h,được_tính,lớp,độ_hiển_thị`; file kết quả tracker có 10 cột `frame,id,x,y,w,h,conf,-1,-1,-1`.
"""

from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

# Bộ chấm MOT17 của TrackEval chỉ chấm lớp 1 ("pedestrian"); ở dự án này lớp 1 nghĩa là "xe" (mọi loại xe).
MOT_TARGET_CLASS = 1
# Lớp 8 = "distractor" của MOT17: detection trùng với box này bị bỏ trước khi chấm (không đúng, không sai).
MOT_DISTRACTOR_CLASS = 8


@dataclass(frozen=True)
class MotBox:
    frame: int  # đánh số từ 1 như MOTChallenge
    track_id: int
    left: float
    top: float
    width: float
    height: float
    confidence: float = 1.0
    mot_class: int = MOT_TARGET_CLASS


def read_mot_file(path: str | Path, first_frame: int = 1) -> list[MotBox]:
    """Đọc file MOT (bỏ qua các cột sau conf). `first_frame` là số của frame đầu trong file:
    MOTChallenge đánh từ 1, nhưng có bộ dữ liệu đánh từ 0 (VnTraffic) -> chuyển về đánh từ 1."""
    boxes = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        values = [float(v) for v in line.split(",")]
        # Cột 8 là lớp ở file nhãn MOT17; file kết quả tracker và bộ không có lớp (VnTraffic) để -1 -> lớp được chấm
        has_class = len(values) > 7 and values[7] > 0
        boxes.append(MotBox(
            frame=int(values[0]) - first_frame + 1, track_id=int(values[1]),
            left=values[2], top=values[3], width=values[4], height=values[5],
            confidence=values[6] if len(values) > 6 else 1.0,
            mot_class=int(values[7]) if has_class else MOT_TARGET_CLASS,
        ))
    return boxes


def write_mot_ground_truth(path: str | Path, boxes: list[MotBox]) -> None:
    """Ghi nhãn chuẩn 9 cột MOT17. Cột "được tính" luôn là 1: box không muốn chấm thì đặt lớp distractor."""
    lines = [
        f"{b.frame},{b.track_id},{b.left:.2f},{b.top:.2f},{b.width:.2f},{b.height:.2f},1,{b.mot_class},1"
        for b in sorted(boxes, key=lambda b: (b.frame, b.track_id))
    ]
    _write_lines(path, lines)


def write_mot_results(path: str | Path, boxes: list[MotBox]) -> None:
    """Ghi kết quả tracker 10 cột MOTChallenge."""
    lines = [
        f"{b.frame},{b.track_id},{b.left:.2f},{b.top:.2f},{b.width:.2f},{b.height:.2f},{b.confidence:.4f},-1,-1,-1"
        for b in sorted(boxes, key=lambda b: (b.frame, b.track_id))
    ]
    _write_lines(path, lines)


def _write_lines(path: str | Path, lines: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")


def fill_track_gaps(boxes: list[MotBox], mot_class: int = MOT_DISTRACTOR_CLASS) -> list[MotBox]:
    """Box nội suy tuyến tính cho các frame bị thiếu ở giữa một track (trước và sau đều có nhãn).

    Dùng cho nhãn chuẩn bị rơi box giữa chừng trong khi xe vẫn nhìn thấy: nếu không lấp, tracker bám
    đúng xe ở các frame đó sẽ bị tính là phát hiện sai. Box lấp mặc định là lớp distractor (trung tính)
    chứ không phải nhãn thật, vì vị trí nội suy chỉ là ước lượng và có thể xe bị che thật.
    """
    by_track: dict[int, list[MotBox]] = defaultdict(list)
    for box in boxes:
        by_track[box.track_id].append(box)

    filled = []
    for track in by_track.values():
        track.sort(key=lambda b: b.frame)
        for before, after in zip(track, track[1:]):
            span = after.frame - before.frame
            for step in range(1, span):
                ratio = step / span
                filled.append(replace(
                    before,
                    frame=before.frame + step,
                    left=_lerp(before.left, after.left, ratio),
                    top=_lerp(before.top, after.top, ratio),
                    width=_lerp(before.width, after.width, ratio),
                    height=_lerp(before.height, after.height, ratio),
                    mot_class=mot_class,
                ))
    return filled


def resolve_duplicate_ids(boxes: list[MotBox], mot_class: int = MOT_DISTRACTOR_CLASS) -> list[MotBox]:
    """Sửa lỗi nhãn "một ID xuất hiện 2 lần trong cùng frame" (MOTChallenge cấm, TrackEval từ chối chấm).

    Giữ box khớp nhất (IoU cao nhất) với chính track đó ở frame gần nhất; box còn lại có thể là một xe thật
    bị gán nhầm ID nên không xoá mà đổi thành box trung tính với ID mới chưa dùng.
    """
    groups: dict[tuple[int, int], list[MotBox]] = defaultdict(list)
    for box in boxes:
        groups[(box.frame, box.track_id)].append(box)
    next_id = max((box.track_id for box in boxes), default=0) + 1

    resolved = []
    for (frame, track_id), same in groups.items():
        if len(same) == 1:
            resolved.extend(same)
            continue
        neighbours = [b for b in boxes if b.track_id == track_id and b.frame != frame]
        if neighbours:
            nearest = min(neighbours, key=lambda b: abs(b.frame - frame))
            same.sort(key=lambda b: box_iou(b, nearest), reverse=True)
        resolved.append(same[0])
        for extra in same[1:]:
            resolved.append(replace(extra, track_id=next_id, mot_class=mot_class))
            next_id += 1
    return resolved


def box_iou(a: MotBox, b: MotBox) -> float:
    overlap_w = max(0.0, min(a.left + a.width, b.left + b.width) - max(a.left, b.left))
    overlap_h = max(0.0, min(a.top + a.height, b.top + b.height) - max(a.top, b.top))
    intersection = overlap_w * overlap_h
    union = a.width * a.height + b.width * b.height - intersection
    return intersection / union if union > 0 else 0.0


def _lerp(start: float, end: float, ratio: float) -> float:
    return start + (end - start) * ratio


def read_cvat_mot_labeled(path: str | Path, label_names: list[str]) -> list[tuple[str, MotBox]]:
    """Đọc gt.txt do CVAT xuất ("MOT 1.1": cột 8 là số thứ tự nhãn trong labels.txt, đánh từ 1) kèm tên nhãn
    của từng box — cần cho chấm detector theo lớp; chấm tracking thì dùng read_cvat_mot."""
    labeled = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        values = [float(v) for v in line.split(",")]
        box = MotBox(frame=int(values[0]), track_id=int(values[1]),
                     left=values[2], top=values[3], width=values[4], height=values[5])
        labeled.append((label_names[int(values[7]) - 1], box))
    return labeled


def read_cvat_mot(path: str | Path, label_names: list[str], ignore_label: str) -> tuple[list[MotBox], list[MotBox]]:
    """Như read_cvat_mot_labeled nhưng cho chấm tracking: trả về (box xe — mọi lớp xe thành lớp được chấm,
    vùng ignore). Vùng ignore không đưa vào nhãn chuẩn mà dùng để bỏ box tracker nằm trong đó
    (xem drop_boxes_in_regions), như vùng bỏ qua của UA-DETRAC.
    """
    labeled = read_cvat_mot_labeled(path, label_names)
    vehicles = [box for label, box in labeled if label != ignore_label]
    ignore_regions = [box for label, box in labeled if label == ignore_label]
    return vehicles, ignore_regions


# Box có ít nhất tỉ lệ diện tích này nằm trong vùng ignore thì bỏ (cùng ngưỡng với bộ chấm COCO của dự án)
IGNORE_OVERLAP = 0.5


def drop_boxes_in_regions(
    boxes: list[MotBox], regions: list[MotBox], min_overlap: float = IGNORE_OVERLAP,
) -> list[MotBox]:
    """Bỏ box mà phần lớn diện tích nằm trong một vùng ignore cùng frame."""
    regions_by_frame: dict[int, list[MotBox]] = defaultdict(list)
    for region in regions:
        regions_by_frame[region.frame].append(region)
    return [
        box for box in boxes
        if not any(_inside_fraction(box, region) >= min_overlap for region in regions_by_frame[box.frame])
    ]


def _inside_fraction(box: MotBox, region: MotBox) -> float:
    overlap_w = max(0.0, min(box.left + box.width, region.left + region.width) - max(box.left, region.left))
    overlap_h = max(0.0, min(box.top + box.height, region.top + region.height) - max(box.top, region.top))
    area = box.width * box.height
    return overlap_w * overlap_h / area if area > 0 else 0.0
