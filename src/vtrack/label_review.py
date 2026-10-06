"""So khớp các box YOLO để rà soát nhãn do người gán: tìm box trùng nhau, box bất thường,
và so bộ nhãn người gán với nhãn sơ bộ / dự đoán của mô hình (để tìm xe có thể bị sót hoặc box có thể thừa)."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    center_x: float  # các toạ độ chuẩn hoá về [0, 1] theo kích thước ảnh
    center_y: float
    width: float
    height: float
    confidence: float = 1.0  # nhãn người gán = 1; dự đoán mô hình = độ tin cậy


@dataclass
class MatchResult:
    pairs: list[tuple[int, int]] = field(default_factory=list)  # (chỉ số trong reference, chỉ số trong candidate)
    unmatched_reference: list[int] = field(default_factory=list)
    unmatched_candidate: list[int] = field(default_factory=list)


# File đánh dấu một tập (thư mục <split>/) có nhãn đã được người kiểm: không script nào được ghi đè <split>/labels/
HUMAN_VERIFIED_MARKER = "HUMAN_VERIFIED.txt"


def is_human_verified(split_dir: Path) -> bool:
    return (split_dir / HUMAN_VERIFIED_MARKER).exists()


def read_yolo_file(path: Path) -> list[YoloBox]:
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            boxes.append(YoloBox(int(parts[0]), *map(float, parts[1:5])))
    return boxes


def read_scored_yolo_file(path: Path) -> list[YoloBox]:
    """Nhãn có điểm (6 cột: lớp, 4 toạ độ, điểm tin cậy), như đầu ra của gán nhãn tự động."""
    if not path.exists():
        return []
    rows = (line.split() for line in path.read_text().splitlines())
    return [YoloBox(int(r[0]), *map(float, r[1:6])) for r in rows if len(r) == 6]


def write_yolo_file(path: Path, boxes: list[YoloBox], with_confidence: bool = False) -> None:
    """Ghi nhãn YOLO 5 cột, hoặc 6 cột (thêm điểm tin cậy) nếu `with_confidence`."""
    lines = []
    for b in boxes:
        line = f"{b.class_id} {b.center_x:.6f} {b.center_y:.6f} {b.width:.6f} {b.height:.6f}"
        lines.append(f"{line} {b.confidence:.4f}" if with_confidence else line)
    path.write_text("\n".join(lines))


def iou(a: YoloBox, b: YoloBox) -> float:
    """Tỉ lệ giao trên hợp (Intersection over Union) của hai box."""
    overlap_w = min(a.center_x + a.width / 2, b.center_x + b.width / 2) - max(a.center_x - a.width / 2, b.center_x - b.width / 2)
    overlap_h = min(a.center_y + a.height / 2, b.center_y + b.height / 2) - max(a.center_y - a.height / 2, b.center_y - b.height / 2)
    if overlap_w <= 0 or overlap_h <= 0:
        return 0.0
    intersection = overlap_w * overlap_h
    return intersection / (a.width * a.height + b.width * b.height - intersection)


def best_iou(box: YoloBox, others: list[YoloBox]) -> float:
    """IoU lớn nhất giữa `box` và một box bất kỳ trong `others` (0 nếu `others` rỗng).

    Dùng để hỏi "chỗ này có box nào không" — khác với `match_boxes` (ghép một-một), vì nhiều mô hình
    có thể cùng phát hiện một xe và tất cả các dự đoán đó đều phải được coi là đã có nhãn.
    """
    return max((iou(box, other) for other in others), default=0.0)


def overlap_fraction(box: YoloBox, region: YoloBox) -> float:
    """Tỉ lệ diện tích của `box` nằm trong `region` (0–1).

    Dùng cho vùng ignore: vùng thường to hơn xe bên trong nhiều, nên IoU nhỏ dù xe nằm trọn trong vùng.
    """
    overlap_w = min(box.center_x + box.width / 2, region.center_x + region.width / 2) - max(box.center_x - box.width / 2, region.center_x - region.width / 2)
    overlap_h = min(box.center_y + box.height / 2, region.center_y + region.height / 2) - max(box.center_y - box.height / 2, region.center_y - region.height / 2)
    if overlap_w <= 0 or overlap_h <= 0 or box.width * box.height == 0:
        return 0.0
    return overlap_w * overlap_h / (box.width * box.height)


def find_duplicates(boxes: list[YoloBox], min_iou: float = 0.8) -> list[tuple[int, int]]:
    """Các cặp box cùng lớp chồng gần như hoàn toàn lên nhau — nhiều khả năng hai box trên cùng một xe.

    Ngưỡng 0.8 cao hơn nhiều so với hai xe khác nhau che nhau (thường IoU < 0.5), nên ít báo nhầm.
    """
    return [
        (i, j)
        for i in range(len(boxes))
        for j in range(i + 1, len(boxes))
        if boxes[i].class_id == boxes[j].class_id and iou(boxes[i], boxes[j]) >= min_iou
    ]


def match_boxes(reference: list[YoloBox], candidate: list[YoloBox], min_iou: float = 0.5) -> MatchResult:
    """Ghép từng box của `reference` với box của `candidate` theo IoU lớn nhất (tham lam, mỗi box ghép một lần).

    Không xét lớp khi ghép, để phát hiện được cả trường hợp đúng vị trí nhưng khác lớp.
    """
    scored = sorted(
        ((iou(r, c), i, j) for i, r in enumerate(reference) for j, c in enumerate(candidate)),
        reverse=True,
    )
    used_reference, used_candidate = set(), set()
    result = MatchResult()
    for score, i, j in scored:
        if score < min_iou:
            break
        if i in used_reference or j in used_candidate:
            continue
        used_reference.add(i)
        used_candidate.add(j)
        result.pairs.append((i, j))
    result.unmatched_reference = [i for i in range(len(reference)) if i not in used_reference]
    result.unmatched_candidate = [j for j in range(len(candidate)) if j not in used_candidate]
    return result


def deduplicate(boxes: list[YoloBox], min_iou: float = 0.5) -> list[YoloBox]:
    """Bỏ các box chỉ cùng một chỗ (vd. hai mô hình cùng phát hiện một xe), giữ box có độ tin cậy cao hơn."""
    kept: list[YoloBox] = []
    for candidate in sorted(boxes, key=lambda b: b.confidence, reverse=True):
        if all(iou(candidate, existing) < min_iou for existing in kept):
            kept.append(candidate)
    return kept


def invalid_boxes(boxes: list[YoloBox], image_width: int, image_height: int, min_side_px: float = 4.0) -> list[int]:
    """Box có cạnh nhỏ hơn `min_side_px` pixel (thường do lỡ tay click) hoặc có tâm nằm ngoài ảnh."""
    return [
        i
        for i, box in enumerate(boxes)
        if box.width * image_width < min_side_px
        or box.height * image_height < min_side_px
        or not (0 <= box.center_x <= 1 and 0 <= box.center_y <= 1)
    ]


# Box nằm >= tỉ lệ này bên trong một box cùng lớp lớn hơn coi là trùng (vd. "chỉ phần xe" trong box "xe + người lái")
NESTED_DUPLICATE_FRACTION = 0.8


def drop_nested_duplicates(boxes: list[YoloBox], min_inside: float = NESTED_DUPLICATE_FRACTION) -> list[YoloBox]:
    """Bỏ box nằm gần trọn bên trong một box CÙNG LỚP lớn hơn — giữ box lớn, vì quy ước dự án là box xe máy bao cả
    người lái. Dùng để dọn nhãn sơ bộ của mô hình COCO trước khi người gán kiểm (không dùng cho nhãn đã kiểm)."""
    kept = []
    for box in boxes:
        area = box.width * box.height
        nested = any(
            other is not box and other.class_id == box.class_id and other.width * other.height > area
            and overlap_fraction(box, other) >= min_inside
            for other in boxes
        )
        if not nested:
            kept.append(box)
    return kept
