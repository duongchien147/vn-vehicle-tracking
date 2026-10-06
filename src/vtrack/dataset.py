"""Các bước xử lý dataset YOLO: đổi nhãn về lớp dự án, tìm ảnh gần trùng và chia train/val/test
theo nhóm ảnh gần trùng (tránh rò rỉ dữ liệu giữa train và test)."""

import random
from collections.abc import Sequence

import cv2
import numpy as np

from vtrack import CLASS_NAMES


def remap_label_lines(
    lines: Sequence[str], source_names: Sequence[str], class_map: dict[str, str]
) -> list[str]:
    """Đổi id lớp trong các dòng nhãn YOLO (`cls cx cy w h`) từ dataset gốc sang id lớp dự án.

    Box thuộc lớp không có trong `class_map` bị bỏ.
    """
    remapped = []
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        source_name = str(source_names[int(parts[0])])
        target_name = class_map.get(source_name)
        if target_name is None:
            continue
        remapped.append(" ".join([str(CLASS_NAMES.index(target_name)), *parts[1:5]]))
    return remapped


# Tên lớp VisDrone theo thứ tự id YOLO (id gốc trong file nhãn VisDrone = id YOLO + 1;
# id gốc 0 là "vùng bỏ qua", 11 là "khác" — cả hai đều bị bỏ)
VISDRONE_NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]


VISDRONE_MOTOR = VISDRONE_NAMES.index("motor")
VISDRONE_PEOPLE = VISDRONE_NAMES.index("people")  # người không đi bộ: đang ngồi/lái xe

# Box người lái được gộp vào box xe máy nếu ít nhất tỉ lệ này diện tích box người nằm trong box xe
# (người ngồi trên xe: nửa dưới cơ thể chồng lên xe; người đứng cạnh xe thường chồng rất ít).
RIDER_MIN_OVERLAP = 0.2


def merge_riders_into_motorbikes(
    boxes: list[tuple[int, float, float, float, float]],
    min_overlap: float = RIDER_MIN_OVERLAP,
    motor_class: int = VISDRONE_MOTOR,
    rider_class: int = VISDRONE_PEOPLE,
) -> list[tuple[int, float, float, float, float]]:
    """Mở rộng box xe máy để bao cả người lái (box người chồng lên nó).

    VisDrone và COCO vẽ xe và người lái thành hai box riêng, còn các dataset Roboflow Việt Nam vẽ một box bao
    cả người lẫn xe. Gộp lại để mọi nguồn cùng một khái niệm "xe máy". Box dạng (lớp, x1, y1, x2, y2), theo pixel
    hay chuẩn hoá đều được; mỗi người lái chỉ được gộp vào một xe (xe chồng nhiều nhất). Mặc định là id lớp
    VisDrone; với COCO truyền motor_class=3 (motorcycle), rider_class=0 (person).
    """
    merged = [list(box) for box in boxes]
    motor_indices = [i for i, box in enumerate(boxes) if box[0] == motor_class]
    for person in boxes:
        if person[0] != rider_class:
            continue
        _, px1, py1, px2, py2 = person
        person_area = (px2 - px1) * (py2 - py1)
        best_index, best_overlap = None, min_overlap
        for i in motor_indices:
            _, mx1, my1, mx2, my2 = boxes[i]
            intersection = max(0.0, min(px2, mx2) - max(px1, mx1)) * max(0.0, min(py2, my2) - max(py1, my1))
            if person_area > 0 and intersection / person_area >= best_overlap:
                best_index, best_overlap = i, intersection / person_area
        if best_index is not None:
            box = merged[best_index]
            box[1:] = [min(box[1], px1), min(box[2], py1), max(box[3], px2), max(box[4], py2)]
    return [(int(box[0]), box[1], box[2], box[3], box[4]) for box in merged]


def visdrone_to_yolo_lines(annotation_rows: Sequence[str], image_width: int, image_height: int) -> list[str]:
    """Đổi nhãn VisDrone (`x,y,w,h,score,category,truncation,occlusion` theo pixel) sang dòng YOLO chuẩn hoá.

    Bỏ vùng bị đánh dấu bỏ qua (score = 0) và các lớp không thuộc 10 lớp chính; box xe máy được mở rộng
    để bao cả người lái (xem `merge_riders_into_motorbikes`).
    """
    boxes = []
    for row in annotation_rows:
        fields = row.strip().split(",")
        if len(fields) < 6 or fields[4] == "0":
            continue
        category = int(fields[5])
        if not 1 <= category <= len(VISDRONE_NAMES):
            continue
        left, top, width, height = map(int, fields[:4])
        if width <= 0 or height <= 0:
            continue
        boxes.append((category - 1, float(left), float(top), float(left + width), float(top + height)))

    lines = []
    for class_id, x1, y1, x2, y2 in merge_riders_into_motorbikes(boxes):
        center_x = (x1 + x2) / 2 / image_width
        center_y = (y1 + y2) / 2 / image_height
        box_width = (x2 - x1) / image_width
        box_height = (y2 - y1) / image_height
        lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}")
    return lines


def has_min_boxes(label_lines: Sequence[str], min_boxes: dict[str, int]) -> bool:
    """True nếu nhãn (đã đổi sang lớp dự án) có ít nhất `min_boxes[lớp]` box cho mọi lớp được yêu cầu."""
    counts = {name: 0 for name in CLASS_NAMES}
    for line in label_lines:
        counts[CLASS_NAMES[int(line.split()[0])]] += 1
    return all(counts[name] >= minimum for name, minimum in min_boxes.items())


def sample_items(items: Sequence, max_items: int | None, seed: int) -> list:
    """Lấy ngẫu nhiên tối đa `max_items` phần tử (giữ thứ tự gốc); None = lấy hết. Cố định `seed` để tái lập."""
    if max_items is None or len(items) <= max_items:
        return list(items)
    chosen = sorted(random.Random(seed).sample(range(len(items)), max_items))
    return [items[i] for i in chosen]


def dhash(image: np.ndarray, hash_size: int = 8) -> int:
    """Difference hash 64-bit: so sánh độ sáng các điểm ảnh kề nhau trên ảnh thu nhỏ.

    Hai ảnh gần giống nhau (frame liên tiếp, khác nén/độ sáng một chút) cho hash gần nhau.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    small = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def group_near_duplicates(hashes: Sequence[int], max_distance: int) -> list[int]:
    """Gán mỗi ảnh một id nhóm; hai ảnh có khoảng cách Hamming <= `max_distance` chung nhóm.

    Dùng union-find, nên quan hệ "gần trùng" có tính bắc cầu (A~B, B~C => A, B, C chung nhóm).
    """
    parent = list(range(len(hashes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    as_array = np.array(hashes, dtype=np.uint64)
    for i in range(len(as_array)):
        xor = (as_array[i + 1 :] ^ as_array[i]).view(np.uint8).reshape(-1, 8)
        distances = np.unpackbits(xor, axis=1).sum(axis=1)
        for offset in np.flatnonzero(distances <= max_distance):
            root_a, root_b = find(i), find(i + 1 + int(offset))
            if root_a != root_b:
                parent[root_b] = root_a
    return [find(i) for i in range(len(hashes))]


def split_by_group(
    group_ids: Sequence[int], ratios: dict[str, float], seed: int
) -> list[str]:
    """Chia ảnh vào các tập (train/val/test) sao cho mọi ảnh cùng nhóm nằm cùng một tập.

    Nhóm được xáo trộn ngẫu nhiên (theo `seed`) rồi lấp dần vào từng tập theo tỉ lệ số ảnh.
    """
    groups: dict[int, list[int]] = {}
    for index, group in enumerate(group_ids):
        groups.setdefault(group, []).append(index)
    order = list(groups)
    random.Random(seed).shuffle(order)

    total = len(group_ids)
    split_names = list(ratios)
    targets = {name: ratios[name] * total for name in split_names}
    assigned = [""] * total
    counts = {name: 0 for name in split_names}
    current = 0
    for group in order:
        # chuyển sang tập kế tiếp khi tập hiện tại đã đủ số ảnh (tập cuối nhận phần còn lại)
        while current < len(split_names) - 1 and counts[split_names[current]] >= targets[split_names[current]]:
            current += 1
        name = split_names[current]
        for index in groups[group]:
            assigned[index] = name
        counts[name] += len(groups[group])
    return assigned


# Lưới 16x16 (256 bit) thay vì 8x8: để nhận ra "cùng một camera" cần phân biệt được bố cục nền chi tiết hơn,
# hash 64 bit dễ cho hai camera khác nhau nhưng bố cục giống nhau (đường thẳng, trời ở trên) trùng nhau.
CAMERA_HASH_SIZE = 16


def dhash_bits(image: np.ndarray, hash_size: int = CAMERA_HASH_SIZE) -> np.ndarray:
    """Difference hash dạng mảng bool (hash_size * hash_size bit) — như `dhash` nhưng không giới hạn 64 bit."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    small = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).flatten()


def nearest_hamming_distances(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Với mỗi hash trong `query` (n x bit), khoảng cách Hamming nhỏ nhất tới các hash trong `reference` (m x bit).

    Dùng để loại ảnh test chụp từ cùng camera với ảnh train (camera cố định -> nền gần như không đổi -> hash gần).
    """
    distances = (query[:, None, :] != reference[None, :, :]).sum(axis=2)
    return distances.min(axis=1)
