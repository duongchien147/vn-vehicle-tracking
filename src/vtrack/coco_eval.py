"""Chấm mAP kiểu COCO (pycocotools — bộ chấm chính thức của COCO) cho nhãn YOLO có vùng ignore.

Dùng cho tập test Hà Nội: `yolo val` của ultralytics không hỗ trợ vùng bỏ qua, còn COCO có sẵn cơ chế `iscrowd`:
phát hiện rơi vào vùng crowd không bị tính là sai, và vùng đó không bị tính là bỏ sót.
"""

import contextlib
import io
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from vtrack import CLASS_NAMES
from vtrack.frame_sampling import parse_frame_name
from vtrack.label_review import YoloBox

# COCO mặc định chấm tối đa 100 phát hiện/ảnh; frame Hà Nội có ~110–140 xe nên phải nâng lên (như benchmark cảnh
# đông CrowdHuman), nếu không mô hình tốt bị cắt bớt phát hiện và điểm thấp oan.
MAX_DETECTIONS = 500


@dataclass
class EvalImage:
    file_name: str
    width: int
    height: int
    truth: list[YoloBox]  # nhãn người, không gồm vùng ignore
    ignore: list[YoloBox]
    predictions: list[YoloBox] = field(default_factory=list)


@dataclass
class CocoMetrics:
    map50_95: float  # chỉ số chính của COCO: trung bình AP ở IoU 0.50:0.05:0.95
    map50: float
    # AP ở IoU 0.75: đo độ khít của box. Lưu ý: trên tập có nhãn sửa từ nhãn sơ bộ của một mô hình, chỉ số IoU chặt
    # thiên vị mô hình (hoặc họ mô hình) đã vẽ nhãn sơ bộ (vd. test Hà Nội sửa từ nhãn sơ bộ của v2)
    map75: float
    map_small: float  # AP50-95 cho vật nhỏ (diện tích < 32x32 px)
    recall: float  # AR ở MAX_DETECTIONS phát hiện/ảnh
    ap50_per_class: dict[str, float]
    ap50_95_per_class: dict[str, float]


def _absolute_xywh(box: YoloBox, width: int, height: int) -> list[float]:
    return [(box.center_x - box.width / 2) * width, (box.center_y - box.height / 2) * height,
            box.width * width, box.height * height]


def to_coco(images: Sequence[EvalImage]) -> tuple[dict, list[dict]]:
    """Đổi sang định dạng COCO: (nhãn chuẩn dạng dataset COCO, danh sách phát hiện dạng kết quả COCO).

    Vùng ignore không gắn với lớp nào, nên được thêm thành annotation `iscrowd=1` cho mọi lớp.
    """
    categories = [{"id": index + 1, "name": name} for index, name in enumerate(CLASS_NAMES)]
    coco_images, annotations, detections = [], [], []
    for image_id, image in enumerate(images, start=1):
        coco_images.append({"id": image_id, "file_name": image.file_name, "width": image.width, "height": image.height})
        crowd = [(region, category["id"]) for region in image.ignore for category in categories]
        for box, category_id, is_crowd in [(b, b.class_id + 1, 0) for b in image.truth] + [(r, c, 1) for r, c in crowd]:
            xywh = _absolute_xywh(box, image.width, image.height)
            annotations.append({"id": len(annotations) + 1, "image_id": image_id, "category_id": category_id,
                                "bbox": xywh, "area": xywh[2] * xywh[3], "iscrowd": is_crowd})
        for box in image.predictions:
            detections.append({"image_id": image_id, "category_id": box.class_id + 1,
                               "bbox": _absolute_xywh(box, image.width, image.height), "score": box.confidence})
    return {"images": coco_images, "annotations": annotations, "categories": categories}, detections


def _matched_evaluator(images: Sequence[EvalImage], iou_thresholds: Sequence[float] | None = None) -> COCOeval:
    """COCOeval đã ghép phát hiện với nhãn từng ảnh (evaluate()); `iou_thresholds=None` = 0.50:0.05:0.95 của COCO."""
    ground_truth, detections = to_coco(images)
    with contextlib.redirect_stdout(io.StringIO()):  # pycocotools in rất nhiều dòng; chỉ lấy số
        coco_truth = COCO()
        coco_truth.dataset = cast(Any, ground_truth)  # stub kiểu của pycocotools quá hẹp; định dạng là dataset COCO
        coco_truth.createIndex()
        # loadRes nhận cả danh sách kết quả (không chỉ đường dẫn file như stub kiểu khai báo)
        evaluator = COCOeval(coco_truth, coco_truth.loadRes(cast(Any, detections)), "bbox")
        evaluator.params.maxDets = [1, 10, MAX_DETECTIONS]
        if iou_thresholds is not None:
            evaluator.params.iouThrs = np.array(iou_thresholds)
        evaluator.evaluate()
    return evaluator


def evaluate(images: Sequence[EvalImage]) -> CocoMetrics:
    if not any(image.predictions for image in images):
        zeros = {name: 0.0 for name in CLASS_NAMES}
        return CocoMetrics(0.0, 0.0, 0.0, 0.0, 0.0, zeros, dict(zeros))
    evaluator = _matched_evaluator(images)
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.accumulate()

    # Không dùng evaluator.summarize()/stats: hàm đó viết cứng maxDets=100 cho mAP50-95 và trả -1 khi maxDets bị
    # đổi, nên tính trực tiếp từ mảng kết quả.
    # precision: [IoU, mức recall, lớp, vùng diện tích, maxDets]; recall: [IoU, lớp, vùng diện tích, maxDets].
    # Vùng diện tích: 0 = tất cả, 1 = nhỏ (< 32x32 px). Giá trị -1 = không có nhãn chuẩn cho ô đó (bỏ khi lấy trung bình).
    precision, recall = evaluator.eval["precision"], evaluator.eval["recall"]
    all_areas, small_area, max_dets = 0, 1, -1
    iou_50, iou_75 = 0, 5  # vị trí của IoU 0.50 và 0.75 trong dãy 0.50:0.05:0.95

    def mean_valid(values: np.ndarray) -> float:
        valid = values[values > -1]
        return float(valid.mean()) if valid.size else float("nan")

    return CocoMetrics(
        map50_95=mean_valid(precision[:, :, :, all_areas, max_dets]),
        map50=mean_valid(precision[iou_50, :, :, all_areas, max_dets]),
        map75=mean_valid(precision[iou_75, :, :, all_areas, max_dets]),
        map_small=mean_valid(precision[:, :, :, small_area, max_dets]),
        recall=mean_valid(recall[:, :, all_areas, max_dets]),
        ap50_per_class={
            name: mean_valid(precision[iou_50, :, i, all_areas, max_dets]) for i, name in enumerate(CLASS_NAMES)
        },
        ap50_95_per_class={
            name: mean_valid(precision[:, :, i, all_areas, max_dets]) for i, name in enumerate(CLASS_NAMES)
        },
    )


def instance_counts(images: Sequence[EvalImage]) -> dict[str, int]:
    """Số box nhãn chuẩn của mỗi lớp trong tập."""
    counts = {name: 0 for name in CLASS_NAMES}
    for image in images:
        for box in image.truth:
            counts[CLASS_NAMES[box.class_id]] += 1
    return counts


def well_represented_classes(images: Sequence[EvalImage], min_instances: int) -> list[str]:
    """Lớp có ít nhất `min_instances` box nhãn chuẩn — chỉ các lớp này mới cho AP đủ tin cậy.

    COCO lấy trung bình đều AP các lớp vì lớp nào cũng có hàng nghìn mẫu; khi một lớp chỉ có vài box (vd. 2 xe tải),
    một phát hiện đúng/sai làm AP lớp đó nhảy 0 <-> 1 và kéo cả mAP theo, nên mAP chính chỉ tính trên các lớp đủ mẫu.
    """
    return [name for name, count in instance_counts(images).items() if count >= min_instances]


def mean_ap50(metrics: CocoMetrics, classes: Sequence[str]) -> float:
    return float(np.mean([metrics.ap50_per_class[name] for name in classes]))


# Số lần lấy mẫu lại: khoảng tin cậy theo phân vị 2.5%/97.5% cần >= 1000 lần mới ổn định (Efron & Tibshirani);
# với 20 lần, hai phân vị đó gần như là giá trị nhỏ nhất/lớn nhất của 20 số và thay đổi theo seed.
BOOTSTRAP_ROUNDS = 1000
# 101 mức recall 0, 0.01, ..., 1 mà COCO dùng để nội suy precision (COCOeval.params.recThrs)
RECALL_THRESHOLDS = np.linspace(0.0, 1.0, 101)


@dataclass
class ClassMatches:
    """Kết quả ghép ở IoU 0.5 của một lớp trong một ảnh: đúng các mảng pycocotools dùng khi tính AP."""

    scores: np.ndarray  # độ tin cậy của các phát hiện
    true_positive: np.ndarray  # bool: ghép được với một nhãn chuẩn
    false_positive: np.ndarray  # bool: không ghép được và không rơi vào vùng ignore
    truth_count: int  # số nhãn chuẩn (không tính vùng ignore)


def iou50_matches(images: Sequence[EvalImage]) -> list[list[ClassMatches]]:
    """Ghép phát hiện với nhãn ở IoU 0.5 một lần cho mọi ảnh -> [lớp][ảnh]. Việc ghép làm riêng từng ảnh, nên lấy mẫu
    lại tập ảnh chỉ cần gộp lại kết quả đã ghép, không phải chạy lại COCOeval."""
    evaluator = _matched_evaluator(images, iou_thresholds=[0.5])
    image_count, area_count = len(images), len(evaluator.params.areaRng)
    empty = ClassMatches(np.empty(0), np.empty(0, bool), np.empty(0, bool), 0)
    per_class = []
    for class_index in range(len(CLASS_NAMES)):
        rows = []
        for image_index in range(image_count):
            # evalImgs xếp theo [lớp][vùng diện tích][ảnh]; vùng diện tích 0 = tất cả
            record = evaluator.evalImgs[class_index * area_count * image_count + image_index]
            if record is None:  # ảnh không có nhãn lẫn phát hiện của lớp này
                rows.append(empty)
                continue
            matched = record["dtMatches"][0] > 0
            ignored = record["dtIgnore"][0].astype(bool)
            rows.append(ClassMatches(
                scores=np.asarray(record["dtScores"], dtype=float),
                true_positive=matched & ~ignored,
                false_positive=~matched & ~ignored,
                truth_count=int(np.count_nonzero(np.asarray(record["gtIgnore"]) == 0)),
            ))
        per_class.append(rows)
    return per_class


def average_precision(parts: Sequence[ClassMatches]) -> float:
    """AP nội suy 101 điểm, cùng cách COCOeval.accumulate(); nan nếu không có nhãn chuẩn nào."""
    truth = sum(part.truth_count for part in parts)
    if truth == 0:
        return float("nan")
    order = np.argsort(-np.concatenate([part.scores for part in parts]), kind="mergesort")
    tp_sum = np.cumsum(np.concatenate([part.true_positive for part in parts])[order], dtype=float)
    fp_sum = np.cumsum(np.concatenate([part.false_positive for part in parts])[order], dtype=float)
    recall = tp_sum / truth
    precision = tp_sum / (tp_sum + fp_sum + np.spacing(1))
    precision = np.maximum.accumulate(precision[::-1])[::-1]  # precision nội suy: max của các điểm recall lớn hơn
    positions = np.searchsorted(recall, RECALL_THRESHOLDS, side="left")
    interpolated = np.zeros(len(RECALL_THRESHOLDS))
    reached = positions < len(precision)
    interpolated[reached] = precision[positions[reached]]
    return float(interpolated.mean())


# Độ dài khối thời gian của bootstrap theo khối: trên nhãn tracking clip PVD, 90% số xe ở trong khung hình < 8.2 s
# (trung vị 1.3 s), nên hai frame cách nhau > 10 s gần như không chung xe nào -> coi các khối 10 s là độc lập.
BLOCK_SECONDS = 10.0
# Ít khối quá thì số cách lấy mẫu lại khác nhau rất ít (2 khối -> 3 tổ hợp), phân vị 2.5%/97.5% vô nghĩa
MIN_BOOTSTRAP_UNITS = 5


def time_blocks(clips: Sequence[str], seconds: Sequence[float], block_seconds: float = BLOCK_SECONDS) -> list[str]:
    """Nhãn khối của từng frame: frame cùng clip, cách frame đầu của clip trong cùng một đoạn `block_seconds` giây.

    Tính từ frame đầu của mỗi clip (không từ giây 0) để các khối có số frame đều nhau.
    """
    first = {clip: min(t for c, t in zip(clips, seconds) if c == clip) for clip in set(clips)}
    return [f"{clip}#{int((t - first[clip]) // block_seconds)}" for clip, t in zip(clips, seconds)]


def frame_name_blocks(file_names: Sequence[str], block_seconds: float = BLOCK_SECONDS) -> list[str] | None:
    """Khối thời gian từ tên frame '<clip>_MMmSSs.jpg' (extract_label_frames.py); None nếu có tên không theo mẫu
    (vd. ảnh Roboflow) -> không chia khối được."""
    clips, seconds = [], []
    for name in file_names:
        parsed = parse_frame_name(Path(name).stem)
        if parsed is None:
            return None
        clips.append(parsed[0])
        seconds.append(parsed[1])
    return time_blocks(clips, seconds, block_seconds)


def bootstrap_map50(
    images: Sequence[EvalImage], rounds: int, classes: Sequence[str], seed: int = 0,
    groups: Sequence[str] | None = None,
) -> tuple[float, float]:
    """Khoảng tin cậy 95% của mAP50 (trung bình trên `classes`) bằng bootstrap (lấy mẫu lại có hoàn lại).

    Tập test nhỏ (30 ảnh) nên một con số mAP đơn lẻ dao động nhiều; khoảng tin cậy cho biết hai mô hình
    có khác nhau thật hay chỉ do may rủi chọn ảnh.
    - `groups=None`: lấy mẫu lại từng ảnh. Các frame gần nhau trong một clip chung nhiều xe mà cách này coi chúng
      độc lập, nên khoảng ra hẹp hơn thực tế.
    - `groups` = nhãn khối/cụm của từng ảnh (vd. `time_blocks`, hay tên video): lấy mẫu lại cả khối (block / cluster
      bootstrap, cách chuẩn cho dữ liệu có tương quan). Ít hơn MIN_BOOTSTRAP_UNITS khối -> (nan, nan).
    """
    if groups is None:
        units = [[index] for index in range(len(images))]
    else:
        by_group: dict[str, list[int]] = {}
        for index, group in enumerate(groups):
            by_group.setdefault(group, []).append(index)
        units = list(by_group.values())
        if len(units) < MIN_BOOTSTRAP_UNITS:
            return float("nan"), float("nan")
    matches = iou50_matches(images)
    class_rows = [matches[CLASS_NAMES.index(name)] for name in classes]
    generator = np.random.default_rng(seed)
    scores = []
    for _ in range(rounds):
        picked = generator.integers(0, len(units), size=len(units))
        sample = [index for unit in picked for index in units[unit]]
        scores.append(np.mean([average_precision([rows[i] for i in sample]) for rows in class_rows]))
    return float(np.nanpercentile(scores, 2.5)), float(np.nanpercentile(scores, 97.5))
