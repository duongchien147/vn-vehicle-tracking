"""Chấm một mô hình phát hiện trên một tập ảnh có nhãn (bộ chấm kiểu COCO, có vùng ignore) — dùng chung cho
mọi tập test (frame Hà Nội, val UVH-26...) để các bảng số liệu tính theo cùng một cách."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vtrack import CLASS_NAMES
from vtrack.coco_eval import CocoMetrics, EvalImage, bootstrap_map50, evaluate, mean_ap50
from vtrack.label_review import YoloBox
from vtrack.models import FALLBACK_IMGSZ, load_model, training_imgsz
from vtrack.mot_format import MotBox
from vtrack.pseudo_label import predict_teacher

# Ngưỡng điểm rất thấp là chuẩn khi tính mAP: đường precision-recall cần cả các phát hiện điểm thấp
EVAL_CONF = 0.001


@dataclass(frozen=True)
class ModelSpec:
    weights: str
    imgsz: int | None  # None = imgsz lúc train/export của mô hình

    @classmethod
    def parse(cls, spec: str) -> "ModelSpec":
        """'weights/x.pt@1280' -> imgsz 1280; không có '@' -> imgsz lúc train."""
        weights, _, imgsz = spec.partition("@")
        return cls(weights, int(imgsz) if imgsz else None)


@dataclass
class ModelScore:
    name: str
    imgsz: int
    metrics: CocoMetrics
    main_map50: float
    main_map50_95: float
    ci_low: float  # khoảng tin cậy bootstrap của mAP50 chính (nan nếu không bootstrap)
    ci_high: float
    seconds_per_image: float

    def to_row(self, counts: dict[str, int]) -> dict[str, object]:
        """Một dòng CSV cho báo cáo (cột giống nhau cho mọi tập test)."""
        row: dict[str, object] = {
            "model": self.name, "imgsz": self.imgsz,
            "main_mAP50": self.main_map50, "main_mAP50_ci_low": self.ci_low, "main_mAP50_ci_high": self.ci_high,
            "main_mAP50-95": self.main_map50_95,
            "mAP50_all_classes": self.metrics.map50, "mAP50-95_all_classes": self.metrics.map50_95,
            "recall": self.metrics.recall,
            **{f"AP50_{n}": self.metrics.ap50_per_class[n] for n in CLASS_NAMES},
            **{f"AP50-95_{n}": self.metrics.ap50_95_per_class[n] for n in CLASS_NAMES},
            **{f"n_{n}": counts[n] for n in CLASS_NAMES},
            "s_per_image_cpu": self.seconds_per_image,
        }
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}


def score_model(
    spec: ModelSpec,
    images: list[EvalImage],
    load_frame: Callable[[int], np.ndarray],
    main_classes: list[str],
    bootstrap_rounds: int,
) -> ModelScore:
    """Dự đoán trên mọi ảnh rồi chấm. `load_frame(i)` trả ảnh thứ i (đọc khi cần: tập lớn không nạp hết vào RAM).
    Ghi dự đoán vào `images[i].predictions` (ghi đè kết quả của mô hình trước)."""
    model = load_model(spec.weights)
    imgsz = spec.imgsz or training_imgsz(model) or FALLBACK_IMGSZ
    started = time.perf_counter()
    for index, image in enumerate(images):
        image.predictions = predict_teacher(model, load_frame(index), imgsz, EVAL_CONF)
    seconds_per_image = (time.perf_counter() - started) / len(images)

    metrics = evaluate(images)
    if bootstrap_rounds:
        low, high = bootstrap_map50(images, bootstrap_rounds, main_classes)
    else:
        low, high = float("nan"), float("nan")
    return ModelScore(
        name=Path(spec.weights).stem, imgsz=imgsz, metrics=metrics,
        main_map50=mean_ap50(metrics, main_classes),
        main_map50_95=float(np.mean([metrics.ap50_95_per_class[n] for n in main_classes])),
        ci_low=low, ci_high=high, seconds_per_image=seconds_per_image,
    )


def describe(score: ModelScore) -> str:
    """Một dòng tóm tắt để in ra màn hình: mAP50 chính + AP50 từng lớp."""
    per_class = " | ".join(f"{n} {score.metrics.ap50_per_class[n]:.3f}" for n in CLASS_NAMES)
    return (f"  {score.name:20s} @{score.imgsz:<5d} mAP50 chính {score.main_map50:.3f} "
            f"[{score.ci_low:.3f}–{score.ci_high:.3f}] | AP50: {per_class}")


def frames_to_eval_images(
    labeled: list[tuple[str, MotBox]], frames: list[int], width: int, height: int, ignore_label: str,
) -> list[EvalImage]:
    """Nhãn tracking (box pixel theo frame, đánh từ 1) -> ảnh chấm detector cho các frame được chọn.
    Box nhãn `ignore_label` thành vùng ignore; lớp không thuộc dự án thì bỏ."""
    by_frame: dict[int, list[tuple[str, MotBox]]] = {frame: [] for frame in frames}
    for label, box in labeled:
        if box.frame in by_frame:
            by_frame[box.frame].append((label, box))

    def to_yolo(class_id: int, box: MotBox) -> YoloBox:
        return YoloBox(class_id, (box.left + box.width / 2) / width, (box.top + box.height / 2) / height,
                       box.width / width, box.height / height)

    return [
        EvalImage(
            file_name=f"frame_{frame:06d}", width=width, height=height,
            truth=[to_yolo(CLASS_NAMES.index(label), box) for label, box in by_frame[frame] if label in CLASS_NAMES],
            ignore=[to_yolo(-1, box) for label, box in by_frame[frame] if label == ignore_label],
        )
        for frame in frames
    ]
