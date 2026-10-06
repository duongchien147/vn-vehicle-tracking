"""Chấm tracking theo chuẩn MOTChallenge bằng TrackEval (HOTA, CLEAR MOT, Identity).

Dùng nguyên bộ nạp và bước tiền xử lý MOT17 của TrackEval (không tự viết lại công thức):
nhãn chuẩn lớp 1 được chấm, lớp distractor (8) là vùng trung tính. Chấm không phân lớp xe
("xe" là một lớp) — đúng với cách tracker của ultralytics ghép track chỉ theo IoU, không xét lớp;
việc phân loại được chấm riêng bằng mAP của detector.
"""

import configparser
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trackeval

from vtrack.mot_format import MotBox, write_mot_ground_truth, write_mot_results

# Tên lớp duy nhất mà bộ nạp MOT17 cho phép chấm; ở đây nó đại diện cho "xe" (xem mot_format.MOT_TARGET_CLASS)
_TRACKEVAL_CLASS = "pedestrian"


class _CaseSensitiveConfig(configparser.ConfigParser):
    """seqinfo.ini của MOTChallenge viết hoa tên khoá (seqLength...); ConfigParser mặc định đổi thành chữ thường."""

    def optionxform(self, optionstr: str) -> str:
        return optionstr


@dataclass
class TrackingMetrics:
    hota: float  # trung bình hình học của DetA và AssA, lấy trung bình trên các ngưỡng IoU 0.05–0.95
    det_a: float  # độ chính xác phát hiện (bị kéo xuống nếu nhãn chuẩn thiếu xe)
    ass_a: float  # độ chính xác giữ ID, chỉ tính trên các cặp đã khớp -> không bị ảnh hưởng bởi xe thiếu nhãn
    mota: float
    idf1: float
    idr: float  # ID recall: không phụ thuộc phát hiện thừa
    idp: float  # ID precision
    id_switches: int
    fragmentations: int
    mostly_tracked: int
    mostly_lost: int
    false_positives: int
    false_negatives: int
    gt_ids: int
    predicted_ids: int


def write_ground_truth_sequence(
    gt_root: Path, sequence: str, boxes: list[MotBox], num_frames: int, fps: float, width: int, height: int,
) -> None:
    """Tạo thư mục sequence chuẩn MOTChallenge: <gt_root>/<sequence>/{seqinfo.ini, gt/gt.txt}."""
    sequence_dir = gt_root / sequence
    write_mot_ground_truth(sequence_dir / "gt" / "gt.txt", boxes)
    info = _CaseSensitiveConfig()
    info["Sequence"] = {
        "name": sequence, "imDir": "img1", "frameRate": f"{fps:g}", "seqLength": str(num_frames),
        "imWidth": str(width), "imHeight": str(height), "imExt": ".jpg",
    }
    with (sequence_dir / "seqinfo.ini").open("w", encoding="utf-8") as f:
        info.write(f)


def write_tracker_sequence(trackers_root: Path, tracker: str, sequence: str, boxes: list[MotBox]) -> None:
    """Kết quả của một tracker trên một sequence: <trackers_root>/<tracker>/data/<sequence>.txt."""
    write_mot_results(trackers_root / tracker / "data" / f"{sequence}.txt", boxes)


def evaluate_trackers(
    gt_root: Path, trackers_root: Path, sequences: list[str], trackers: list[str], output_dir: Path,
) -> dict[str, TrackingMetrics]:
    """Chấm mọi tracker trên mọi sequence (gộp các sequence như MOTChallenge). TrackEval ghi chi tiết vào `output_dir`."""
    eval_config = {
        **trackeval.Evaluator.get_default_eval_config(),
        "USE_PARALLEL": False, "PRINT_RESULTS": False, "PRINT_CONFIG": False, "TIME_PROGRESS": False,
        "PLOT_CURVES": False, "LOG_ON_ERROR": str(output_dir / "trackeval_error.log"),
    }
    dataset_config = {
        **trackeval.datasets.MotChallenge2DBox.get_default_dataset_config(),
        "GT_FOLDER": str(gt_root), "TRACKERS_FOLDER": str(trackers_root), "OUTPUT_FOLDER": str(output_dir),
        "TRACKERS_TO_EVAL": trackers, "SEQ_INFO": {sequence: None for sequence in sequences},
        "BENCHMARK": "MOT17", "SKIP_SPLIT_FOL": True, "DO_PREPROC": True, "PRINT_CONFIG": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = trackeval.Evaluator(eval_config)
    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
    # TrackEval luôn tự chạy thêm chỉ số Count (số box, số ID)
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    results, _ = evaluator.evaluate([dataset], metrics)
    by_tracker = results[dataset.get_name()]
    return {tracker: _to_metrics(by_tracker[tracker]["COMBINED_SEQ"][_TRACKEVAL_CLASS]) for tracker in trackers}


def _to_metrics(result: dict) -> TrackingMetrics:
    hota, clear, identity = result["HOTA"], result["CLEAR"], result["Identity"]
    return TrackingMetrics(
        hota=float(np.mean(hota["HOTA"])),
        det_a=float(np.mean(hota["DetA"])),
        ass_a=float(np.mean(hota["AssA"])),
        mota=float(clear["MOTA"]),
        idf1=float(identity["IDF1"]),
        idr=float(identity["IDR"]),
        idp=float(identity["IDP"]),
        id_switches=int(clear["IDSW"]),
        fragmentations=int(clear["Frag"]),
        mostly_tracked=int(clear["MT"]),
        mostly_lost=int(clear["ML"]),
        false_positives=int(clear["CLR_FP"]),
        false_negatives=int(clear["CLR_FN"]),
        gt_ids=int(result["Count"]["GT_IDs"]),
        predicted_ids=int(result["Count"]["IDs"]),
    )
