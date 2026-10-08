"""Phân loại lỗi để minh hoạ trong báo cáo (thí nghiệm E4): ghép dự đoán với nhãn trên một ảnh (đúng / sai lớp /
bỏ sót / thừa) và dò các lần đổi ID trong kết quả tracking so với nhãn tracking."""

from collections import defaultdict
from dataclasses import dataclass

from vtrack.label_review import YoloBox
from vtrack.mot_format import MotBox, box_iou

MATCH_IOU = 0.5  # cùng ngưỡng với mAP50 và với bộ chấm tracking (TrackEval CLEAR/Identity)


def yolo_iou(a: YoloBox, b: YoloBox) -> float:
    ax1, ay1, ax2, ay2 = a.center_x - a.width / 2, a.center_y - a.height / 2, a.center_x + a.width / 2, a.center_y + a.height / 2
    bx1, by1, bx2, by2 = b.center_x - b.width / 2, b.center_y - b.height / 2, b.center_x + b.width / 2, b.center_y + b.height / 2
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


@dataclass
class DetectionOutcome:
    correct: list[tuple[YoloBox, YoloBox]]  # (nhãn, dự đoán) cùng lớp
    wrong_class: list[tuple[YoloBox, YoloBox]]  # khớp vị trí nhưng khác lớp
    missed: list[YoloBox]  # nhãn không có dự đoán nào khớp
    false_positives: list[YoloBox]  # dự đoán không khớp nhãn nào


def classify_detections(truth: list[YoloBox], predictions: list[YoloBox], min_iou: float = MATCH_IOU) -> DetectionOutcome:
    """Ghép tham lam theo điểm dự đoán giảm dần (như COCO): mỗi dự đoán lấy nhãn chưa ghép có IoU cao nhất."""
    outcome = DetectionOutcome([], [], [], [])
    unmatched = list(range(len(truth)))
    for prediction in sorted(predictions, key=lambda p: p.confidence, reverse=True):
        best, best_iou = None, min_iou
        for index in unmatched:
            iou = yolo_iou(truth[index], prediction)
            if iou >= best_iou:
                best, best_iou = index, iou
        if best is None:
            outcome.false_positives.append(prediction)
            continue
        unmatched.remove(best)
        pair = (truth[best], prediction)
        (outcome.correct if truth[best].class_id == prediction.class_id else outcome.wrong_class).append(pair)
    outcome.missed = [truth[i] for i in unmatched]
    return outcome


@dataclass(frozen=True)
class IdSwitch:
    truth_id: int
    frame: int  # frame đầu tiên xe được gán ID mới
    previous_frame: int  # frame cuối cùng còn mang ID cũ
    old_id: int
    new_id: int


def find_id_switches(truth: list[MotBox], predicted: list[MotBox], min_iou: float = MATCH_IOU) -> list[IdSwitch]:
    """Mỗi frame, mỗi xe trong nhãn được ghép với box dự đoán IoU cao nhất (>= min_iou). Một xe đang khớp ID A rồi
    khớp sang ID B = một lần đổi ID (cùng ý nghĩa với IDSW của CLEAR MOT, nhưng ghép đơn giản hơn để minh hoạ)."""
    predicted_by_frame: defaultdict[int, list[MotBox]] = defaultdict(list)
    for box in predicted:
        predicted_by_frame[box.frame].append(box)
    last_match: dict[int, tuple[int, int]] = {}  # truth_id -> (frame, predicted_id)
    switches = []
    for box in sorted(truth, key=lambda b: b.frame):
        candidates = [(box_iou(box, p), p.track_id) for p in predicted_by_frame[box.frame]]
        iou, predicted_id = max(candidates, default=(0.0, -1))
        if iou < min_iou:
            continue
        previous = last_match.get(box.track_id)
        if previous is not None and previous[1] != predicted_id:
            switches.append(IdSwitch(box.track_id, box.frame, previous[0], previous[1], predicted_id))
        last_match[box.track_id] = (box.frame, predicted_id)
    return switches
