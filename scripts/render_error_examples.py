"""Vẽ ảnh minh hoạ lỗi cho báo cáo (thí nghiệm E4) theo configs/error_examples.yaml: mỗi trường hợp một ảnh cắt quanh
loại lỗi cần minh hoạ (bỏ sót / sai lớp / thừa), cộng một ảnh đổi ID trong tracking.
Kết quả: outputs/error_analysis/<tên>.jpg + cases.csv (số lỗi mỗi loại trên từng ảnh).

Màu: xanh lá = phát hiện đúng, đỏ = bỏ sót (nhãn không có dự đoán), cam = đúng chỗ nhưng sai lớp, tím = thừa,
xám = vùng ignore (không chấm).

Ví dụ:
    python scripts/render_error_examples.py
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from vtrack import CLASS_NAMES, IGNORE_CLASS_ID, IGNORE_LABEL
from vtrack.detectors import load_detector
from vtrack.error_analysis import DetectionOutcome, IdSwitch, classify_detections, find_id_switches
from vtrack.label_review import YoloBox, overlap_fraction, read_yolo_file
from vtrack.mot_format import IGNORE_OVERLAP, MotBox, read_cvat_mot, read_cvat_mot_labeled, read_mot_file
from vtrack.video_io import read_frame

OUT_DIR = Path("outputs/error_analysis")
FONT_PATH = "C:/Windows/Fonts/arial.ttf"  # font có dấu tiếng Việt (hàm vẽ chữ của OpenCV không hiển thị được dấu)
COLORS = {"correct": (0, 200, 0), "missed": (255, 0, 0), "wrong_class": (255, 140, 0),
          "false_positive": (220, 0, 220), "ignore": (160, 160, 160)}  # RGB
CROP_MARGIN_PX = 80
# Cắt quanh CỤM lỗi dày nhất: lỗi trong bán kính này (tỉ lệ bề rộng ảnh) quanh lỗi có nhiều hàng xóm nhất. Nếu bao mọi
# lỗi thì một lỗi lẻ ở góc ảnh kéo vùng cắt ra gần cả ảnh, xe nhỏ không còn nhìn thấy.
CLUSTER_RADIUS_FRACTION = 0.15
MIN_CROP = (640, 360)  # (rộng, cao) tối thiểu của ảnh cắt, để còn thấy bối cảnh quanh lỗi
OUTPUT_WIDTH = 1280  # phóng ảnh cắt lên bề rộng này: xe nhỏ ở xa mới nhìn rõ trong báo cáo
TITLE_HEIGHT = 44


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def to_pixels(box: YoloBox, width: int, height: int) -> tuple[int, int, int, int]:
    return (int((box.center_x - box.width / 2) * width), int((box.center_y - box.height / 2) * height),
            int((box.center_x + box.width / 2) * width), int((box.center_y + box.height / 2) * height))


def mot_to_yolo(box: MotBox, class_id: int, width: int, height: int) -> YoloBox:
    return YoloBox(class_id, (box.left + box.width / 2) / width, (box.top + box.height / 2) / height,
                   box.width / width, box.height / height)


def load_case(case: dict, sequences: dict) -> tuple[np.ndarray, list[YoloBox], list[YoloBox]]:
    """(ảnh BGR, nhãn xe, vùng ignore) cho ảnh trong tập gán nhãn hoặc một frame của clip có nhãn tracking CVAT."""
    if "image" in case:
        image_path = Path(case["image"])
        image = cv2.imread(str(image_path))
        if image is None:
            raise SystemExit(f"Không đọc được ảnh: {image_path}")
        labels = read_yolo_file(image_path.parent.parent / "labels" / f"{image_path.stem}.txt")
        return image, [b for b in labels if b.class_id != IGNORE_CLASS_ID], [b for b in labels if b.class_id == IGNORE_CLASS_ID]

    sequence = sequences[case["sequence"]]
    image = read_frame(sequence["video"], case["frame"] - 1)  # nhãn CVAT đánh frame từ 1, video từ 0
    height, width = image.shape[:2]
    label_names = Path(sequence["labels"]).read_text(encoding="utf-8").split()
    at_frame = [(label, box) for label, box in read_cvat_mot_labeled(sequence["ground_truth"], label_names)
                if box.frame == case["frame"]]
    truth = [mot_to_yolo(b, CLASS_NAMES.index(label), width, height) for label, b in at_frame if label in CLASS_NAMES]
    ignore = [mot_to_yolo(b, IGNORE_CLASS_ID, width, height) for label, b in at_frame if label == IGNORE_LABEL]
    return image, truth, ignore


def densest_cluster(boxes: list[tuple[int, int, int, int]], radius: float) -> list[tuple[int, int, int, int]]:
    """Các box có tâm cách tâm của box "đông hàng xóm" nhất không quá `radius` px."""
    centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]

    def neighbours(i: int) -> list[int]:
        return [j for j, c in enumerate(centers)
                if (c[0] - centers[i][0]) ** 2 + (c[1] - centers[i][1]) ** 2 <= radius ** 2]

    best = max(range(len(boxes)), key=lambda i: len(neighbours(i)))
    return [boxes[j] for j in neighbours(best)]


def focus_crop(boxes: list[tuple[int, int, int, int]], width: int, height: int) -> tuple[int, int, int, int]:
    """Vùng cắt bao cụm lỗi dày nhất + lề, tối thiểu MIN_CROP, giữ trong ảnh."""
    if not boxes:
        return 0, 0, width, height
    boxes = densest_cluster(boxes, CLUSTER_RADIUS_FRACTION * width)
    x1, y1 = min(b[0] for b in boxes) - CROP_MARGIN_PX, min(b[1] for b in boxes) - CROP_MARGIN_PX
    x2, y2 = max(b[2] for b in boxes) + CROP_MARGIN_PX, max(b[3] for b in boxes) + CROP_MARGIN_PX
    if x2 - x1 < MIN_CROP[0]:
        center = (x1 + x2) // 2
        x1, x2 = center - MIN_CROP[0] // 2, center + MIN_CROP[0] // 2
    if y2 - y1 < MIN_CROP[1]:
        center = (y1 + y2) // 2
        y1, y2 = center - MIN_CROP[1] // 2, center + MIN_CROP[1] // 2
    x1, y1 = max(0, x1), max(0, y1)
    return x1, y1, min(width, x2), min(height, y2)


def with_title(canvas: Image.Image, title: str) -> Image.Image:
    framed = Image.new("RGB", (canvas.width, canvas.height + TITLE_HEIGHT), (255, 255, 255))
    framed.paste(canvas, (0, TITLE_HEIGHT))
    ImageDraw.Draw(framed).text((10, 8), title, fill=(0, 0, 0), font=font(24))
    return framed


def render_detection(image: np.ndarray, outcome: DetectionOutcome, ignore: list[YoloBox], focus: str, title: str) -> Image.Image:
    height, width = image.shape[:2]
    canvas = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas)
    small = font(18)
    for region in ignore:
        draw.rectangle(to_pixels(region, width, height), outline=COLORS["ignore"], width=2)
    for _, prediction in outcome.correct:
        draw.rectangle(to_pixels(prediction, width, height), outline=COLORS["correct"], width=1)
    for truth, prediction in outcome.wrong_class:
        x1, y1, x2, y2 = to_pixels(prediction, width, height)
        draw.rectangle((x1, y1, x2, y2), outline=COLORS["wrong_class"], width=3)
        draw.text((x1, max(0, y1 - 20)), f"{CLASS_NAMES[truth.class_id]}→{CLASS_NAMES[prediction.class_id]}",
                  fill=COLORS["wrong_class"], font=small)
    for truth in outcome.missed:
        draw.rectangle(to_pixels(truth, width, height), outline=COLORS["missed"], width=3)
    for prediction in outcome.false_positives:
        draw.rectangle(to_pixels(prediction, width, height), outline=COLORS["false_positive"], width=3)

    focus_boxes = {"missed": [to_pixels(b, width, height) for b in outcome.missed],
                   "wrong_class": [to_pixels(p, width, height) for _, p in outcome.wrong_class],
                   "false_positive": [to_pixels(p, width, height) for p in outcome.false_positives]}[focus]
    crop = canvas.crop(focus_crop(focus_boxes, width, height))
    scale = OUTPUT_WIDTH / crop.width
    return with_title(crop.resize((OUTPUT_WIDTH, round(crop.height * scale)), Image.Resampling.LANCZOS), title)


def render_id_switch(video: str, switch: IdSwitch, truth: list[MotBox], predicted: list[MotBox], note: str) -> Image.Image:
    """Hai frame cạnh nhau: frame cuối còn ID cũ và frame xe nhận ID mới, cắt quanh xe."""
    tiles = []
    for frame in (switch.previous_frame, switch.frame):
        image = cv2.cvtColor(read_frame(video, frame - 1), cv2.COLOR_BGR2RGB)
        canvas = Image.fromarray(image)
        draw = ImageDraw.Draw(canvas)
        car = next(b for b in truth if b.track_id == switch.truth_id and b.frame == frame)
        for box in (b for b in predicted if b.frame == frame):
            draw.rectangle((box.left, box.top, box.left + box.width, box.top + box.height), outline=(0, 160, 255), width=2)
            draw.text((box.left, max(0, box.top - 22)), f"ID {box.track_id}", fill=(0, 160, 255), font=font(20))
        draw.rectangle((car.left, car.top, car.left + car.width, car.top + car.height), outline=COLORS["correct"], width=3)
        center_x, center_y = car.left + car.width / 2, car.top + car.height / 2
        x1, y1 = int(max(0, center_x - 240)), int(max(0, center_y - 180))
        tile = canvas.crop((x1, y1, min(canvas.width, x1 + 480), min(canvas.height, y1 + 360)))
        ImageDraw.Draw(tile).text((8, 8), f"frame {frame}", fill=(255, 255, 255), font=font(22))
        tiles.append(tile.resize((640, round(tile.height * 640 / tile.width)), Image.Resampling.LANCZOS))
    joined = Image.new("RGB", (sum(t.width for t in tiles) + 10, max(t.height for t in tiles)), (255, 255, 255))
    joined.paste(tiles[0], (0, 0))
    joined.paste(tiles[1], (tiles[0].width + 10, 0))
    title = (f"Đổi ID: xe #{switch.truth_id} (nhãn) mang ID {switch.old_id} rồi đổi sang ID {switch.new_id} "
             f"(clip test PVD, xanh lá = nhãn, xanh dương = tracker)")
    titled = with_title(joined, title)
    if not note:
        return titled
    return with_title(titled, note)  # dòng giải thích nguyên nhân nằm trên cùng


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/error_examples.yaml")
    parser.add_argument("--sequences", default="configs/tracking_sequences.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    sequences = yaml.safe_load(Path(args.sequences).read_text(encoding="utf-8"))["sequences"]
    detector = load_detector(config["weights"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in config["cases"]:
        image, truth, ignore = load_case(case, sequences)
        # Như bộ chấm mAP: dự đoán nằm phần lớn trong vùng ignore không tính là thừa
        predictions = [p for p in detector.predict(image, config["conf"])
                       if not any(overlap_fraction(p, region) >= IGNORE_OVERLAP for region in ignore)]
        outcome = classify_detections(truth, predictions)
        render_detection(image, outcome, ignore, case["focus"], case["title"]).save(OUT_DIR / f"{case['name']}.jpg", quality=90)
        rows.append({"name": case["name"], "truth": len(truth), "correct": len(outcome.correct),
                     "wrong_class": len(outcome.wrong_class), "missed": len(outcome.missed),
                     "false_positive": len(outcome.false_positives)})
        print(rows[-1], flush=True)

    tracking = config["tracking"]
    sequence = sequences[tracking["sequence"]]
    label_names = Path(sequence["labels"]).read_text(encoding="utf-8").split()
    truth_tracks, _ = read_cvat_mot(sequence["ground_truth"], label_names, IGNORE_LABEL)
    predicted_tracks = read_mot_file(tracking["result"])
    switches = find_id_switches(truth_tracks, predicted_tracks)
    ids_in_frame = {(b.frame, b.track_id) for b in predicted_tracks}
    # Đổi ID "thật": ở frame đổi, ID cũ không còn box nào (track bị mất rồi xe nhận ID mới). Loại trường hợp hai box
    # tracker trùng nhau trên cùng một xe rồi phép ghép đơn giản nhảy qua lại giữa chúng (TrackEval không tính các lần đó).
    genuine = [s for s in switches if (s.frame, s.old_id) not in ids_in_frame]
    print(f"{len(switches)} lần đổi ID theo phép ghép đơn giản (IoU >= 0.5), {len(genuine)} lần ID cũ đã biến mất:",
          [(s.truth_id, s.frame, s.old_id, s.new_id) for s in genuine])
    track_length = {s.truth_id: sum(b.track_id == s.truth_id for b in truth_tracks) for s in genuine}
    # Minh hoạ lần đổi ID của xe có track nhãn dài nhất: xe được theo dõi lâu, dễ thấy ID trước và sau
    chosen = config["tracking"].get("switch_truth_id") or max(track_length, key=track_length.__getitem__)
    switch = next(s for s in genuine if s.truth_id == chosen)
    render_id_switch(sequence["video"], switch, truth_tracks, predicted_tracks, tracking.get("note", "")).save(
        OUT_DIR / "7_doi_id.jpg", quality=90)

    with (OUT_DIR / "cases.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nĐã ghi {OUT_DIR}")


if __name__ == "__main__":
    main()
