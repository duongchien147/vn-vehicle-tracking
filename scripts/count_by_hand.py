"""Đếm tay số xe qua vạch trên video (nhãn chuẩn cho sai số đếm ở clip chưa có nhãn tracking, vd. clip đêm).

Xem video gốc (không có box của mô hình) với vạch đếm vẽ sẵn; mỗi khi ĐÁY xe (chỗ bánh chạm đường) đi qua vạch thì
bấm phím theo lớp xe. Mỗi lần bấm được lưu ngay vào CSV, mở lại thì làm tiếp. Hướng dẫn: docs/huong_dan_dem_tay.md.

Phím:  m / c / b / t      xe máy / ô tô / buýt / tải đi theo chiều RA (mũi tên cam)
       M / C / B / T      (giữ Shift) cùng các lớp đó đi theo chiều VÀO (mũi tên xanh)
       Space              dừng / chạy tiếp          , .   lùi / tới 1 frame (khi dừng)
       a / d              lùi / tới 2 giây          - =   chậm hơn / nhanh hơn
       u                  xoá lượt gần frame hiện tại nhất (trong 2 giây)
       q hoặc Esc         lưu và thoát

Ví dụ:
    python scripts/count_by_hand.py --source data/videos/ham_chui_dem_1.mp4
Kết quả: data/count_gt/<tên clip>/manual_counts.csv + count_config.json (vạch, video, đoạn đã xem).
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from vtrack.manual_count import Crossing, count_crossings, in_direction, read_crossings, write_crossings

CLASS_KEYS = {"m": "motorbike", "c": "car", "b": "bus", "t": "truck"}
SPEEDS = [0.1, 0.25, 0.5, 0.75, 1.0]  # tốc độ phát so với thời gian thật; mặc định 0.25 để kịp nhìn từng xe
SEEK_S = 2.0
UNDO_WINDOW_S = 2.0
IN_COLOR, OUT_COLOR, LINE_COLOR = (80, 200, 80), (0, 140, 255), (0, 255, 255)


def read_frame(capture: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    """Nhảy tới frame `frame_index` (đánh từ 0, cùng cách đánh số với sự kiện của pipeline) và đọc nó."""
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok:
        raise SystemExit(f"Không đọc được frame {frame_index}")
    return frame


def draw_overlay(frame: np.ndarray, line: tuple[float, float, float, float], crossings: list[Crossing],
                 frame_index: int, fps: float, speed: float, paused: bool) -> np.ndarray:
    """Vẽ vạch, hai mũi tên chiều vào/ra, số đã đếm và các lượt gần thời điểm đang xem (để tránh bấm trùng khi tua)."""
    height, width = frame.shape[:2]
    canvas = frame.copy()
    start = (int(line[0] * width), int(line[1] * height))
    end = (int(line[2] * width), int(line[3] * height))
    cv2.line(canvas, start, end, LINE_COLOR, 2)
    dx, dy = in_direction(line, width, height)
    middle = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
    for sign, color, label in ((1, IN_COLOR, "VAO (Shift)"), (-1, OUT_COLOR, "RA")):
        tip = (int(middle[0] + sign * dx * 80), int(middle[1] + sign * dy * 80))
        cv2.arrowedLine(canvas, middle, tip, color, 3, tipLength=0.3)
        cv2.putText(canvas, label, (tip[0] + 8, tip[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    counts = count_crossings(crossings)
    now = frame_index / fps
    lines = [
        f"t = {now:6.2f} s  (frame {frame_index})  toc do x{speed:g}  {'DUNG' if paused else ''}",
        "RA:  " + "  ".join(f"{name} {counts[name]['out']}" for name in CLASS_KEYS.values()),
        "VAO: " + "  ".join(f"{name} {counts[name]['in']}" for name in CLASS_KEYS.values()),
        "gan day: " + ", ".join(f"{c.time_s:.1f}s {c.class_name} {c.direction}"
                                for c in crossings if abs(c.time_s - now) <= 3.0),
    ]
    for row, text in enumerate(lines):
        position = (10, 30 + 30 * row)
        cv2.putText(canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
        cv2.putText(canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True)
    parser.add_argument("--line", type=float, nargs=4, default=[0.0, 0.6, 1.0, 0.6], metavar=("X1", "Y1", "X2", "Y2"),
                        help="Vạch đếm, toạ độ tương đối [0, 1] — phải trùng vạch của lần chạy pipeline cần chấm")
    parser.add_argument("--out-dir", default="data/count_gt")
    args = parser.parse_args()
    source = Path(args.source)
    line = tuple(args.line)
    out_dir = Path(args.out_dir) / source.stem
    counts_path = out_dir / "manual_counts.csv"
    crossings = read_crossings(counts_path) if counts_path.exists() else []

    capture = cv2.VideoCapture(str(source))
    fps = capture.get(cv2.CAP_PROP_FPS)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    config_path = out_dir / "count_config.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {"source": str(source), "line": list(line), "fps": fps, "frames": total, "anchor": "bottom_center"}
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8"))["line"] != list(line):
        raise SystemExit(f"{config_path} đã đếm với vạch khác — dùng đúng vạch cũ hoặc thư mục khác (--out-dir)")
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    window = f"Dem tay - {source.name}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)  # kéo giãn / phóng to cửa sổ được
    frame_index, speed_index, paused = 0, SPEEDS.index(0.25), True
    frame = read_frame(capture, frame_index)
    while True:
        started = time.perf_counter()
        cv2.imshow(window, draw_overlay(frame, line, crossings, frame_index, fps, SPEEDS[speed_index], paused))
        frame_ms = 1000 / (fps * SPEEDS[speed_index])
        key = cv2.waitKey(0 if paused else max(1, int(frame_ms - (time.perf_counter() - started) * 1000)))
        char = chr(key & 0xFF) if key != -1 else ""

        if char.lower() in CLASS_KEYS:
            direction = "in" if char.isupper() else "out"
            crossings.append(Crossing(frame_index, frame_index / fps, CLASS_KEYS[char.lower()], direction))
            write_crossings(counts_path, crossings)
        elif char == "u" and crossings:
            nearest = min(crossings, key=lambda c: abs(c.frame - frame_index))
            if abs(nearest.frame - frame_index) <= UNDO_WINDOW_S * fps:
                crossings.remove(nearest)
                write_crossings(counts_path, crossings)
        elif char == " ":
            paused = not paused
        elif char in {",", ".", "a", "d"}:
            step = {",": -1, ".": 1, "a": -int(SEEK_S * fps), "d": int(SEEK_S * fps)}[char]
            frame_index = int(np.clip(frame_index + step, 0, total - 1))
            frame = read_frame(capture, frame_index)
        elif char in {"-", "="}:
            speed_index = int(np.clip(speed_index + (1 if char == "=" else -1), 0, len(SPEEDS) - 1))
        elif char == "q" or key == 27:
            break

        if not paused:
            ok, next_frame = capture.read()
            if ok:
                frame_index, frame = frame_index + 1, next_frame
            else:  # hết video: dừng ở frame cuối
                paused = True

    write_crossings(counts_path, crossings)
    capture.release()
    cv2.destroyAllWindows()
    counts = count_crossings(crossings)
    print(f"Đã lưu {len(crossings)} lượt vào {counts_path}")
    for name, by_direction in counts.items():
        print(f"  {name:10s} vào {by_direction['in']:3d}  ra {by_direction['out']:3d}")


if __name__ == "__main__":
    main()
