"""Web demo Streamlit: chọn clip có sẵn hoặc tải video lên -> phát hiện, theo dõi, đếm xe qua vạch -> xem video kết quả,
bảng đếm theo lớp/hướng, biểu đồ theo thời gian, tải CSV. Chỉ là lớp giao diện: mọi xử lý nằm trong
vtrack.pipeline.process_video() (giống hệt CLI scripts/infer_video.py). Chế độ "Kết quả GPU xử lý sẵn" chỉ nạp lại
video + CSV mà notebooks/kaggle_demo_gpu đã tạo bằng cùng pipeline trên GPU T4 (trên CPU không chạy kịp thời gian thực).

Chạy:  .venv\\Scripts\\streamlit.exe run app\\streamlit_app.py
"""

import json
from dataclasses import asdict
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
import supervision as sv
from ultralytics import YOLO

from vtrack import CLASS_NAMES
from vtrack.models import DEFAULT_WEIGHTS, load_model
from vtrack.pipeline import VideoResult, process_video
from vtrack.saved_runs import find_saved_runs, load_saved_run
from vtrack.tracking import DEFAULT_DETECTION_CONF, DEFAULT_TRACKER
from vtrack.video_io import read_frame, safe_upload_name

VIDEO_DIR = Path("data/videos")
OUTPUT_DIR = Path("outputs/streamlit")
GPU_DEMO_DIR = Path("outputs/kaggle/vtrack-demo-gpu/demo")  # tải về bằng: scripts/kaggle_train.py download
# Clip mở sẵn: clip test PVD có nhãn tracking — video khớp bảng HOTA và sai số đếm trong báo cáo
GPU_DEFAULT_CLIP = "pvd_goc_a"
# Chế độ đầu tiên là mặc định: trình chiếu video GPU xử lý sẵn (mượt, có ngay); hai chế độ chạy trực tiếp trên CPU
# giữ lại để chạy video mới khi được hỏi (chậm ~10 frame/s)
MODES = ("Kết quả GPU xử lý sẵn", "Clip có sẵn (chạy CPU)", "Tải video lên (chạy CPU)")
UPLOAD_SUFFIXES = ("mp4", "avi", "mov")
# Mô hình chọn sẵn trong demo: cùng trọng số v5 nhưng chạy bằng OpenVINO — cả pipeline 6.9 so với 5.3 frame/s trên
# CPU (outputs/benchmark_speed/), mAP50 val Hà Nội 0.916 so với 0.919 của bản PyTorch (chênh trong mức dao động)
DEMO_WEIGHTS = "weights/yolo26n_vn_v5_best_openvino_model"
# Tên hiển thị theo thứ tự trong danh sách; mô hình chưa có trong weights/ thì không hiện
MODEL_LABELS = {
    DEMO_WEIGHTS: "v5 — OpenVINO (mặc định, nhanh nhất trên CPU)",
    DEFAULT_WEIGHTS: "v5 — PyTorch",
    "weights/yolo26n_vn_v3_best.pt": "v3 — trước khi thêm dữ liệu camera giao thông UVH-26",
    "weights/yolo26n_vn_v4_best.pt": "v4 — v3 ở imgsz 1280",
    "weights/yolo26n_vn_v2_best.pt": "v2",
    "weights/yolo26n.pt": "COCO gốc — baseline chưa fine-tune",
}
TRACKER_LABELS = {
    DEFAULT_TRACKER: "BoT-SORT đã chỉnh — mặc định (camera đứng yên)",
    "configs/botsort_tuned.yaml": "BoT-SORT đã chỉnh + bù rung camera (GMC) — video quay tay/rung, chậm hơn",
    "configs/botsort.yaml": "BoT-SORT mặc định",
    "configs/bytetrack.yaml": "ByteTrack mặc định",
}
DEFAULT_LINE = (0.0, 0.6, 1.0, 0.6)
DEFAULT_CLIP = "nga_tu_hanoi_a"  # cảnh ban ngày, đông xe máy — đúng phạm vi đồ án
# Không đưa vào danh sách: clip kiểm tra pipeline (cao tốc nước ngoài) và 2 clip Hà Nội trùng đoạn đã lấy frame
# val (b) / train (c) — trình diễn trên đó sẽ cho kết quả đẹp giả tạo (xem data/videos/SOURCES.md)
HIDDEN_CLIPS = {"sample_8s.mp4", "nga_tu_hanoi_b.mp4", "nga_tu_hanoi_c.mp4"}


@st.cache_resource(show_spinner="Đang nạp mô hình...")
def cached_model(weights: str) -> YOLO:
    """Nạp mỗi mô hình một lần cho cả phiên; process_video tự tạo tracker mới cho mỗi video (persist=False ở frame đầu)."""
    return load_model(weights)


def available_models() -> list[str]:
    return [weights for weights in MODEL_LABELS if Path(weights).exists()]


def choose_source(mode: str) -> Path | None:
    if mode == MODES[1]:
        clips = sorted(p for p in VIDEO_DIR.glob("*.mp4") if p.name not in HIDDEN_CLIPS)
        if not clips:
            st.sidebar.warning(f"Không có video trong {VIDEO_DIR}")
            return None
        default = next((i for i, p in enumerate(clips) if p.stem == DEFAULT_CLIP), 0)
        return st.sidebar.selectbox("Clip", clips, index=default, format_func=lambda p: p.stem)
    uploaded = st.sidebar.file_uploader("Video", type=list(UPLOAD_SUFFIXES))
    if uploaded is None:
        return None
    name = safe_upload_name(uploaded.name, UPLOAD_SUFFIXES)
    if name is None:
        st.sidebar.error("Tên file không hợp lệ")
        return None
    upload_dir = (OUTPUT_DIR / "uploads").resolve()
    path = upload_dir / name
    if path.resolve().parent != upload_dir:  # chốt chặn cuối: đường dẫn phải nằm đúng trong uploads/
        st.sidebar.error("Tên file không hợp lệ")
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(uploaded.getbuffer())
    return path


def choose_line() -> tuple[float, float, float, float]:
    with st.sidebar.expander("Vạch đếm (toạ độ tương đối 0–1)", expanded=True):
        left, right = st.columns(2)
        x1 = left.slider("x đầu", 0.0, 1.0, DEFAULT_LINE[0], 0.01)
        y1 = left.slider("y đầu", 0.0, 1.0, DEFAULT_LINE[1], 0.01)
        x2 = right.slider("x cuối", 0.0, 1.0, DEFAULT_LINE[2], 0.01)
        y2 = right.slider("y cuối", 0.0, 1.0, DEFAULT_LINE[3], 0.01)
    return x1, y1, x2, y2


def show_line_preview(source: Path, line: tuple[float, float, float, float]) -> None:
    frame = read_frame(source)
    height, width = frame.shape[:2]
    start = (round(line[0] * width), round(line[1] * height))
    end = (round(line[2] * width), round(line[3] * height))
    cv2.line(frame, start, end, (0, 0, 255), max(2, width // 400))
    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Frame đầu và vạch đếm (đỏ)", width="stretch")


def show_results(result: VideoResult, duration_s: float, settings: dict, events_csv: Path, hardware: str) -> None:
    st.video(str(result.output_path))
    a, b, c = st.columns(3)
    a.metric("Frame đã xử lý", result.frames)
    b.metric(f"Tốc độ cả quy trình (frame/s, {hardware})", f"{result.processing_fps:.1f}")
    c.metric("Kích thước ảnh vào mô hình", result.imgsz)

    st.subheader("Số xe qua vạch")
    counts = result.counts()
    table = pd.DataFrame(
        [{"lớp": name, "vào": counts[name]["in"], "ra": counts[name]["out"],
          "tổng": counts[name]["in"] + counts[name]["out"]} for name in CLASS_NAMES]
    )
    st.dataframe(table, hide_index=True, width="stretch")
    st.line_chart(pd.DataFrame(result.cumulative_counts(duration_s)).set_index("time_s"),
                  x_label="thời gian (giây)", y_label="số xe cộng dồn")

    events = pd.DataFrame([asdict(e) for e in result.events])
    with st.expander(f"Danh sách {len(events)} sự kiện qua vạch"):
        st.dataframe(events, hide_index=True, width="stretch")
    left, right = st.columns(2)
    left.download_button("Tải CSV sự kiện", events_csv.read_bytes(), events_csv.name, "text/csv")
    right.download_button("Tải video kết quả", result.output_path.read_bytes(), result.output_path.name, "video/mp4")
    with st.expander("Cấu hình đã dùng (để tái lập kết quả)"):
        st.code(json.dumps({**settings, "imgsz": result.imgsz}, indent=2, ensure_ascii=False), language="json")


def show_gpu_demo() -> None:
    clips = find_saved_runs(GPU_DEMO_DIR)
    if not clips:
        st.info(f"Chưa có kết quả trong {GPU_DEMO_DIR}: chạy kernel notebooks/kaggle_demo_gpu rồi tải output về.")
        return
    default = clips.index(GPU_DEFAULT_CLIP) if GPU_DEFAULT_CLIP in clips else 0
    clip = st.sidebar.selectbox("Clip", clips, index=default)
    run = load_saved_run(GPU_DEMO_DIR, clip)
    st.markdown(
        f"**{clip}** — xử lý sẵn trên GPU Tesla T4 (Kaggle) bằng đúng pipeline của dự án: "
        f"`{Path(run.config['weights']).name}` + `{Path(run.config['tracker']).name}`, vạch đếm {run.config['line']}. "
        "Tốc độ dưới đây gồm cả vẽ box và mã hoá video; riêng phát hiện + theo dõi đạt ~42 frame/s."
    )
    video_fps = sv.VideoInfo.from_video_path(str(run.result.output_path)).fps
    show_results(run.result, run.result.frames / video_fps, run.config, run.events_csv, hardware="GPU T4")


def main() -> None:
    st.set_page_config(page_title="Theo dõi phương tiện giao thông", layout="wide")
    st.title("Phát hiện, phân loại và theo dõi phương tiện giao thông Việt Nam")
    st.caption("YOLO26 fine-tune trên dữ liệu Việt Nam + BoT-SORT; đếm xe qua vạch là ứng dụng minh hoạ của tracking.")

    mode = st.sidebar.radio("Video", MODES, horizontal=True)
    if mode == MODES[0]:
        show_gpu_demo()
        return

    source = choose_source(mode)
    models = available_models()
    weights = st.sidebar.selectbox("Mô hình", models, format_func=lambda w: MODEL_LABELS[w])
    tracker = st.sidebar.selectbox("Tracker", list(TRACKER_LABELS), format_func=lambda t: TRACKER_LABELS[t])
    conf = st.sidebar.slider(
        "Ngưỡng tin cậy detection", 0.05, 0.5, DEFAULT_DETECTION_CONF, 0.05,
        help="Để thấp (0.1): tracker dùng cả detection điểm thấp để nối xe bị che; tạo track mới do cấu hình tracker quyết định.",
    )
    vid_stride = st.sidebar.select_slider(
        "Xử lý 1 trên mỗi N frame", [1, 2, 3], 1,
        help="N > 1 nhanh hơn nhưng xe dịch xa hơn giữa hai frame -> dễ đổi ID; thời gian giữ track đã mất cũng dài gấp N.",
    )
    line = choose_line()
    if source is None:
        st.info("Chọn một clip có sẵn hoặc tải video lên ở thanh bên trái.")
        return

    info = sv.VideoInfo.from_video_path(str(source))
    video_seconds = (info.total_frames or 0) / info.fps
    max_seconds = st.sidebar.slider("Chỉ xử lý N giây đầu", 5, max(5, int(video_seconds)), min(20, max(5, int(video_seconds))),
                                     help="Trên CPU ~3–5 frame/s: 20 giây video 30 fps mất khoảng 2–3 phút.")
    max_frames = int(max_seconds * info.fps / vid_stride)

    col_preview, col_info = st.columns([3, 1])
    with col_preview:
        show_line_preview(source, line)
    with col_info:
        st.markdown(f"**{source.name}**  \n{info.width}×{info.height}, {info.fps:g} fps, {video_seconds:.0f} giây")
        run = st.button("Chạy", type="primary", width="stretch")

    settings = {"source": str(source), "weights": weights, "tracker": tracker, "conf": conf,
                "vid_stride": vid_stride, "line": line, "max_seconds": max_seconds}
    if run:
        progress = st.progress(0.0, text="Đang xử lý...")

        def on_progress(done: int, total: int) -> None:
            if total and (done % 10 == 0 or done == total):
                progress.progress(min(done / total, 1.0), text=f"Đang xử lý: {done}/{total} frame")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        result = process_video(
            source, OUTPUT_DIR / f"{source.stem}_tracked.mp4", cached_model(weights), tracker=tracker, line=line,
            conf=conf, vid_stride=vid_stride, max_frames=max_frames, on_progress=on_progress,
        )
        progress.empty()
        events_csv = result.write_events_csv(result.output_path.with_name(result.output_path.stem + "_events.csv"))
        st.session_state["last_run"] = (result, max_seconds, settings, events_csv, "CPU")

    if "last_run" in st.session_state:
        show_results(*st.session_state["last_run"])


if __name__ == "__main__":
    main()
