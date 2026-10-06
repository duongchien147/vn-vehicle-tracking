from vtrack.camera_motion import MotionSample, stable_segments


def samples_from(shifts: list[float], cut_at: int | None = None) -> list[MotionSample]:
    """Mỗi mẫu cách nhau 1 giây."""
    return [MotionSample(float(i), shift, 0.0, i == cut_at) for i, shift in enumerate(shifts)]


def test_whole_video_steady_is_one_segment():
    assert stable_segments(samples_from([0.2] * 100), min_length_s=30) == [(0.0, 99.0)]


def test_zoom_or_cut_splits_segments_and_short_parts_are_dropped():
    shifts = [0.2] * 40 + [50.0] + [0.2] * 10 + [0.2] * 50  # lia máy ở giây 40
    segments = stable_segments(samples_from(shifts, cut_at=51), min_length_s=30)
    # 0-40 giữ lại; 41-51 quá ngắn bị bỏ; sau cắt cảnh ở giây 51: 52-100
    assert segments == [(0.0, 40.0), (52.0, 100.0)]


def test_shaky_camera_has_no_segment():
    assert stable_segments(samples_from([5.0, 0.1] * 50), min_length_s=10) == []
