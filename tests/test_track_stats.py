import pytest

from vtrack.mot_format import MotBox
from vtrack.track_stats import track_statistics


def test_track_statistics_counts_tracks_and_short_tracks():
    long_track = [MotBox(f, 1, 0, 0, 10, 10, confidence=0.8) for f in range(1, 11)]  # 10 frame
    short_track = [MotBox(f, 2, 50, 0, 10, 10, confidence=0.4) for f in (1, 2)]  # 2 frame

    stats = track_statistics(long_track + short_track, frames=10, short_frames=5)

    assert stats.tracks == 2
    assert stats.boxes_per_frame == pytest.approx(1.2)
    assert stats.median_track_frames == pytest.approx(6.0)
    assert stats.short_track_fraction == pytest.approx(0.5)
    assert stats.median_confidence == pytest.approx(0.8)


def test_track_statistics_of_empty_output_is_zero():
    stats = track_statistics([], frames=5, short_frames=30)

    assert stats.tracks == 0 and stats.boxes_per_frame == 0.0
