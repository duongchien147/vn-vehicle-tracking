import numpy as np
import pytest
import yaml

from vtrack.tracker_config import TrackerVariant, expand_grid, write_variant
from vtrack.mot_format import MotBox
from vtrack.tracking import TrackAgeFilter, filter_young_tracks


def test_expand_grid_builds_every_combination_with_readable_names():
    variants = expand_grid("configs/botsort.yaml", {"track_buffer": [30, 60], "with_reid": [False, True]})

    assert [v.name for v in variants] == [
        "botsort_track_buffer30_with_reidFalse", "botsort_track_buffer30_with_reidTrue",
        "botsort_track_buffer60_with_reidFalse", "botsort_track_buffer60_with_reidTrue",
    ]


def test_write_variant_overrides_only_the_given_keys(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("tracker_type: bytetrack\ntrack_buffer: 30\nmatch_thresh: 0.8\n", encoding="utf-8")

    path = write_variant(TrackerVariant(str(base), {"track_buffer": 90}), tmp_path / "out")

    assert yaml.safe_load(path.read_text(encoding="utf-8")) == {
        "tracker_type": "bytetrack", "track_buffer": 90, "match_thresh": 0.8,
    }


def test_write_variant_rejects_misspelled_parameter(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("track_buffer: 30\n", encoding="utf-8")

    with pytest.raises(ValueError, match="track_bufer"):
        write_variant(TrackerVariant(str(base), {"track_bufer": 90}), tmp_path)


def test_age_filter_hides_tracks_until_old_enough():
    age_filter = TrackAgeFilter(min_age=3)

    assert age_filter.update(np.array([1])).tolist() == [False]
    assert age_filter.update(np.array([1, 2])).tolist() == [False, False]
    assert age_filter.update(np.array([1, 2])).tolist() == [True, False]
    assert age_filter.update(np.array([2])).tolist() == [True]


def test_age_filter_of_one_keeps_everything():
    assert TrackAgeFilter(min_age=1).update(np.array([5, 6])).tolist() == [True, True]


def test_filter_young_tracks_drops_first_frames_of_each_track_and_short_tracks():
    long_track = [MotBox(f, 1, 0, 0, 10, 10) for f in (3, 1, 2, 4)]  # thứ tự lộn xộn như khi đọc file
    short_track = [MotBox(f, 2, 50, 0, 10, 10) for f in (2, 3)]

    kept = filter_young_tracks(long_track + short_track, min_age=3)

    assert [(b.track_id, b.frame) for b in kept] == [(1, 3), (1, 4)]
