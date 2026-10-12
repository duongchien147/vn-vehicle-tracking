import numpy as np
import pytest

from vtrack.frame_sampling import (
    frame_signature,
    keep_distinct,
    parse_frame_name,
    parse_timestamp,
    sample_timestamps,
)


def test_parse_timestamp():
    assert parse_timestamp("0:35") == 35
    assert parse_timestamp("1:35") == 95
    assert parse_timestamp("1:02:03") == 3723


def test_sample_timestamps_excludes_end():
    assert sample_timestamps(35, 95, 2) == [35 + 2 * i for i in range(30)]  # 30 frame, không có 95
    assert sample_timestamps(0, 10, 3) == [0, 3, 6, 9]
    assert sample_timestamps(5, 5, 1) == []


def test_sample_timestamps_rejects_bad_step():
    with pytest.raises(ValueError):
        sample_timestamps(0, 10, 0)


def test_parse_frame_name():
    assert parse_frame_name("pvd_goc_b_00m05s") == ("pvd_goc_b", 5.0)
    assert parse_frame_name("hanoi_01m45s") == ("hanoi", 105.0)
    assert parse_frame_name("frame_000031") is None


def test_keep_distinct_drops_repeated_frames():
    rng = np.random.default_rng(0)
    scene_a = rng.integers(0, 255, (90, 160, 3), dtype=np.uint8)
    scene_b = rng.integers(0, 255, (90, 160, 3), dtype=np.uint8)
    noisy_a = np.clip(scene_a.astype(int) + 1, 0, 255).astype(np.uint8)  # cùng cảnh, nén lại khác 1 mức xám
    signatures = [frame_signature(f) for f in (scene_a, scene_b, noisy_a, scene_b)]
    assert keep_distinct(signatures, min_distance=2.0) == [0, 1]
