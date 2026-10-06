import pytest

from vtrack.frame_sampling import parse_timestamp, sample_timestamps


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
