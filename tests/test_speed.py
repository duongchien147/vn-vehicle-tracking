import pytest

from vtrack.speed import round_order, summarize


def test_summarize_reports_median_and_quartiles():
    result = summarize([10, 20, 30, 40, 1000])  # một frame chậm bất thường không kéo lệch trung vị
    assert result.ms_per_frame == 30
    assert (result.ms_q1, result.ms_q3) == (20, 40)
    assert result.fps == pytest.approx(1000 / 30)


def test_round_order_alternates_direction():
    names = ["a", "b", "c"]
    assert round_order(names, 0) == ["a", "b", "c"]
    assert round_order(names, 1) == ["c", "b", "a"]
    assert round_order(names, 2) == ["a", "b", "c"]
