from vtrack.tracker_selection import CandidateMetrics, SelectionRules, select_tracker

RULES = SelectionRules(max_fn_increase=0.02, max_hota_drop=0.005, max_idr_drop=0.005, min_id_switch_reduction=0.10)
BASE = CandidateMetrics("base", hota=0.70, false_negatives=1000, idr=0.83, id_switches=100)


def test_candidate_missing_more_vehicles_is_rejected_even_with_fewer_id_switches():
    fewer_switches = CandidateMetrics("strict", hota=0.71, false_negatives=1100, idr=0.84, id_switches=40)
    result = select_tracker([BASE, fewer_switches], "base", RULES)
    assert result.failures["strict"] == ["bỏ sót"]
    assert result.chosen == "base"


def test_fewest_id_switches_among_passing_candidates_is_chosen():
    a = CandidateMetrics("a", hota=0.70, false_negatives=1010, idr=0.83, id_switches=80)
    b = CandidateMetrics("b", hota=0.698, false_negatives=1015, idr=0.828, id_switches=70)
    result = select_tracker([BASE, a, b], "base", RULES)
    assert result.chosen == "b"
    assert abs(result.id_switch_reduction - 0.30) < 1e-9


def test_small_reduction_keeps_baseline():
    slightly_better = CandidateMetrics("slight", hota=0.70, false_negatives=1000, idr=0.83, id_switches=95)
    result = select_tracker([BASE, slightly_better], "base", RULES)
    assert result.best_passing == "slight"
    assert result.chosen == "base"


def test_hota_and_idr_drops_are_reported():
    worse = CandidateMetrics("worse", hota=0.69, false_negatives=1000, idr=0.82, id_switches=50)
    result = select_tracker([BASE, worse], "base", RULES)
    assert result.failures["worse"] == ["HOTA", "IDR"]
