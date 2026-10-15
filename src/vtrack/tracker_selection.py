"""Chọn cấu hình tracker trên nhiều tập dev theo tiêu chí chốt trước (configs/tracker_tuning_v6_dev2.yaml):
ứng viên phải không tệ hơn mốc về bỏ sót, HOTA và IDR quá mức cho phép; trong số đạt, chọn ít đổi ID nhất,
và chỉ đổi khỏi mốc khi số lần đổi ID giảm đủ nhiều."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateMetrics:
    """Chỉ số của một cấu hình tracker, trung bình các lần huấn luyện mô hình."""

    name: str
    hota: float  # trên tập dev có nhãn đầy đủ (pvd_goc_c_dev)
    false_negatives: float  # số box bỏ sót, cùng tập dev đó
    idr: float  # trên VnTraffic (nhãn thiếu nên chỉ dùng chỉ số không phụ thuộc phát hiện thừa)
    id_switches: float  # tổng số lần đổi ID trên các tập dev


@dataclass(frozen=True)
class SelectionRules:
    max_fn_increase: float  # tỉ lệ, vd. 0.02 = bỏ sót tăng tối đa 2%
    max_hota_drop: float
    max_idr_drop: float
    min_id_switch_reduction: float  # tỉ lệ giảm đổi ID tối thiểu để bỏ mốc


@dataclass
class SelectionResult:
    chosen: str
    best_passing: str  # ứng viên đạt điều kiện có ít đổi ID nhất (có thể không được chọn nếu giảm chưa đủ)
    id_switch_reduction: float  # của best_passing so với mốc
    failures: dict[str, list[str]]  # ứng viên -> các điều kiện không đạt (rỗng = đạt)


def failed_conditions(candidate: CandidateMetrics, baseline: CandidateMetrics, rules: SelectionRules) -> list[str]:
    failed = []
    if candidate.false_negatives > baseline.false_negatives * (1 + rules.max_fn_increase):
        failed.append("bỏ sót")
    if candidate.hota < baseline.hota - rules.max_hota_drop:
        failed.append("HOTA")
    if candidate.idr < baseline.idr - rules.max_idr_drop:
        failed.append("IDR")
    return failed


def select_tracker(candidates: list[CandidateMetrics], baseline_name: str, rules: SelectionRules) -> SelectionResult:
    """Mốc luôn đạt điều kiện (so với chính nó) nên luôn có ít nhất một ứng viên đạt."""
    baseline = next(c for c in candidates if c.name == baseline_name)
    failures = {c.name: failed_conditions(c, baseline, rules) for c in candidates}
    passing = [c for c in candidates if not failures[c.name]]
    best = min(passing, key=lambda c: (c.id_switches, -c.hota))
    reduction = 1 - best.id_switches / baseline.id_switches if baseline.id_switches else 0.0
    chosen = best.name if reduction >= rules.min_id_switch_reduction else baseline_name
    return SelectionResult(chosen, best.name, reduction, failures)
