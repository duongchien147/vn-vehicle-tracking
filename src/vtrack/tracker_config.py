"""Sinh các biến thể cấu hình tracker (file yaml của ultralytics) từ một lưới tham số, để chỉnh tracker
một cách có hệ thống: mỗi biến thể là một file yaml thật, lưu cùng kết quả nên truy ngược được."""

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from ultralytics.utils.checks import check_yaml


@dataclass(frozen=True)
class TrackerVariant:
    base: str  # file yaml gốc: đường dẫn, hoặc tên cấu hình có sẵn của ultralytics (vd. "ocsort.yaml")
    overrides: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """Tên đọc được, vd. "botsort_track_buffer60_with_reidTrue"."""
        parts = [Path(self.base).stem] + [f"{key}{value}" for key, value in self.overrides.items()]
        return "_".join(parts)


def expand_grid(base: str, grid: dict[str, list[Any]]) -> list[TrackerVariant]:
    """Mọi tổ hợp giá trị trong `grid` (tích Descartes), theo thứ tự khai báo."""
    keys = list(grid)
    return [TrackerVariant(base, dict(zip(keys, values))) for values in itertools.product(*grid.values())]


def write_variant(variant: TrackerVariant, out_dir: Path) -> Path:
    """Ghi cấu hình gốc + giá trị ghi đè ra `<out_dir>/<tên>.yaml`; báo lỗi nếu ghi đè khoá không có trong gốc
    (gõ sai tên tham số thì ultralytics sẽ lặng lẽ bỏ qua -> thí nghiệm vô nghĩa mà không ai biết)."""
    base_path = Path(str(check_yaml(variant.base)))  # tìm cả trong thư mục cấu hình tracker có sẵn của ultralytics
    config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    unknown = set(variant.overrides) - set(config)
    if unknown:
        raise ValueError(f"{variant.base} không có tham số {sorted(unknown)}")
    config.update(variant.overrides)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{variant.name}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path
