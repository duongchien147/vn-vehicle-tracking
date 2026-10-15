"""Chọn tracker cho v6 trên hai tập dev theo tiêu chí chốt trước trong configs/tracker_tuning_v6_dev2.yaml:
đọc kết quả chấm pvd_goc_c_dev và VnTraffic, lấy trung bình hai seed, áp điều kiện, in bảng + cấu hình được chọn.
Kết quả: outputs/eval_tracking/tuning_v6_dev2/selection.csv

Ví dụ:
    python scripts/select_tracker_v6.py
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import yaml

from vtrack.tracker_selection import CandidateMetrics, SelectionRules, select_tracker

RESULTS = Path("outputs/eval_tracking")


def config_metrics(result_dirs: list[str]) -> dict[str, dict[str, float]]:
    """{tên cấu hình: trung bình các seed của từng chỉ số}; tên cấu hình = tên lần chạy bỏ phần tên trọng số."""
    per_config: dict[str, list[dict]] = defaultdict(list)
    for result_dir in result_dirs:
        with (RESULTS / result_dir / "results.csv").open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if int(row["min_track_age"]) != 1:
                    continue
                config = row["run"].removeprefix(Path(row["weights"]).stem + "-")
                per_config[config].append(row)
    keys = ("hota", "false_negatives", "idr", "id_switches")
    return {config: {k: sum(float(r[k]) for r in rows) / len(rows) for k in keys} | {"seeds": len(rows)}
            for config, rows in per_config.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid", default="configs/tracker_tuning_v6_dev2.yaml")
    parser.add_argument("--dev", nargs="+", default=["tuning_v6_dev2", "tuning_v6_dev2_nms"],
                        help="Thư mục kết quả trên pvd_goc_c_dev")
    parser.add_argument("--vntraffic", nargs="+", default=["tuning_v6", "tuning_v6_nms"],
                        help="Thư mục kết quả trên VnTraffic (cùng ứng viên, cùng hai seed)")
    args = parser.parse_args()

    selection = yaml.safe_load(Path(args.grid).read_text(encoding="utf-8"))["selection"]
    rules = SelectionRules(selection["max_fn_increase"], selection["max_hota_drop"], selection["max_idr_drop"],
                           selection["min_id_switch_reduction"])
    baseline = Path(selection["baseline_tracker"]).stem
    dev, vntraffic = config_metrics(args.dev), config_metrics(args.vntraffic)
    missing = set(dev) ^ set(vntraffic)
    if missing:
        raise SystemExit(f"Ứng viên chỉ có kết quả trên một tập dev: {sorted(missing)}")

    candidates = [
        CandidateMetrics(name, hota=dev[name]["hota"], false_negatives=dev[name]["false_negatives"],
                         idr=vntraffic[name]["idr"], id_switches=dev[name]["id_switches"] + vntraffic[name]["id_switches"])
        for name in sorted(dev)
    ]
    result = select_tracker(candidates, baseline, rules)

    rows = []
    for c in sorted(candidates, key=lambda c: c.id_switches):
        failed = result.failures[c.name]
        rows.append({"config": c.name, "hota_dev2": round(c.hota, 4), "fn_dev2": round(c.false_negatives, 1),
                     "idsw_dev2": round(dev[c.name]["id_switches"], 1), "idr_vntraffic": round(c.idr, 4),
                     "idsw_vntraffic": round(vntraffic[c.name]["id_switches"], 1), "idsw_total": round(c.id_switches, 1),
                     "failed": " + ".join(failed) or "đạt"})
        print(f"{c.name:80s} HOTA {c.hota:.3f} | FN {c.false_negatives:7.1f} | IDR VnT {c.idr:.3f} | "
              f"đổi ID {dev[c.name]['id_switches']:5.1f} + {vntraffic[c.name]['id_switches']:5.1f} = {c.id_switches:6.1f}"
              f" | {' + '.join(failed) or 'đạt'}")
    print(f"\nỨng viên đạt có ít đổi ID nhất: {result.best_passing} (giảm {result.id_switch_reduction:.1%} so với mốc)")
    print(f"Cấu hình được chọn: {result.chosen}")

    out = RESULTS / args.dev[0] / "selection.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Đã ghi {out}")


if __name__ == "__main__":
    main()
