"""Kiểm tiêu chí nhận YOLO26s-v6 (chốt trước trong configs/accept_yolo26s_v6.yaml) từ kết quả chấm trên dev của
scripts/rerun_yolo26s_v6_dev.sh: mAP50 dev đêm, mAP50 val Hà Nội, IDR VnTraffic (chưa lọc box trùng), từng seed.

Ví dụ:
    python scripts/check_accept_yolo26s_v6.py
"""

import argparse
import csv
from pathlib import Path

import yaml


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/accept_yolo26s_v6.yaml")
    args = parser.parse_args()
    thresholds = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["thresholds"]

    night = {r["model"]: float(r["main_mAP50"])
             for r in read_rows(Path("outputs/eval_hanoi/yolo26s_v6_dev_nightdev/val/results.csv"))}
    val = {r["model"]: float(r["main_mAP50"]) for r in read_rows(Path("outputs/eval_hanoi/yolo26s_v6_dev/val/results.csv"))}
    idr = {Path(r["weights"]).stem: float(r["idr"])
           for r in read_rows(Path("outputs/eval_tracking/yolo26s_v6_tracker_vntraffic/results.csv"))
           if r["agnostic_nms_iou"] in ("", "None") and int(r["min_track_age"]) == 1}

    checks = []
    for model in sorted(night):
        checks += [
            (model, "dev đêm", night[model], night[model] > thresholds["night_dev_map50_gt"],
             f"> {thresholds['night_dev_map50_gt']}"),
            (model, "val Hà Nội", val[model], val[model] >= thresholds["val_hanoi_map50_ge"],
             f">= {thresholds['val_hanoi_map50_ge']}"),
            (model, "IDR VnTraffic", idr[model], idr[model] >= thresholds["vntraffic_idr_ge"],
             f">= {thresholds['vntraffic_idr_ge']}"),
        ]
    for model, name, value, passed, rule in checks:
        print(f"{model:32s} {name:14s} {value:.4f} ({rule}): {'ĐẠT' if passed else 'KHÔNG ĐẠT'}")
    accepted = all(passed for *_, passed, _ in checks)
    print(f"\nYOLO26s-v6 {'ĐƯỢC NHẬN' if accepted else 'KHÔNG được nhận'} theo {args.config}")


if __name__ == "__main__":
    main()
