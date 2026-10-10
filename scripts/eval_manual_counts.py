"""Sai số đếm xe qua vạch so với số NGƯỜI đếm tay (scripts/count_by_hand.py) trên các clip chưa có nhãn tracking.

Với mỗi clip có data/count_gt/<clip>/manual_counts.csv và mỗi thư mục kết quả pipeline (--runs: chứa
<clip>_events.csv + <clip>_run_config.json do infer_video.py / kernel demo GPU ghi), so số lượt theo lớp và hướng,
rồi ghép từng lượt theo thời điểm để liệt kê lượt bị sót, đếm thừa, sai lớp (mở video ở đúng giây đó để xem lại).
Vạch của lần chạy pipeline phải trùng vạch người đếm (kiểm trong count_config.json), nếu không thì từ chối.
Kết quả: outputs/eval_manual_counts/<--name>/{results.csv, mistakes.csv}.

Ví dụ:
    python scripts/eval_manual_counts.py --runs outputs/kaggle/vtrack-demo-gpu/demo outputs/manual_count_runs/v5
"""

import argparse
import csv
import json
from pathlib import Path

from vtrack.count_eval import compare_counts
from vtrack.manual_count import MATCH_TOLERANCE_S, count_crossings, match_crossings, read_crossings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True, help="Thư mục chứa <clip>_events.csv + _run_config.json")
    parser.add_argument("--count-dir", default="data/count_gt")
    parser.add_argument("--tolerance", type=float, default=MATCH_TOLERANCE_S, help="Lệch thời gian tối đa khi ghép (s)")
    parser.add_argument("--name", default="default", help="Tên thí nghiệm -> outputs/eval_manual_counts/<tên>")
    args = parser.parse_args()
    out_dir = Path("outputs/eval_manual_counts") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    results, mistakes = [], []
    for counts_path in sorted(Path(args.count_dir).glob("*/manual_counts.csv")):
        clip = counts_path.parent.name
        count_config = json.loads((counts_path.parent / "count_config.json").read_text(encoding="utf-8"))
        truth = read_crossings(counts_path)
        for run_dir in map(Path, args.runs):
            events_path = run_dir / f"{clip}_events.csv"
            if not events_path.exists():
                continue
            run_config = json.loads((run_dir / f"{clip}_run_config.json").read_text(encoding="utf-8"))
            if run_config["line"] != count_config["line"]:
                raise SystemExit(f"{clip}: vạch pipeline {run_config['line']} khác vạch đếm tay {count_config['line']}")
            model = Path(run_config["weights"]).name
            predicted = read_crossings(events_path)
            for row in compare_counts(count_crossings(truth), count_crossings(predicted)):
                results.append({"clip": clip, "model": model, "class": row.class_name,
                                "truth_in": row.truth_in, "truth_out": row.truth_out,
                                "predicted_in": row.predicted_in, "predicted_out": row.predicted_out,
                                "absolute_error": row.absolute_error, "relative_error": round(row.relative_error, 4)})
            match = match_crossings(truth, predicted, args.tolerance)
            for kind, crossing, other in (
                *(("missed", c, "") for c in match.missed),
                *(("extra", c, "") for c in match.extra),
                *(("wrong_class", t, p.class_name) for t, p in match.wrong_class),
            ):
                mistakes.append({"clip": clip, "model": model, "kind": kind, "time_s": round(crossing.time_s, 2),
                                 "class_name": crossing.class_name, "predicted_class": other,
                                 "direction": crossing.direction})
            total = results[-1]
            print(f"  {clip:16s} {model:35s} thật {total['truth_in']}/{total['truth_out']} (vào/ra), "
                  f"đếm {total['predicted_in']}/{total['predicted_out']}, sai số {total['absolute_error']} | "
                  f"ghép {len(match.pairs)}, sót {len(match.missed)}, thừa {len(match.extra)}, "
                  f"sai lớp {len(match.wrong_class)}")

    if not results:
        raise SystemExit("Chưa có clip nào vừa có số đếm tay vừa có kết quả pipeline")
    for path, rows in ((out_dir / "results.csv", results), (out_dir / "mistakes.csv", mistakes)):
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["clip"])
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nĐã ghi {out_dir / 'results.csv'} và mistakes.csv")


if __name__ == "__main__":
    main()
