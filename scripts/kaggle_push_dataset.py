"""Đưa dataset đã gộp (data/processed/vn_vehicles) lên Kaggle dưới dạng dataset private.
Lần đầu tạo mới; các lần sau tạo phiên bản mới kèm ghi chú. Cần file ~/.kaggle/kaggle.json.

Ví dụ:
    python scripts/kaggle_push_dataset.py
    python scripts/kaggle_push_dataset.py --message "Thêm tập in-domain Hà Nội"
"""

import argparse
import json
import sys
from pathlib import Path

from vtrack.kaggle_cli import kaggle_username, push_dataset

DATASET_DIR = Path("data/processed/vn_vehicles")
METADATA_TEMPLATE = Path("configs/kaggle/dataset-metadata.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--message", default="Cập nhật dataset", help="Ghi chú cho phiên bản mới")
    args = parser.parse_args()

    if not (DATASET_DIR / "data.yaml").exists():
        sys.exit(f"Chưa có {DATASET_DIR}/data.yaml — chạy scripts/prepare_data.py trước.")

    # prepare_data.py xoá và tạo lại DATASET_DIR, nên metadata được ghi lại mỗi lần push
    metadata_text = METADATA_TEMPLATE.read_text(encoding="utf-8").replace("{username}", kaggle_username())
    (DATASET_DIR / "dataset-metadata.json").write_text(metadata_text, encoding="utf-8")
    push_dataset(DATASET_DIR, json.loads(metadata_text)["id"], args.message)


if __name__ == "__main__":
    main()
