"""Đưa dataset đã gộp (data/processed/vn_vehicles) lên Kaggle dưới dạng dataset private.
Lần đầu tạo mới; các lần sau tạo phiên bản mới kèm ghi chú. Cần file ~/.kaggle/kaggle.json.

Ví dụ:
    python scripts/kaggle_push_dataset.py
    python scripts/kaggle_push_dataset.py --message "Thêm tập in-domain Hà Nội"
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

DATASET_DIR = Path("data/processed/vn_vehicles")
METADATA_TEMPLATE = Path("configs/kaggle/dataset-metadata.json")

# Dùng kaggle CLI cài trong cùng môi trường Python đang chạy script (.venv)
KAGGLE_CLI = str(Path(sys.executable).with_name("kaggle.exe" if os.name == "nt" else "kaggle"))


def kaggle_username() -> str:
    credentials = Path.home() / ".kaggle" / "kaggle.json"
    return json.loads(credentials.read_text(encoding="utf-8"))["username"]


def dataset_exists(dataset_ref: str) -> bool:
    result = subprocess.run([KAGGLE_CLI, "datasets", "status", dataset_ref], capture_output=True, text=True)
    return result.returncode == 0 and "ready" in result.stdout.lower()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--message", default="Cập nhật dataset", help="Ghi chú cho phiên bản mới")
    args = parser.parse_args()

    if not (DATASET_DIR / "data.yaml").exists():
        sys.exit(f"Chưa có {DATASET_DIR}/data.yaml — chạy scripts/prepare_data.py trước.")

    # prepare_data.py xoá và tạo lại DATASET_DIR, nên metadata được ghi lại mỗi lần push
    metadata_text = METADATA_TEMPLATE.read_text(encoding="utf-8").replace("{username}", kaggle_username())
    (DATASET_DIR / "dataset-metadata.json").write_text(metadata_text, encoding="utf-8")
    dataset_ref = json.loads(metadata_text)["id"]

    # --dir-mode zip: mỗi thư mục con (train/val/test) được nén khi upload, Kaggle tự giải nén
    if dataset_exists(dataset_ref):
        command = [KAGGLE_CLI, "datasets", "version", "-p", str(DATASET_DIR), "-m", args.message, "--dir-mode", "zip"]
    else:
        command = [KAGGLE_CLI, "datasets", "create", "-p", str(DATASET_DIR), "--dir-mode", "zip"]
    print(" ".join(command))
    subprocess.run(command, check=True)
    print(f"\nXong: https://www.kaggle.com/datasets/{dataset_ref}")


if __name__ == "__main__":
    main()
