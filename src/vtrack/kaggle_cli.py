"""Gọi Kaggle CLI (cài trong cùng .venv) — dùng chung cho các script đưa dataset/kernel lên Kaggle.
Cần file ~/.kaggle/kaggle.json."""

import json
import os
import subprocess
import sys
from pathlib import Path

# Dùng kaggle CLI cài trong cùng môi trường Python đang chạy script (.venv)
KAGGLE_CLI = str(Path(sys.executable).with_name("kaggle.exe" if os.name == "nt" else "kaggle"))


def kaggle_username() -> str:
    credentials = Path.home() / ".kaggle" / "kaggle.json"
    return json.loads(credentials.read_text(encoding="utf-8"))["username"]


def run_kaggle(arguments: list[str]) -> None:
    command = [KAGGLE_CLI, *arguments]
    print(" ".join(command))
    # Kaggle CLI đọc file bằng bảng mã mặc định của Windows (cp1252) -> lỗi với tiếng Việt;
    # PYTHONUTF8=1 buộc Python trong tiến trình CLI dùng UTF-8.
    subprocess.run(command, check=True, env={**os.environ, "PYTHONUTF8": "1"})


def dataset_exists(dataset_ref: str) -> bool:
    result = subprocess.run([KAGGLE_CLI, "datasets", "status", dataset_ref], capture_output=True, text=True)
    return result.returncode == 0 and "ready" in result.stdout.lower()


def push_dataset(directory: Path, dataset_ref: str, message: str) -> None:
    """Lần đầu tạo dataset mới (mặc định private), các lần sau tạo phiên bản mới kèm ghi chú.
    directory phải có sẵn dataset-metadata.json với đúng dataset_ref."""
    # --dir-mode zip: mỗi thư mục con được nén khi upload, Kaggle tự giải nén
    if dataset_exists(dataset_ref):
        run_kaggle(["datasets", "version", "-p", str(directory), "-m", message, "--dir-mode", "zip"])
    else:
        run_kaggle(["datasets", "create", "-p", str(directory), "--dir-mode", "zip"])
    print(f"\nXong: https://www.kaggle.com/datasets/{dataset_ref}")
