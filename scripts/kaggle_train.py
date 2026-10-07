"""Chạy kernel train (mặc định notebooks/kaggle_train = YOLO26) trên GPU của Kaggle qua Kaggle CLI.
Thay cho extension Kaggle Studio, vốn gọi CLI với dấu nháy kiểu Linux nên lỗi trên Windows.

Ví dụ:
    python scripts/kaggle_train.py push       # đẩy notebook lên và bắt đầu chạy
    python scripts/kaggle_train.py status     # xem trạng thái: queued / running / complete / error
    python scripts/kaggle_train.py download   # tải output (best.pt, test_metrics.csv, ...) về outputs/kaggle/
    python scripts/kaggle_train.py push --notebook-dir notebooks/kaggle_train_rtdetr   # kernel RT-DETRv2
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_NOTEBOOK_DIR = Path("notebooks/kaggle_train")
DOWNLOAD_DIR = Path("outputs/kaggle")

# Dùng kaggle CLI cài trong cùng môi trường Python đang chạy script (.venv)
KAGGLE_CLI = str(Path(sys.executable).with_name("kaggle.exe" if os.name == "nt" else "kaggle"))


def kernel_ref(notebook_dir: Path) -> str:
    metadata = json.loads((notebook_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    return metadata["id"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["push", "status", "download"])
    parser.add_argument("--notebook-dir", type=Path, default=DEFAULT_NOTEBOOK_DIR,
                        help="Thư mục chứa kernel-metadata.json + code của kernel")
    args = parser.parse_args()

    ref = kernel_ref(args.notebook_dir)
    if args.action == "push":
        command = [KAGGLE_CLI, "kernels", "push", "-p", str(args.notebook_dir)]
    elif args.action == "status":
        command = [KAGGLE_CLI, "kernels", "status", ref]
    else:
        target = DOWNLOAD_DIR / ref.split("/")[1]
        target.mkdir(parents=True, exist_ok=True)
        command = [KAGGLE_CLI, "kernels", "output", ref, "-p", str(target)]
    print(" ".join(command))
    # Kaggle CLI đọc notebook bằng bảng mã mặc định của Windows (cp1252) -> lỗi với tiếng Việt;
    # PYTHONUTF8=1 buộc Python trong tiến trình CLI dùng UTF-8.
    subprocess.run(command, check=True, env={**os.environ, "PYTHONUTF8": "1"})
    if args.action == "push":
        print(f"\nTheo dõi: https://www.kaggle.com/code/{ref}")


if __name__ == "__main__":
    main()
