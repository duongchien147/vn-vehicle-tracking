"""Xuất nhãn job CVAT của clip dev ở định dạng "MOT 1.1" (không kèm ảnh) qua API, giải nén vào <out>/gt/.
Chạy: python cvat_export_mot.py data/tracking_gt/pvd_goc_c_dev"""
import io
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cvat_api import HOST, JOB_ID, session  # noqa: E402

FORMAT = "MOT 1.1"


def main(out_dir: str) -> None:
    s = session()
    r = s.post(f"{HOST}/api/jobs/{JOB_ID}/dataset/export", params={"format": FORMAT, "save_images": False}, timeout=60)
    r.raise_for_status()
    rq_id = r.json()["rq_id"]
    for _ in range(120):
        status = s.get(f"{HOST}/api/requests/{rq_id}", timeout=30).json()
        if status["status"] in ("finished", "failed"):
            break
        time.sleep(2)
    if status["status"] != "finished":
        sys.exit(f"xuất thất bại: {status}")
    data = s.get(status["result_url"], timeout=120).content
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        print(archive.namelist())
        archive.extractall(out_dir)


if __name__ == "__main__":
    main(sys.argv[1])
