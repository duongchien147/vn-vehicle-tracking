"""Gọi API CVAT cho job gán nhãn clip dev (không in token)."""
import json
import sys
from pathlib import Path

import requests

HOST = "https://app.cvat.ai"
JOB_ID = 4590297
TOKEN = Path(r"E:\cvat_config\cvat_token.txt").read_text(encoding="utf-8").strip()


def session() -> requests.Session:
    for scheme in ("Bearer", "Token"):
        s = requests.Session()
        s.headers["Authorization"] = f"{scheme} {TOKEN}"
        r = s.get(f"{HOST}/api/users/self", timeout=30)
        if r.ok:
            return s
        print(scheme, r.status_code)
    sys.exit("không xác thực được")


if __name__ == "__main__":
    s = session()
    me = s.get(f"{HOST}/api/users/self", timeout=30).json()
    print("user:", me.get("username"))
    job = s.get(f"{HOST}/api/jobs/{JOB_ID}", timeout=30).json()
    print({k: job.get(k) for k in ("id", "task_id", "start_frame", "stop_frame", "updated_date", "state", "stage")})
    ann = s.get(f"{HOST}/api/jobs/{JOB_ID}/annotations", timeout=60).json()
    out = Path(sys.argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ann, ensure_ascii=False), encoding="utf-8")
    print("tracks", len(ann["tracks"]), "shapes", len(ann["shapes"]), "version", ann.get("version"))
    labels = s.get(f"{HOST}/api/labels", params={"job_id": JOB_ID}, timeout=30).json()
    print({l["id"]: l["name"] for l in labels["results"]})
