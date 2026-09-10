#!/usr/bin/env python3
"""Resume XJTU-SY zip download. Same author dataset package, HF mirror of the zip."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

URL = (
    "https://huggingface.co/datasets/DavidNguyen/XJTU-SY_Bearing_Datasets/"
    "resolve/main/XJTU-SY_Bearing_Datasets.zip?download=true"
)


def main() -> int:
    dest = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/bearings/XJTU-SY_Bearing_Datasets.zip")
    dest.parent.mkdir(parents=True, exist_ok=True)
    pos = dest.stat().st_size if dest.exists() else 0
    print(f"resume_from={pos}", flush=True)
    session = requests.Session()
    session.headers["User-Agent"] = "PredictiveMaintenanceLab/0.1"
    headers = {"Range": f"bytes={pos}-"} if pos else {}
    with session.get(URL, headers=headers, stream=True, timeout=120, allow_redirects=True) as r:
        r.raise_for_status()
        print("status", r.status_code, "content_range", r.headers.get("Content-Range"), flush=True)
        mode = "ab" if pos and r.status_code == 206 else "wb"
        if mode == "wb":
            pos = 0
        written = pos
        t0 = time.time()
        with dest.open(mode) as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                written += len(chunk)
                if written % (64 * 1024 * 1024) < 1024 * 1024:
                    dt = max(time.time() - t0, 1e-6)
                    print(f"bytes={written} speed_MBps={(written - pos) / dt / 1e6:.1f}", flush=True)
    print("done", dest.stat().st_size, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
