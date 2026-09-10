from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import requests

from pdm.io_util import atomic_write_json, sha256_file
from pdm.paths import data_raw, project_root

ProgressFn = Callable[[str, dict[str, Any]], None]

AUTHOR_BEARINGS_PAGE = "https://biaowang.tech/xjtu-sy-bearing-datasets/"
GDRIVE_FOLDER_ID = "1_ycmG46PARiykt82ShfnFfyQsaXv3_VK"
HF_XJTU_ZIP = (
    "https://huggingface.co/datasets/DavidNguyen/XJTU-SY_Bearing_Datasets/"
    "resolve/main/XJTU-SY_Bearing_Datasets.zip?download=true"
)
KAGGLE_SLUG = "prognosticshse/preventive-to-predicitve-maintenance"


def _emit(progress: ProgressFn | None, stage: str, **kwargs: Any) -> None:
    if progress:
        progress(stage, kwargs)


def download_url(url: str, dest: Path, progress: ProgressFn | None = None, timeout: int = 60) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "PredictiveMaintenanceLab/0.1"
    pos = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={pos}-"} if pos else {}
    _emit(progress, "download", url=url, dest=str(dest), resume_from=pos)
    with session.get(url, headers=headers, stream=True, timeout=timeout, allow_redirects=True) as r:
        r.raise_for_status()
        mode = "ab" if pos and r.status_code == 206 else "wb"
        if mode == "wb":
            pos = 0
        written = pos
        with dest.open(mode) as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                written += len(chunk)
                if written % (32 * 1024 * 1024) < 1024 * 1024:
                    _emit(progress, "download_progress", bytes=written, dest=str(dest))
    return dest


def _copy_tree(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        shutil.copy2(src, dest / src.name)
        return
    for item in src.iterdir():
        target = dest / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def download_filters(local_path: str | None = None, progress: ProgressFn | None = None) -> dict[str, Any]:
    dest = data_raw() / "filters"
    dest.mkdir(parents=True, exist_ok=True)
    source = {"kind": None, "path": None, "url": f"https://www.kaggle.com/datasets/{KAGGLE_SLUG}"}
    if local_path:
        src = Path(local_path).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"Local filters path not found: {src}")
        _emit(progress, "copy_local", path=str(src))
        _copy_tree(src, dest)
        source["kind"] = "local"
        source["path"] = str(src)
    else:
        _emit(progress, "kagglehub", slug=KAGGLE_SLUG)
        import kagglehub

        cached = Path(kagglehub.dataset_download(KAGGLE_SLUG))
        _copy_tree(cached, dest)
        source["kind"] = "kagglehub"
        source["path"] = str(cached)
        source["version_dir"] = cached.name
    files = _manifest_files(dest)
    rec = {
        "dataset_id": "filters",
        "obtained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": source,
        "license": "CC BY 4.0",
        "files": files,
    }
    _update_global_manifest("filters", rec)
    return rec


def download_bearings(local_path: str | None = None, progress: ProgressFn | None = None) -> dict[str, Any]:
    dest = data_raw() / "bearings"
    dest.mkdir(parents=True, exist_ok=True)
    source: dict[str, Any] = {
        "kind": None,
        "author_page": AUTHOR_BEARINGS_PAGE,
        "google_drive_folder_id": GDRIVE_FOLDER_ID,
    }
    errors: list[str] = []
    zip_path = dest / "XJTU-SY_Bearing_Datasets.zip"

    if local_path:
        src = Path(local_path).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"Local bearings path not found: {src}")
        _emit(progress, "copy_local", path=str(src))
        if src.is_file():
            shutil.copy2(src, dest / src.name)
        else:
            _copy_tree(src, dest)
        source["kind"] = "local"
        source["path"] = str(src)
    else:
        try:
            _emit(progress, "gdown", id=GDRIVE_FOLDER_ID)
            import gdown

            gdown.download_folder(
                id=GDRIVE_FOLDER_ID,
                output=str(dest / "gdrive"),
                quiet=False,
                remaining_ok=True,
            )
            source["kind"] = "google_drive"
            source["path"] = str(dest / "gdrive")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Google Drive (author link) failed: {exc}")
            _emit(progress, "gdrive_failed", error=str(exc))
            _emit(
                progress,
                "huggingface_mirror",
                note="Same XJTU-SY_Bearing_Datasets.zip, not a substitute dataset",
                url=HF_XJTU_ZIP,
            )
            download_url(HF_XJTU_ZIP, zip_path, progress=progress)
            source["kind"] = "huggingface_author_zip_mirror"
            source["url"] = HF_XJTU_ZIP
            source["note"] = (
                "Official Google Drive from biaowang.tech was unavailable; "
                "downloaded XJTU-SY_Bearing_Datasets.zip from a public copy of the author package."
            )
        source["errors"] = errors

    files = _manifest_files(dest)
    rec = {
        "dataset_id": "bearings",
        "obtained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": source,
        "use": "validation of prognostics algorithms; citation requested by author",
        "files": files,
    }
    _update_global_manifest("bearings", rec)
    return rec


def download_dataset(dataset_id: str, local_path: str | None = None, progress: ProgressFn | None = None) -> dict[str, Any]:
    if dataset_id == "filters":
        return download_filters(local_path=local_path, progress=progress)
    if dataset_id == "bearings":
        return download_bearings(local_path=local_path, progress=progress)
    raise ValueError(f"Unknown dataset_id={dataset_id}")


def _manifest_files(root: Path) -> list[dict[str, Any]]:
    out = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.name in {"download.log", "manifest.json"}:
            continue
        rec = {"relpath": str(p.relative_to(root)), "bytes": p.stat().st_size}
        # Skip hashing multi-GB files during download; hashed later in inspect if needed.
        if p.stat().st_size <= 64 * 1024 * 1024:
            rec["sha256"] = sha256_file(p)
        out.append(rec)
    return out


def _update_global_manifest(dataset_id: str, rec: dict[str, Any]) -> None:
    path = project_root() / "data" / "manifest.json"
    if path.exists():
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = {"datasets": {}}
    data.setdefault("datasets", {})[dataset_id] = rec
    atomic_write_json(path, data)


def looks_like_url(value: str) -> bool:
    try:
        return urlparse(value).scheme in {"http", "https"}
    except Exception:
        return False
