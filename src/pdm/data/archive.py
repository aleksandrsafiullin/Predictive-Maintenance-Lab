from __future__ import annotations

import zipfile
from pathlib import Path


class UnsafeArchiveError(RuntimeError):
    pass


def _is_within(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def validate_zip_member(name: str, dest: Path | None = None) -> str:
    norm = name.replace("\\", "/")
    if norm.startswith("/") or norm.startswith("\\") or ".." in Path(norm).parts:
        raise UnsafeArchiveError(f"Refusing archive member with traversal: {name}")
    if dest is not None:
        target = (dest / norm).resolve()
        if not _is_within(dest.resolve(), target):
            raise UnsafeArchiveError(f"Refusing archive member outside dest: {name}")
    return norm


def iter_zip_csv_names(zip_path: Path) -> list[str]:
    names: list[str] = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = validate_zip_member(info.filename)
            if name.lower().endswith(".csv"):
                names.append(name)
    return names
