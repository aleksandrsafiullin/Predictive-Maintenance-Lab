from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import yaml


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_short(digest: str, n: int = 12) -> str:
    """Truncate a hex digest for display / directory names."""
    return digest[:n]


def checkpoint_hash(path: Path | None = None, state_dict: Mapping[str, Any] | None = None) -> str:
    """SHA-256 of `best.pt` (or any ckpt) bytes, else of state_dict keys+shapes."""
    if path is not None:
        p = Path(path)
        if p.exists() and p.is_file():
            return sha256_file(p)
    if state_dict is not None:
        payload = []
        for key in sorted(state_dict):
            val = state_dict[key]
            shape = tuple(int(x) for x in val.shape) if hasattr(val, "shape") else None
            dtype = str(getattr(val, "dtype", type(val).__name__))
            payload.append({"k": key, "shape": shape, "dtype": dtype})
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    raise FileNotFoundError("checkpoint_hash requires an existing file path or a state_dict")


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        Path(tmp).replace(path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, obj: Any, indent: int = 2) -> None:
    atomic_write_text(path, json.dumps(obj, indent=indent, default=str) + "\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def dump_yaml(path: Path, obj: Any) -> None:
    atomic_write_text(path, yaml.safe_dump(obj, sort_keys=False, allow_unicode=True))


def append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")
        f.flush()
