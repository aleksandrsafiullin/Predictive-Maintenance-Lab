from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any


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


_MAT_META_KEYS = {"__header__", "__version__", "__globals__", "__function_workspace__"}
# scipy.io wrappers around MATLAB objects — not experiment-table columns.
SCIPY_MAT_ENVELOPE_FIELDS = frozenset({"s0", "s1", "s2", "arr"})
_OPAQUE_MATLAB_TYPES = frozenset({"MatlabOpaque", "MatlabObject"})


def non_envelope_mat_fieldnames(names: list[str] | None) -> list[str] | None:
    """Drop scipy envelope names; leftover names may be real struct fields."""
    if not names:
        return None
    kept = [str(n) for n in names if str(n) not in SCIPY_MAT_ENVELOPE_FIELDS]
    return kept or None


def mat_fieldnames_are_envelope(names: list[str] | None) -> bool:
    raw = [str(n) for n in (names or [])]
    return bool(raw) and non_envelope_mat_fieldnames(raw) is None


def probe_mat(path: Path) -> dict[str, Any]:
    """Attempt scipy.io.loadmat. Does not decode MATLAB MCOS/table objects into columns."""
    rec: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "load_ok": False,
        "error": None,
        "matlab_header": None,
        "variables": [],
    }
    if not rec["exists"]:
        rec["error"] = "file not found"
        return rec
    try:
        from scipy.io import loadmat

        blob = loadmat(path, squeeze_me=False, struct_as_record=False)
    except Exception as exc:  # noqa: BLE001 — inspect must not crash on opaque/corrupt MAT
        rec["error"] = f"{type(exc).__name__}: {exc}"
        return rec
    rec["load_ok"] = True
    header = blob.get("__header__")
    if isinstance(header, (bytes, bytearray)):
        rec["matlab_header"] = header.decode("latin-1", "replace")
    elif header is not None:
        rec["matlab_header"] = str(header)
    for name, value in blob.items():
        if name in _MAT_META_KEYS or str(name).startswith("__"):
            continue
        rec["variables"].append(_describe_mat_variable(name, value))
    return rec


def _describe_mat_variable(name: str, value: Any) -> dict[str, Any]:
    type_name = type(value).__name__
    opaque, type_system, matlab_class = _opacity_info(value)
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    raw_names: list[str] | None = None
    # Only scipy-visible struct/record names — never invent MCOS table columns.
    names = getattr(dtype, "names", None) if dtype is not None else None
    if names:
        raw_names = [str(n) for n in names]
    elif hasattr(value, "_fieldnames"):
        raw_names = [str(n) for n in list(value._fieldnames)]
    envelope = mat_fieldnames_are_envelope(raw_names)
    fieldnames = None if opaque or envelope else non_envelope_mat_fieldnames(raw_names)
    return {
        "name": str(name),
        "python_type": type_name,
        "shape": [int(x) for x in shape] if shape is not None else None,
        "dtype": str(dtype) if dtype is not None else None,
        "opaque": opaque,
        "scipy_envelope": envelope,
        "matlab_type_system": type_system,
        "matlab_class": matlab_class,
        "fieldnames": fieldnames,
    }


def _opacity_info(value: Any) -> tuple[bool, str | None, str | None]:
    for item in _walk_matlab_values(value):
        if type(item).__name__ not in _OPAQUE_MATLAB_TYPES:
            continue
        type_system, matlab_class = _opaque_matlab_class(item)
        return True, type_system, matlab_class
    return False, None, None


def _walk_matlab_values(value: Any, *, depth: int = 0):
    yield value
    if depth >= 4:
        return
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return
    names = getattr(dtype, "names", None)
    if names:
        try:
            flat = value.reshape(-1)
        except Exception:  # noqa: BLE001
            return
        for item in list(flat)[:32]:
            for n in names:
                try:
                    yield from _walk_matlab_values(item[n], depth=depth + 1)
                except Exception:  # noqa: BLE001
                    continue
            try:
                for i in range(len(item)):
                    yield from _walk_matlab_values(item[i], depth=depth + 1)
            except Exception:  # noqa: BLE001
                continue
        return
    if getattr(dtype, "kind", None) != "O":
        return
    try:
        flat = value.reshape(-1)
    except Exception:  # noqa: BLE001
        return
    for item in list(flat)[:32]:
        yield from _walk_matlab_values(item, depth=depth + 1)


def _opaque_matlab_class(value: Any) -> tuple[str | None, str | None]:
    try:
        row = value.reshape(-1)[0]
        type_system = str(row[0]) if len(row) else None
        matlab_class = str(row[1]) if len(row) > 1 else None
        return type_system, matlab_class
    except Exception:  # noqa: BLE001
        return None, None
