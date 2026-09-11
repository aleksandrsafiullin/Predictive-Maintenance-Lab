from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from pdm.data.archive import iter_zip_csv_names, validate_zip_member
from pdm.features import spectral_band_energy, time_domain_features
from pdm.paths import dataset_raw

ProgressFn = Callable[[str, dict[str, Any]], None]

BEARING_RE = re.compile(r"Bearing(\d+)_(\d+)", re.IGNORECASE)
CSV_NUM_RE = re.compile(r"^(\d+)\.csv$", re.IGNORECASE)
REGIME_MAP = {
    "35hz12kn": {
        "regime_id": "35Hz12kN",
        "rpm": 2100.0,
        "load_kn": 12.0,
        "freq_hz": 35.0,
    },
    "37.5hz11kn": {
        "regime_id": "37.5Hz11kN",
        "rpm": 2250.0,
        "load_kn": 11.0,
        "freq_hz": 37.5,
    },
    "40hz10kn": {
        "regime_id": "40Hz10kN",
        "rpm": 2400.0,
        "load_kn": 10.0,
        "freq_hz": 40.0,
    },
}


def numeric_csv_index(name: str) -> int | None:
    stem = Path(name).name
    m = CSV_NUM_RE.match(stem)
    return int(m.group(1)) if m else None


def sort_csv_names_numerically(names: Iterable[str]) -> list[str]:
    indexed = []
    for n in names:
        idx = numeric_csv_index(n)
        if idx is None:
            continue
        indexed.append((idx, n))
    indexed.sort(key=lambda x: x[0])
    return [n for _, n in indexed]


def _norm_token(s: str) -> str:
    return s.replace(" ", "").replace("_", "").lower()


def infer_regime(parts: list[str]) -> dict[str, Any] | None:
    for part in parts:
        key = _norm_token(part)
        if key in REGIME_MAP:
            return dict(REGIME_MAP[key])
    joined = _norm_token("/".join(parts))
    for key, meta in REGIME_MAP.items():
        if key in joined:
            return dict(meta)
    return None


def parse_bearing_path(relpath: str) -> dict[str, Any] | None:
    parts = Path(relpath.replace("\\", "/")).parts
    bearing = None
    regime_part = None
    csv_idx = numeric_csv_index(parts[-1]) if parts else None
    if csv_idx is None:
        return None
    for part in parts:
        m = BEARING_RE.search(part)
        if m:
            bearing = {
                "folder": part,
                "regime_index": int(m.group(1)),
                "instance": int(m.group(2)),
                "unit_id": part if part.lower().startswith("bearing") else f"Bearing{m.group(1)}_{m.group(2)}",
            }
        key = _norm_token(part)
        if key in REGIME_MAP:
            regime_part = part
    if bearing is None:
        return None
    regime = infer_regime(list(parts))
    if regime is None and bearing["regime_index"] in {1, 2, 3}:
        # Fallback only after seeing BearingN_M; still require a known folder if possible.
        order = ["35hz12kn", "37.5hz11kn", "40hz10kn"]
        regime = dict(REGIME_MAP[order[bearing["regime_index"] - 1]])
        regime["regime_inferred_from_bearing_index"] = True
    return {
        "relpath": relpath.replace("\\", "/"),
        "csv_index": csv_idx,
        "regime_folder": regime_part,
        "regime": regime,
        **bearing,
    }


def locate_bearings_source(raw_dir: Path | None = None) -> dict[str, Any]:
    raw_dir = raw_dir or dataset_raw("bearings")
    zips = sorted(raw_dir.rglob("*.zip"))
    # Prefer a folder that already has Bearing*/n.csv
    csvs = [p for p in raw_dir.rglob("*.csv") if numeric_csv_index(p.name)]
    if csvs:
        return {"kind": "folder", "path": str(raw_dir), "n_csv": len(csvs)}
    if zips:
        # pick largest zip
        zips.sort(key=lambda p: p.stat().st_size, reverse=True)
        return {"kind": "zip", "path": str(zips[0]), "bytes": zips[0].stat().st_size}
    raise FileNotFoundError(
        f"No XJTU-SY CSV files or zip found under {raw_dir}. "
        "Pass a local path to the author archive or extracted folder."
    )


def inspect_bearings(raw_dir: Path | None = None, max_files: int = 3) -> dict[str, Any]:
    src = locate_bearings_source(raw_dir)
    samples = []
    issues = []
    if src["kind"] == "zip":
        zip_path = Path(src["path"])
        names = sort_csv_names_numerically(iter_zip_csv_names(zip_path))
        parsed = [parse_bearing_path(n) for n in names]
        parsed = [p for p in parsed if p]
        units = sorted({p["unit_id"] for p in parsed})
        with zipfile.ZipFile(zip_path) as zf:
            for name in names[:max_files]:
                samples.append(_inspect_csv_bytes(zf.read(name), name, issues))
        src["n_csv"] = len(names)
        src["n_units"] = len(units)
        src["unit_ids"] = units
        src["sample_headers"] = samples
    else:
        csvs = [p for p in Path(src["path"]).rglob("*.csv") if numeric_csv_index(p.name)]
        csvs = sorted(csvs, key=lambda p: (str(p.parent), numeric_csv_index(p.name) or 0))
        for p in csvs[:max_files]:
            samples.append(_inspect_csv_bytes(p.read_bytes(), str(p), issues))
        parsed = [parse_bearing_path(str(p.relative_to(src["path"]))) for p in csvs]
        parsed = [p for p in parsed if p]
        src["n_csv"] = len(csvs)
        src["n_units"] = len({p["unit_id"] for p in parsed})
        src["unit_ids"] = sorted({p["unit_id"] for p in parsed})
        src["sample_headers"] = samples
    src["issues"] = issues
    return src


def _inspect_csv_bytes(data: bytes, name: str, issues: list[str]) -> dict[str, Any]:
    text = data[:2000]
    head = text.split(b"\n", 1)[0].decode("utf-8", "replace")
    try:
        arr, quality = _read_vibration_csv(data, expected_samples=32768)
    except Exception as exc:  # noqa: BLE001
        issues.append(f"{name}: failed to parse CSV ({exc})")
        return {"name": name, "header_preview": head, "parse_ok": False}
    rec = {
        "name": name,
        "header_preview": head,
        "parse_ok": True,
        "shape": [int(arr.shape[0]), int(arr.shape[1])],
        "header_fields": quality.get("header_fields"),
        "first_row": arr[0].tolist() if arr.size else [],
    }
    if arr.shape[1] < 2:
        issues.append(f"{name}: expected 2 channels, found {arr.shape[1]} columns. Fields={rec}")
    return rec


def _read_vibration_csv(data: bytes, expected_samples: int) -> tuple[np.ndarray, dict[str, Any]]:
    bio = io.BytesIO(data)
    first = bio.readline().decode("utf-8", "replace").strip()
    header_fields = [h.strip() for h in first.split(",")]
    skip = 0
    try:
        [float(x) for x in header_fields[:2]]
        header_fields = ["col0", "col1"]
        bio.seek(0)
    except ValueError:
        skip = 1
        bio.seek(0)
    try:
        body = data.split(b"\n", 1)[1] if skip else data
        body = body.replace(b"\r", b"")
        flat = np.fromstring(body.replace(b",", b" "), dtype=np.float64, sep=" ")
        if flat.size % 2 != 0:
            raise ValueError(f"odd number of values: {flat.size}")
        arr = flat.reshape(-1, 2)
    except Exception:
        bio.seek(0)
        arr = np.loadtxt(bio, delimiter=",", skiprows=skip, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
    quality: dict[str, Any] = {
        "n_samples": int(arr.shape[0]),
        "n_channels": int(arr.shape[1]),
        "n_nan": int(np.isnan(arr).sum()),
        "expected_samples": expected_samples,
        "sample_count_ok": int(arr.shape[0]) == expected_samples,
        "channels_ok": int(arr.shape[1]) >= 2,
        "header_fields": header_fields,
    }
    if arr.shape[1] < 2:
        raise ValueError(
            f"Expected at least 2 vibration channels, found shape={arr.shape}. "
            f"Actual fields={header_fields}"
        )
    return arr[:, :2], quality


def extract_bearings_features(
    cfg: dict[str, Any],
    raw_dir: Path | None = None,
    progress: ProgressFn | None = None,
) -> pd.DataFrame:
    src = locate_bearings_source(raw_dir)
    expected = int(cfg.get("samples_per_fragment", 32768))
    fs = float(cfg.get("sampling_rate_hz", 25600))
    interval_s = float(cfg.get("fragment_interval_s", 60))
    band_edges = list(cfg.get("features", {}).get("spectral_band_edges_hz") or [0, 3200, 6400, 9600, 12800])
    rows: list[dict[str, Any]] = []

    def handle_one(relpath: str, data: bytes) -> None:
        meta = parse_bearing_path(relpath)
        if meta is None or meta.get("regime") is None:
            return
        arr, quality = _read_vibration_csv(data, expected)
        rec: dict[str, Any] = {
            "dataset_id": "bearings",
            "unit_id": meta["unit_id"],
            "regime_id": meta["regime"]["regime_id"],
            "regime_index": meta["regime_index"],
            "instance": meta["instance"],
            "file_index": meta["csv_index"],
            "timestamp_s": float(meta["csv_index"]) * interval_s,
            "rpm": meta["regime"]["rpm"],
            "load_kn": meta["regime"]["load_kn"],
            "operating_age_s": float(meta["csv_index"]) * interval_s,
            "n_samples": quality["n_samples"],
            "sample_count_ok": quality["sample_count_ok"],
            "n_nan": quality["n_nan"],
            "relpath": relpath,
        }
        for ch_i, ch_name in enumerate(("horizontal", "vertical")):
            td = time_domain_features(arr[:, ch_i], prefix=ch_name)
            rec.update(td)
            bands = spectral_band_energy(arr[:, ch_i], fs=fs, edges_hz=band_edges, prefix=ch_name)
            rec.update(bands)
        rows.append(rec)
        if progress and len(rows) % 50 == 0:
            progress("features", {"n_fragments": len(rows), "unit_id": rec["unit_id"]})

    if src["kind"] == "zip":
        zip_path = Path(src["path"])
        names = sort_csv_names_numerically(iter_zip_csv_names(zip_path))
        with zipfile.ZipFile(zip_path) as zf:
            for name in names:
                validate_zip_member(name)
                handle_one(name, zf.read(name))
    else:
        root = Path(src["path"])
        csvs = [p for p in root.rglob("*.csv") if numeric_csv_index(p.name)]
        csvs = sorted(
            csvs,
            key=lambda p: (
                str(p.parent),
                numeric_csv_index(p.name) or 0,
            ),
        )
        for p in csvs:
            handle_one(str(p.relative_to(root)), p.read_bytes())

    if not rows:
        raise RuntimeError("No bearing CSV fragments were parsed. Check archive layout.")
    df = pd.DataFrame(rows).sort_values(["unit_id", "file_index"]).reset_index(drop=True)
    df = _mark_bearing_gaps(df)
    return df


def _mark_bearing_gaps(df: pd.DataFrame) -> pd.DataFrame:
    gap_before = np.zeros(len(df), dtype=bool)
    for unit, idx in df.groupby("unit_id").groups.items():
        sub = df.loc[list(idx)].sort_values("file_index")
        prev = None
        for i, row in sub.iterrows():
            if prev is not None and int(row["file_index"]) != prev + 1:
                gap_before[df.index.get_loc(i)] = True
            prev = int(row["file_index"])
    df = df.copy()
    df["gap_before"] = gap_before
    return df


def build_bearing_units(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    recs = []
    for unit_id, g in df.groupby("unit_id"):
        g = g.sort_values("file_index")
        last = g.iloc[-1]
        files = set(g["file_index"].astype(int))
        expected = set(range(int(g["file_index"].min()), int(g["file_index"].max()) + 1))
        missing = sorted(expected - files)
        recs.append(
            {
                "dataset_id": "bearings",
                "unit_id": unit_id,
                "origin_unit_id": unit_id,
                "regime_id": last["regime_id"],
                "instance": int(last["instance"]),
                "n_measurements": int(len(g)),
                "file_min": int(g["file_index"].min()),
                "file_max": int(g["file_index"].max()),
                "missing_file_indices": missing,
                "n_gaps": int(g["gap_before"].sum()),
                "observation_end_s": float(last["timestamp_s"]),
                "event_time_s": float(last["timestamp_s"]),
                "event_observed": 1,
                "endpoint_definition": cfg.get("endpoint_definition", "last_recorded_sample"),
                "event_source": "last_valid_fragment_approximation",
            }
        )
    return pd.DataFrame(recs)
