from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import pandas as pd

from pdm.connectome.graph import as_node_id
from pdm.connectome.provenance import SYNTHETIC_DISCLAIMER, build_provenance, file_hash_if_present
from pdm.connectome.sources import MALEMCNS_FILENAME, NEUPRINT_NOTE
from pdm.paths import data_raw

LOGGER = logging.getLogger(__name__)

# Documented human download locations only — this loader never fetches them.
SOMA_GCS_PREFIX = "gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/"
SOMA_DOWNLOAD_HUB = "https://male-cns.janelia.org/download/"

# Ordered exact filenames. No glob. First existing remaining candidate is used.
SOMA_ALLOWLIST: tuple[str, ...] = (
    "body-annotations-male-cns-v1.0-minconf-0.5.feather",
)

GRAPH_MODE_VIZ_SOMA = "viz_soma_xyz"
GRAPH_MODE_UNAVAILABLE = "unavailable"
SOMA_ROLE = "viz_soma_xyz"
SYNTHETIC_SOMA_FIXTURE_NAME = "synthetic_somas.json"
FIXTURE_PACKAGE = "pdm.connectome.fixtures"

ID_COLUMNS = ("bodyId", "body_id", "bodyid", "body")
SOMA_LOCATION_COLUMNS = ("somaLocation",)
PRE_XYZ_COLUMNS = ("x_pre", "y_pre", "z_pre")
WEIGHTS_COLUMNS = ("body_pre", "body_post", "weight")


@dataclass
class SomaTable:
    positions: dict[str, list[float]]
    provenance: dict[str, Any]
    is_synthetic: bool
    n_points: int


def default_soma_dir() -> Path:
    return data_raw() / "connectome"


def _is_forbidden_soma_name(name: str) -> bool:
    return name == MALEMCNS_FILENAME or name.startswith("syn-points-")


def _forbidden_name_error(name: str) -> ValueError:
    return ValueError(
        f"{name} is not a soma anatomy table (MaleCNS weights or syn-points). "
        "This loader skips those filenames by name and does not parse them."
    )


def _pick_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _schema_error(cols: list[str]) -> ValueError:
    return ValueError(
        "Soma table columns do not match documented FlyEM names "
        "(bodyId/body_id/bodyid/body + somaLocation or x,y,z or soma_x,soma_y,soma_z). "
        f"Columns found: {cols}"
    )


def _is_non_soma_schema(columns: list[str]) -> bool:
    lower = {c.lower() for c in columns}
    if set(PRE_XYZ_COLUMNS) <= lower:
        return True
    if set(WEIGHTS_COLUMNS) <= lower:
        return True
    return False


def find_soma_table_path(directory: Path | str | None = None) -> Path | None:
    """First allowlisted soma file under ``directory`` (skip weights / syn-points by name)."""
    root = Path(directory) if directory is not None else default_soma_dir()
    for name in SOMA_ALLOWLIST:
        if _is_forbidden_soma_name(name):
            continue
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def _viz_extra(
    *,
    n_points: int,
    n_dropped: int,
    is_synthetic: bool,
) -> dict[str, Any]:
    return {
        "n_points": int(n_points),
        "n_dropped": int(n_dropped),
        "license": "CC-BY",
        "role": SOMA_ROLE,
        "n_model_unaffected": True,
        "is_synthetic": bool(is_synthetic),
        "documented_gcs_prefix": SOMA_GCS_PREFIX,
        "download_hub": SOMA_DOWNLOAD_HUB,
        "neuprint_note": NEUPRINT_NOTE,
    }


def _empty_unavailable(local_path: Path | str | None) -> SomaTable:
    provenance = build_provenance(
        source="unavailable",
        graph_mode=GRAPH_MODE_UNAVAILABLE,
        n_nodes=0,
        n_edges=0,
        disclaimer="Soma anatomy table not found under the filename allowlist.",
        url=SOMA_DOWNLOAD_HUB,
        local_path=local_path,
        file_hash=None,
        column_names_read=[],
        seed=None,
        extra=_viz_extra(n_points=0, n_dropped=0, is_synthetic=False),
    )
    return SomaTable(positions={}, provenance=provenance, is_synthetic=False, n_points=0)


def _finite_xyz(x: Any, y: Any, z: Any) -> list[float] | None:
    try:
        fx, fy, fz = float(x), float(y), float(z)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(fx) and math.isfinite(fy) and math.isfinite(fz)):
        return None
    return [fx, fy, fz]


def _parse_soma_location(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    elif not isinstance(value, (dict, list, tuple)) and hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        if all(k in value for k in ("x", "y", "z")):
            return _finite_xyz(value["x"], value["y"], value["z"])
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return _finite_xyz(value[0], value[1], value[2])
    return None


def _map_soma_columns(frame: pd.DataFrame) -> tuple[str, str, tuple[str, ...]]:
    cols = [str(c) for c in frame.columns]
    if _is_non_soma_schema(cols):
        raise _schema_error(cols)
    id_col = _pick_column(cols, ID_COLUMNS)
    if id_col is None:
        raise _schema_error(cols)
    loc_col = _pick_column(cols, SOMA_LOCATION_COLUMNS)
    if loc_col is not None:
        return id_col, "somaLocation", (loc_col,)
    x_col = _pick_column(cols, ("x",))
    y_col = _pick_column(cols, ("y",))
    z_col = _pick_column(cols, ("z",))
    if x_col is not None and y_col is not None and z_col is not None:
        return id_col, "xyz", (x_col, y_col, z_col)
    sx = _pick_column(cols, ("soma_x",))
    sy = _pick_column(cols, ("soma_y",))
    sz = _pick_column(cols, ("soma_z",))
    if sx is not None and sy is not None and sz is not None:
        return id_col, "soma_xyz", (sx, sy, sz)
    raise _schema_error(cols)


def _read_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".feather", ".ft"}:
        return pd.read_feather(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(
        f"Unsupported soma table suffix {suffix!r} for {path}. "
        "Use .feather, .parquet, or .json."
    )


def _positions_from_frame(frame: pd.DataFrame) -> tuple[dict[str, list[float]], list[str], int]:
    id_col, kind, used = _map_soma_columns(frame)
    columns_read = [id_col, *used]
    positions: dict[str, list[float]] = {}
    n_dropped = 0
    id_series = frame[id_col]

    def _keep_id(raw_id: Any) -> str | None:
        if raw_id is None or (isinstance(raw_id, float) and math.isnan(raw_id)):
            return None
        nid = as_node_id(raw_id)
        if nid == "" or nid in positions:
            return None
        return nid

    if kind == "somaLocation":
        loc_series = frame[used[0]]
        for raw_id, raw_loc in zip(id_series.tolist(), loc_series.tolist(), strict=True):
            nid = _keep_id(raw_id)
            if nid is None:
                n_dropped += 1
                continue
            xyz = _parse_soma_location(raw_loc)
            if xyz is None:
                n_dropped += 1
                continue
            positions[nid] = xyz
        return positions, columns_read, n_dropped

    xs = pd.to_numeric(frame[used[0]], errors="coerce")
    ys = pd.to_numeric(frame[used[1]], errors="coerce")
    zs = pd.to_numeric(frame[used[2]], errors="coerce")
    for raw_id, x, y, z in zip(id_series.tolist(), xs.tolist(), ys.tolist(), zs.tolist(), strict=True):
        nid = _keep_id(raw_id)
        if nid is None:
            n_dropped += 1
            continue
        xyz = _finite_xyz(x, y, z)
        if xyz is None:
            n_dropped += 1
            continue
        positions[nid] = xyz
    return positions, columns_read, n_dropped


def _table_from_path(path: Path) -> SomaTable:
    LOGGER.info("Soma table columns will be read from %s", path)
    frame = _read_frame(path)
    cols = [str(c) for c in frame.columns]
    LOGGER.info("Soma table columns: %s", cols)
    positions, columns_read, n_dropped = _positions_from_frame(frame)
    n_points = len(positions)
    provenance = build_provenance(
        source="local",
        graph_mode=GRAPH_MODE_VIZ_SOMA,
        n_nodes=0,
        n_edges=0,
        disclaimer="FlyEM MaleCNS soma xyz (viz-only; does not set ESN n_nodes).",
        url=SOMA_DOWNLOAD_HUB,
        local_path=path,
        file_hash=file_hash_if_present(path),
        column_names_read=columns_read,
        seed=None,
        extra=_viz_extra(n_points=n_points, n_dropped=n_dropped, is_synthetic=False),
    )
    return SomaTable(
        positions=positions,
        provenance=provenance,
        is_synthetic=False,
        n_points=n_points,
    )


def load_soma_table(path: str | Path | None = None) -> SomaTable:
    """Load viz-only soma xyz keyed by string body id.

    Default search walks ``SOMA_ALLOWLIST`` under ``default_soma_dir()``. Missing
    allowlist hits return an empty table (``graph_mode=unavailable``) and do not
    raise. Explicit weights / syn-points filenames raise ``ValueError``.
    Never writes ``runs/.../connectome/provenance.json``.
    """
    if path is None:
        loc = find_soma_table_path()
        if loc is None:
            return _empty_unavailable(default_soma_dir())
        return _table_from_path(loc)

    loc = Path(path).expanduser()
    if loc.is_dir():
        found = find_soma_table_path(loc)
        if found is None:
            raise FileNotFoundError(f"No allowlisted soma anatomy file under {loc}")
        return _table_from_path(found)

    if _is_forbidden_soma_name(loc.name):
        raise _forbidden_name_error(loc.name)
    if not loc.is_file():
        raise FileNotFoundError(f"Soma anatomy file not found: {loc}")
    return _table_from_path(loc)


def load_synthetic_somas() -> SomaTable:
    """Tiny labeled fixture. Never used as the missing-file fallback."""
    resource = files(FIXTURE_PACKAGE).joinpath(SYNTHETIC_SOMA_FIXTURE_NAME)
    payload = json.loads(resource.read_text(encoding="utf-8"))
    label = str(payload.get("label") or payload.get("disclaimer") or SYNTHETIC_DISCLAIMER)
    if SYNTHETIC_DISCLAIMER not in label:
        label = f"{SYNTHETIC_DISCLAIMER}"
    raw_positions = payload.get("positions") or {}
    positions: dict[str, list[float]] = {}
    n_dropped = 0
    for raw_id, raw_xyz in raw_positions.items():
        nid = as_node_id(raw_id)
        if not isinstance(raw_xyz, (list, tuple)) or len(raw_xyz) < 3:
            n_dropped += 1
            continue
        xyz = _finite_xyz(raw_xyz[0], raw_xyz[1], raw_xyz[2])
        if xyz is None or nid in positions:
            n_dropped += 1
            continue
        positions[nid] = xyz
        if len(positions) >= 64:
            break
    n_points = len(positions)
    provenance = build_provenance(
        source="synthetic_fixture",
        graph_mode=GRAPH_MODE_VIZ_SOMA,
        n_nodes=0,
        n_edges=0,
        disclaimer=label,
        url=None,
        local_path=f"package:{FIXTURE_PACKAGE}/{SYNTHETIC_SOMA_FIXTURE_NAME}",
        file_hash=None,
        column_names_read=["positions"],
        seed=None,
        extra=_viz_extra(n_points=n_points, n_dropped=n_dropped, is_synthetic=True),
    )
    return SomaTable(
        positions=positions,
        provenance=provenance,
        is_synthetic=True,
        n_points=n_points,
    )
