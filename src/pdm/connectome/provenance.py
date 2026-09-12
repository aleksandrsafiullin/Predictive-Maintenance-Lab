from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from pdm.io_util import atomic_write_json, read_json, sha256_file

ORIENTATION_CONVENTION = "W_res[i,j]=edge j→i"
WEIGHT_POLICY = "log1p(synapse_count)"
NODE_ID_DTYPE = "string"
SYNTHETIC_DISCLAIMER = "Synthetic test graph — not a biological connectome"
GRAPH_MODE_SYNTHETIC = "synthetic_fixture"
GRAPH_MODE_REAL = "real_connectome"
GRAPH_MODE_RANDOM_REWIRE = "random_rewire"
PROVENANCE_FILENAME = "provenance.json"

REQUIRED_FIELDS = (
    "source",
    "url",
    "local_path",
    "file_hash",
    "retrieved_at",
    "column_names_read",
    "graph_mode",
    "n_nodes",
    "n_edges",
    "seed",
    "node_id_dtype",
    "orientation",
    "weight_policy",
    "disclaimer",
)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def file_hash_if_present(path: Path | str | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    return sha256_file(p)


def build_provenance(
    *,
    source: str,
    graph_mode: str,
    n_nodes: int,
    n_edges: int,
    disclaimer: str,
    url: str | None = None,
    local_path: str | Path | None = None,
    file_hash: str | None = None,
    retrieved_at: str | None = None,
    column_names_read: list[str] | None = None,
    seed: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "source": source,
        "url": url,
        "local_path": str(local_path) if local_path is not None else None,
        "file_hash": file_hash,
        "retrieved_at": retrieved_at or utc_now(),
        "column_names_read": list(column_names_read or []),
        "graph_mode": graph_mode,
        "n_nodes": int(n_nodes),
        "n_edges": int(n_edges),
        "seed": seed,
        "node_id_dtype": NODE_ID_DTYPE,
        "orientation": ORIENTATION_CONVENTION,
        "weight_policy": WEIGHT_POLICY,
        "disclaimer": disclaimer,
    }
    if extra:
        rec.update(extra)
    return rec


def write_provenance(path: Path, record: dict[str, Any]) -> Path:
    path = Path(path)
    atomic_write_json(path, record)
    return path


def read_provenance(path: Path) -> dict[str, Any]:
    return read_json(Path(path))


def write_graph_artifact(
    directory: Path,
    payload: dict[str, Any],
    provenance: dict[str, Any],
    *,
    graph_filename: str = "graph.json",
    provenance_filename: str = PROVENANCE_FILENAME,
) -> dict[str, Path]:
    """Write a graph JSON next to a provenance manifest."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    graph_path = directory / graph_filename
    prov_path = directory / provenance_filename
    atomic_write_json(graph_path, payload)
    write_provenance(prov_path, provenance)
    return {"graph": graph_path, "provenance": prov_path}
