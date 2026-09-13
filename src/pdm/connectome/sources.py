from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

from pdm.connectome.graph import graph_from_payload
from pdm.connectome.provenance import (
    GRAPH_MODE_REAL,
    GRAPH_MODE_SYNTHETIC,
    SYNTHETIC_DISCLAIMER,
    build_provenance,
    file_hash_if_present,
)
from pdm.paths import data_raw

LOGGER = logging.getLogger(__name__)

# Documented remote locations only — this loader never downloads them.
MALEMCNS_GCS_URI = (
    "gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/"
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
)
MALEMCNS_HTTPS_URL = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
)
# neuPrint is an alternate human-facing import step, not a test dependency.
NEUPRINT_NOTE = "neuPrint (https://neuprint.janelia.org/) is an alternate human step, not used in tests."
MALEMCNS_FILENAME = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
SYNTHETIC_FIXTURE_NAME = "synthetic_connectome.json"
FIXTURE_PACKAGE = "pdm.connectome.fixtures"

# Documented FlyEM-style names. Do not invent columns for a missing file.
# v1.0 feather uses body_pre/body_post; older exports use pre/post.
SRC_COLUMNS = ("pre", "body_pre")
DST_COLUMNS = ("post", "body_post")
WEIGHT_COLUMNS = ("synapse_count", "weight")


@dataclass
class ConnectomeGraph:
    graph: nx.DiGraph
    provenance: dict[str, Any]
    is_synthetic: bool
    label: str
    payload: dict[str, Any] = field(default_factory=dict)


def default_malemcns_path() -> Path:
    return data_raw() / "connectome" / MALEMCNS_FILENAME


def load_synthetic_fixture() -> ConnectomeGraph:
    text = files(FIXTURE_PACKAGE).joinpath(SYNTHETIC_FIXTURE_NAME).read_text(encoding="utf-8")
    payload = json.loads(text)
    graph = graph_from_payload(payload)
    label = str(payload.get("label") or SYNTHETIC_DISCLAIMER)
    provenance = build_provenance(
        source="synthetic_fixture",
        graph_mode=GRAPH_MODE_SYNTHETIC,
        n_nodes=graph.number_of_nodes(),
        n_edges=graph.number_of_edges(),
        disclaimer=label,
        url=None,
        local_path=f"package:{FIXTURE_PACKAGE}/{SYNTHETIC_FIXTURE_NAME}",
        file_hash=None,
        column_names_read=["nodes", "edges", "is_synthetic", "label"],
        seed=None,
        extra={"is_synthetic": True, "label": label},
    )
    return ConnectomeGraph(
        graph=graph,
        provenance=provenance,
        is_synthetic=True,
        label=label,
        payload=payload,
    )


def _pick_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _map_malemcns_columns(frame: pd.DataFrame) -> tuple[str, str, str]:
    cols = [str(c) for c in frame.columns]
    src = _pick_column(cols, SRC_COLUMNS)
    dst = _pick_column(cols, DST_COLUMNS)
    weight = _pick_column(cols, WEIGHT_COLUMNS)
    if src is None or dst is None or weight is None:
        raise ValueError(
            "MaleCNS table columns do not match documented FlyEM names "
            "(pre/body_pre, post/body_post, synapse_count/weight). "
            f"Columns found: {cols}"
        )
    return src, dst, weight


def load_malemcns(path: str | Path | None = None) -> ConnectomeGraph:
    """Load a local MaleCNS feather file, or fall back to the synthetic fixture.

    Missing files do not invent rows. Provenance ``source=unavailable`` and
    ``graph_mode=synthetic_fixture`` — never silently labeled ``real_connectome``.
    """
    loc = Path(path).expanduser() if path is not None else default_malemcns_path()
    if loc.is_dir():
        loc = loc / MALEMCNS_FILENAME
    if not loc.is_file():
        LOGGER.info("MaleCNS file missing at %s; using synthetic fixture (source=unavailable)", loc)
        syn = load_synthetic_fixture()
        syn.provenance = build_provenance(
            source="unavailable",
            graph_mode=GRAPH_MODE_SYNTHETIC,
            n_nodes=syn.graph.number_of_nodes(),
            n_edges=syn.graph.number_of_edges(),
            disclaimer=syn.label,
            url=MALEMCNS_HTTPS_URL,
            local_path=loc,
            file_hash=None,
            column_names_read=[],
            seed=None,
            extra={
                "is_synthetic": True,
                "label": syn.label,
                "documented_gcs_uri": MALEMCNS_GCS_URI,
                "neuprint_note": NEUPRINT_NOTE,
            },
        )
        return syn

    frame = pd.read_feather(loc)
    src_col, dst_col, weight_col = _map_malemcns_columns(frame)
    edges = [
        {"src": str(row[src_col]), "dst": str(row[dst_col]), "weight": float(row[weight_col])}
        for row in frame[[src_col, dst_col, weight_col]].to_dict(orient="records")
    ]
    payload = {"nodes": sorted({e["src"] for e in edges} | {e["dst"] for e in edges}), "edges": edges}
    graph = graph_from_payload(payload)
    provenance = build_provenance(
        source="local",
        graph_mode=GRAPH_MODE_REAL,
        n_nodes=graph.number_of_nodes(),
        n_edges=graph.number_of_edges(),
        disclaimer="FlyEM Male CNS connectome weights loaded from a local path.",
        url=MALEMCNS_HTTPS_URL,
        local_path=loc,
        file_hash=file_hash_if_present(loc),
        column_names_read=[src_col, dst_col, weight_col],
        seed=None,
        extra={"is_synthetic": False, "documented_gcs_uri": MALEMCNS_GCS_URI},
    )
    return ConnectomeGraph(
        graph=graph,
        provenance=provenance,
        is_synthetic=False,
        label="",
        payload=payload,
    )
