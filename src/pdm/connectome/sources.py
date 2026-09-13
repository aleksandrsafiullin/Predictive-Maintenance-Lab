from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from pdm.connectome.graph import graph_from_edges, graph_from_payload
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
# MaleCNS v1.0 feather uses body_pre / body_post / weight; keep pre / post / synapse_count
# so older tables and the synthetic fixture JSON path still map.
SRC_COLUMNS = ("body_pre", "pre")
DST_COLUMNS = ("body_post", "post")
WEIGHT_COLUMNS = ("weight", "synapse_count")
MALEMCNS_MIN_CONF_NOTE = (
    "FlyEM MaleCNS v1.0 feather is pre-filtered at minimum confidence 0.5 by the source dataset. "
    "No additional weight threshold applied by default."
)


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
            f"(body_pre/pre, body_post/post, weight/synapse_count). Columns found: {cols}"
        )
    return src, dst, weight


def _bfs_subgraph_from_df(
    df: pd.DataFrame,
    src_col: str,
    dst_col: str,
    n_nodes: int,
    seed: int,
) -> tuple[list[str], pd.DataFrame]:
    """Seeded BFS on a DataFrame's edge list. Never builds a full NetworkX graph.

    Returns (node_ids_str, subgraph_df) where subgraph_df contains only the
    edges whose both endpoints are in the sampled node set.
    Same (df, n_nodes, seed) → same result.
    """
    rng = np.random.default_rng(int(seed))

    # Sorted numpy arrays for O(log n) neighbor lookup (no full adjacency dict)
    src_vals = df[src_col].to_numpy()
    dst_vals = df[dst_col].to_numpy()
    if src_vals.size == 0:
        return [], df.iloc[0:0].copy()

    fwd_idx = np.argsort(src_vals, kind="stable")
    src_sorted = src_vals[fwd_idx]
    dst_fwd = dst_vals[fwd_idx]

    bwd_idx = np.argsort(dst_vals, kind="stable")
    dst_sorted = dst_vals[bwd_idx]
    src_bwd = src_vals[bwd_idx]

    all_nodes = np.unique(np.concatenate([src_vals, dst_vals]))
    n = min(int(n_nodes), len(all_nodes))
    if n <= 0:
        return [], df.iloc[0:0].copy()

    start = all_nodes[int(rng.integers(0, len(all_nodes)))]
    if isinstance(start, np.generic):
        start = start.item()

    seen_list: list = []
    seen_set: set = set()
    queued: set = {start}
    queue: list = [start]

    while queue and len(seen_list) < n:
        cur = queue.pop(0)
        if cur in seen_set:
            continue
        seen_list.append(cur)
        seen_set.add(cur)

        lo = int(np.searchsorted(src_sorted, cur, side="left"))
        hi = int(np.searchsorted(src_sorted, cur, side="right"))
        out_nbrs = dst_fwd[lo:hi]

        lo2 = int(np.searchsorted(dst_sorted, cur, side="left"))
        hi2 = int(np.searchsorted(dst_sorted, cur, side="right"))
        in_nbrs = src_bwd[lo2:hi2]

        nbrs = np.concatenate([out_nbrs, in_nbrs])
        new_nbrs = [nb for nb in nbrs.tolist() if nb not in queued]
        rng.shuffle(new_nbrs)
        queue.extend(new_nbrs)
        queued.update(new_nbrs)

    if len(seen_list) < n:
        extra = [nd for nd in all_nodes.tolist() if nd not in seen_set]
        seen_list.extend(extra[: n - len(seen_list)])

    node_ids = seen_list[:n]
    node_set = set(node_ids)

    mask = df[src_col].isin(node_set) & df[dst_col].isin(node_set)
    sub_df = df[mask].reset_index(drop=True)

    return [str(nd) for nd in node_ids], sub_df


def load_malemcns_subgraph(
    path: str | Path | None,
    n_nodes: int,
    seed: int,
    weight_threshold: int | None = None,
) -> ConnectomeGraph:
    """Load a deterministic connected subgraph from the MaleCNS feather.

    Reads the feather as a DataFrame but never builds a NetworkX graph of the full
    ~152 M-edge connectome. BFS sampling is done on sorted numpy arrays (O(log n)
    per node expansion). NetworkX is built only for the sampled subgraph (≤ 2000
    nodes).

    ``weight_threshold`` (optional int): keep only edges with weight >= this value.
    The file is already pre-filtered at minconf-0.5 by FlyEM; set threshold only if
    you have a documented biological reason (default None = no additional filter).

    Missing file → synthetic fixture with source=unavailable; never silently labeled
    real_connectome.
    """
    loc = Path(path).expanduser() if path is not None else default_malemcns_path()
    if loc.is_dir():
        loc = loc / MALEMCNS_FILENAME
    if not loc.is_file():
        LOGGER.info(
            "MaleCNS file missing at %s; using synthetic fixture (source=unavailable)", loc
        )
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
            seed=int(seed),
            extra={
                "is_synthetic": True,
                "label": syn.label,
                "documented_gcs_uri": MALEMCNS_GCS_URI,
                "neuprint_note": NEUPRINT_NOTE,
            },
        )
        return syn

    LOGGER.info("Loading MaleCNS subgraph: n_nodes=%d seed=%d from %s", n_nodes, seed, loc)
    fhash = file_hash_if_present(loc)

    frame = pd.read_feather(loc)
    src_col, dst_col, weight_col = _map_malemcns_columns(frame)

    if weight_threshold is not None:
        before = len(frame)
        frame = frame[frame[weight_col] >= weight_threshold].reset_index(drop=True)
        LOGGER.info(
            "weight_threshold=%d reduced edges from %d to %d",
            weight_threshold,
            before,
            len(frame),
        )

    working = frame[[src_col, dst_col]].copy()
    node_ids, sub_df = _bfs_subgraph_from_df(working, src_col, dst_col, int(n_nodes), int(seed))

    # Add weight column back for subgraph edges.
    # Use native-typed endpoints from sub_df (not str node_ids) so int64 feather
    # columns match; never iterate the full ~152 M-row frame.
    if not sub_df.empty:
        native_set = set(sub_df[src_col]).union(sub_df[dst_col])
        sub_df = frame.loc[
            frame[src_col].isin(native_set) & frame[dst_col].isin(native_set),
            [src_col, dst_col, weight_col],
        ].reset_index(drop=True)

    edges = [
        {"src": str(row[src_col]), "dst": str(row[dst_col]), "weight": float(row[weight_col])}
        for _, row in sub_df.iterrows()
    ]
    graph = graph_from_edges(edges, node_ids)

    threshold_note = (
        f"weight_threshold={weight_threshold} applied"
        if weight_threshold is not None
        else MALEMCNS_MIN_CONF_NOTE
    )

    provenance = build_provenance(
        source="local",
        graph_mode=GRAPH_MODE_REAL,
        n_nodes=graph.number_of_nodes(),
        n_edges=graph.number_of_edges(),
        disclaimer="FlyEM MaleCNS connectome subgraph sampled via seeded BFS; full graph not materialized.",
        url=MALEMCNS_HTTPS_URL,
        local_path=loc,
        file_hash=fhash,
        column_names_read=[src_col, dst_col, weight_col],
        seed=int(seed),
        extra={
            "is_synthetic": False,
            "documented_gcs_uri": MALEMCNS_GCS_URI,
            "full_graph_materialized": False,
            "sampling_method": "seeded_bfs",
            "weight_threshold": weight_threshold,
            "weight_threshold_note": threshold_note,
        },
    )
    return ConnectomeGraph(
        graph=graph,
        provenance=provenance,
        is_synthetic=False,
        label="",
        payload={},
    )


def load_malemcns(path: str | Path | None = None) -> ConnectomeGraph:
    """Load a local MaleCNS feather file, or fall back to the synthetic fixture.

    Missing files do not invent rows. Provenance ``source=unavailable`` and
    ``graph_mode=synthetic_fixture`` — never silently labeled ``real_connectome``.

    .. warning::
        This function materialises the full edge list as Python dicts and builds a
        NetworkX graph of ALL rows (up to 152 M edges). Use ``load_malemcns_subgraph``
        for the real_connectome training path. ``load_malemcns`` is kept for fallback
        / synthetic fixture tests only.
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
