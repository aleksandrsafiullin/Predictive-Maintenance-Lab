from __future__ import annotations

import json
import logging
from collections import deque
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

    # Retry whole weak components; never pad a BFS with disconnected bodies.
    # The first seed draw remains identical to the legacy sampler.
    start_index = int(rng.integers(0, len(all_nodes)))
    candidates = np.concatenate([all_nodes[start_index:], all_nodes[:start_index]])
    exhausted: set = set()
    largest: list = []
    seen_list: list = []
    for start in candidates:
        if start in exhausted:
            continue
        queued = {start}
        queue = deque([start])
        seen_list = []
        while queue and len(seen_list) < n:
            cur = queue.popleft()
            seen_list.append(cur)
            lo = int(np.searchsorted(src_sorted, cur, side="left"))
            hi = int(np.searchsorted(src_sorted, cur, side="right"))
            lo2 = int(np.searchsorted(dst_sorted, cur, side="left"))
            hi2 = int(np.searchsorted(dst_sorted, cur, side="right"))
            nbrs = np.unique(np.concatenate([dst_fwd[lo:hi], src_bwd[lo2:hi2]]))
            new_nbrs = [nb for nb in nbrs.tolist() if nb not in queued]
            rng.shuffle(new_nbrs)
            queue.extend(new_nbrs)
            queued.update(new_nbrs)
        if len(seen_list) == n:
            break
        exhausted.update(seen_list)
        if len(seen_list) > len(largest):
            largest = seen_list
    if len(seen_list) < n:
        seen_list = largest

    node_ids = seen_list[:n]
    node_set = set(node_ids)

    mask = df[src_col].isin(node_set) & df[dst_col].isin(node_set)
    sub_df = df[mask].reset_index(drop=True)

    return [str(nd) for nd in node_ids], sub_df


def _read_soma_induced_weights(path: Path, eligible_ids: list[str]) -> pd.DataFrame:
    """Filter Feather v2 record batches before materializing a pandas edge table.

    The compressed weights file expands to several GB. Reading the full table
    and only then filtering caused paging on 16 GB machines. Retain just the
    soma-induced graph, with native integer body IDs and exact source weights.
    """
    import pyarrow as pa
    import pyarrow.ipc as ipc

    with pa.memory_map(str(path), "r") as source:
        reader = ipc.open_file(source)
        src, dst, weight = _map_malemcns_columns(pd.DataFrame(columns=reader.schema.names))
        cols = [src, dst, weight]
        schema = pa.schema([reader.schema.field(c) for c in cols])
        eligible = {}
        for col in (src, dst):
            dtype = schema.field(col).type
            values = [int(n) for n in eligible_ids] if pa.types.is_integer(dtype) else eligible_ids
            # Reuse the Index's hash engine across record batches. pc.is_in
            # rebuilds the 140k-body lookup on every call/batch.
            eligible[col] = pd.Index(pa.array(values, type=dtype).to_numpy())
        batches = []
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i).select(cols)
            mask = (
                eligible[src].get_indexer(batch.column(src).to_numpy(zero_copy_only=False)) >= 0
            ) & (
                eligible[dst].get_indexer(batch.column(dst).to_numpy(zero_copy_only=False)) >= 0
            )
            selected = batch.filter(pa.array(mask))
            if selected.num_rows:
                batches.append(selected)
        table = pa.Table.from_batches(batches, schema=schema)
        return table.to_pandas()


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

    from pdm.connectome.anatomy import find_soma_table_path, load_soma_table

    soma_path = find_soma_table_path(loc.parent)
    soma = load_soma_table(soma_path) if soma_path is not None else None
    frame = (
        _read_soma_induced_weights(loc, list(soma.positions))
        if soma is not None else pd.read_feather(loc)
    )
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
    else:
        sub_df = frame[[src_col, dst_col, weight_col]].iloc[:0]

    edges = [
        {"src": str(src), "dst": str(dst), "weight": float(weight)}
        for src, dst, weight in sub_df[[src_col, dst_col, weight_col]].itertuples(index=False, name=None)
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
            "sampling_method": "seeded_bfs_soma_xyz" if soma is not None else "seeded_bfs",
            "soma_file_hash": soma.provenance.get("file_hash") if soma is not None else None,
            "soma_local_path": str(soma_path) if soma_path is not None else None,
            "n_with_soma_available": len(soma.positions) if soma is not None else None,
            "anatomy_sampling": "finite_soma_xyz" if soma is not None else "anatomy_missing",
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
