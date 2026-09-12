from __future__ import annotations

import logging

import networkx as nx
import numpy as np

from pdm.connectome.graph import as_node_id
from pdm.connectome.provenance import (
    GRAPH_MODE_RANDOM_REWIRE,
    GRAPH_MODE_REAL,
    GRAPH_MODE_SYNTHETIC,
)

LOGGER = logging.getLogger(__name__)

REAL_CONNECTOME_N_MIN = 500
REAL_CONNECTOME_N_MAX = 2000


def resolve_n_nodes(graph_mode: str, requested: int, available_n: int) -> int:
    """Clamp synthetic requests; raise for ``real_connectome`` out of range.

    ``synthetic_fixture``: ``n_nodes = min(requested, available_n)`` (logged, never raised).
    ``real_connectome``: require ``500 <= requested <= 2000`` and ``requested <= available_n``.
    ``random_rewire`` uses the parent graph's already-resolved ``n_nodes`` (subtask 02b).
    """
    mode = str(graph_mode or "").strip().lower()
    req = int(requested)
    avail = int(available_n)
    if mode == GRAPH_MODE_SYNTHETIC:
        n = min(req, avail)
        if n != req:
            LOGGER.info(
                "synthetic_fixture n_nodes clamped from %s to %s (fixture has %s nodes); "
                "not promoting to real_connectome",
                req,
                n,
                avail,
            )
        return n
    if mode == GRAPH_MODE_REAL:
        if not (REAL_CONNECTOME_N_MIN <= req <= REAL_CONNECTOME_N_MAX):
            raise ValueError(
                f"real_connectome n_nodes must be in "
                f"[{REAL_CONNECTOME_N_MIN}, {REAL_CONNECTOME_N_MAX}], got {req}"
            )
        if req > avail:
            raise ValueError(
                f"real_connectome n_nodes={req} exceeds available graph size {avail}"
            )
        return req
    if mode == GRAPH_MODE_RANDOM_REWIRE:
        # Parent graph size is already resolved; do not clamp or re-validate independently.
        return avail
    raise ValueError(f"Unknown graph_mode={graph_mode!r}")


def sample_connected_subgraph(graph: nx.DiGraph, n_nodes: int, seed: int) -> list[str]:
    """Seeded BFS connected subgraph. Same ``(graph, n_nodes, seed)`` → same node ids."""
    nodes = sorted(as_node_id(n) for n in graph.nodes())
    n = min(int(n_nodes), len(nodes))
    if n <= 0:
        return []
    rng = np.random.default_rng(int(seed))
    start = nodes[int(rng.integers(0, len(nodes)))]
    undirected = graph.to_undirected()
    seen: list[str] = []
    queued = {start}
    queue = [start]
    while queue and len(seen) < n:
        cur = queue.pop(0)
        if cur in seen:
            continue
        seen.append(cur)
        nbrs = [as_node_id(nb) for nb in undirected.neighbors(cur) if as_node_id(nb) not in queued]
        rng.shuffle(nbrs)
        queue.extend(nbrs)
        queued.update(nbrs)
    if len(seen) < n:
        extra = [nid for nid in nodes if nid not in set(seen)]
        seen.extend(extra[: n - len(seen)])
    return seen[:n]


def induced_subgraph(graph: nx.DiGraph, node_ids: list[str]) -> nx.DiGraph:
    keep = [as_node_id(n) for n in node_ids]
    return graph.subgraph(keep).copy()


def rewire_directed(graph: nx.DiGraph, seed: int, n_swaps_multiplier: int = 10) -> nx.DiGraph:
    """Maslov–Sneppen degree-preserving directed two-edge swaps.

    Pick edges ``a→b`` and ``c→d`` and replace them with ``a→d`` and ``c→b``
    when the swap is valid. Self-loops and parallel edges are forbidden.
    In-degree and out-degree sequences are preserved. The returned graph has
    the same node set and the same number of directed edges. Edge attributes
    (``synapse_count`` / ``weight``) travel with the source stub.

    Deterministic given ``seed``. Does not mutate ``graph``.
    """
    out = graph.copy()
    n_edges = out.number_of_edges()
    if n_edges < 2:
        return out
    rng = np.random.default_rng(int(seed))
    edge_list = list(out.edges())
    edge_set = set(edge_list)
    n_target = max(int(n_swaps_multiplier) * n_edges, n_edges)
    max_tries = max(n_target * 20, 100)
    swaps = 0
    tries = 0
    while swaps < n_target and tries < max_tries:
        tries += 1
        i, j = rng.choice(n_edges, size=2, replace=False)
        a, b = edge_list[int(i)]
        c, d = edge_list[int(j)]
        if a == c or b == d:
            continue
        if a == d or c == b:
            continue
        new_ab = (a, d)
        new_cd = (c, b)
        if new_ab in edge_set or new_cd in edge_set:
            continue
        data_ab = dict(out.edges[a, b])
        data_cd = dict(out.edges[c, d])
        out.remove_edge(a, b)
        out.remove_edge(c, d)
        out.add_edge(a, d, **data_ab)
        out.add_edge(c, b, **data_cd)
        edge_set.remove((a, b))
        edge_set.remove((c, d))
        edge_set.add(new_ab)
        edge_set.add(new_cd)
        edge_list[int(i)] = new_ab
        edge_list[int(j)] = new_cd
        swaps += 1
    return out
