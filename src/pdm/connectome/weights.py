from __future__ import annotations

from collections.abc import Sequence

import networkx as nx
import numpy as np

from pdm.connectome.graph import as_node_id

# W_res[i, j] = weight of directed edge j → i. Never transpose to "fix" a plot.
ORIENTATION = "W_res[i,j]=edge j→i"


def ordered_node_ids(graph: nx.DiGraph, node_order: Sequence[str] | None = None) -> list[str]:
    if node_order is not None:
        return [as_node_id(n) for n in node_order]
    return sorted(as_node_id(n) for n in graph.nodes())


def log1p_adjacency(
    graph: nx.DiGraph,
    node_order: Sequence[str] | None = None,
) -> np.ndarray:
    """Directed log1p synapse-count matrix in j→i orientation.

    For an edge ``src → dst`` (j→i with j=src, i=dst), ``A[dst, src] != 0``
    and ``A[src, dst] == 0`` unless a reverse edge also exists.
    """
    nodes = ordered_node_ids(graph, node_order)
    index = {nid: i for i, nid in enumerate(nodes)}
    n = len(nodes)
    adj = np.zeros((n, n), dtype=np.float64)
    for src, dst, data in graph.edges(data=True):
        j = index[as_node_id(src)]
        i = index[as_node_id(dst)]
        syn = float(data.get("synapse_count", data.get("weight", 1.0)))
        # A[i, j] = log1p(synapse_count of directed edge j→i); do not transpose.
        adj[i, j] = np.log1p(syn)
    return adj


def scale_spectral_radius(matrix: np.ndarray, spectral_radius: float = 0.9) -> np.ndarray:
    """Stub: return a copy. Actual spectral-radius scaling is implemented in subtask 02b."""
    del spectral_radius
    return np.array(matrix, dtype=np.float64, copy=True)
