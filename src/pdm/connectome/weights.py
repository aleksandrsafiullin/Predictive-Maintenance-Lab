from __future__ import annotations

import io
from collections.abc import Sequence
from pathlib import Path

import networkx as nx
import numpy as np

from pdm.connectome.graph import as_node_id
from pdm.io_util import atomic_write_bytes

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


_RHO_EPS = 1e-12


def scale_to_spectral_radius(A: np.ndarray, target_sr: float = 0.9) -> np.ndarray:
    """Scale A so its largest absolute eigenvalue equals ``target_sr``.

    Empty or all-zero matrices raise. If the spectral radius is ~0 but A is not
    identically zero (nilpotent / DAG), scale by the largest singular value so a
    one-edge graph keeps a finite, orientation-preserving ``W_res``. Never
    transposes A.
    """
    mat = np.array(A, dtype=np.float64, copy=True)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"adjacency must be square, got shape {mat.shape}")
    if mat.size == 0 or not np.any(np.abs(mat) > 0):
        raise ValueError("Cannot scale an empty or all-zero adjacency matrix to a spectral radius")
    target = float(target_sr)
    if not np.isfinite(target) or target <= 0.0:
        raise ValueError(f"target spectral radius must be positive and finite, got {target_sr}")
    rho = float(np.max(np.abs(np.linalg.eigvals(mat))))
    if not np.isfinite(rho) or rho < _RHO_EPS:
        svals = np.linalg.svd(mat, compute_uv=False)
        rho = float(svals[0]) if svals.size else 0.0
    if not np.isfinite(rho) or rho < _RHO_EPS:
        raise ValueError("Cannot scale an empty or all-zero adjacency matrix to a spectral radius")
    return mat * (target / rho)


def scale_spectral_radius(matrix: np.ndarray, spectral_radius: float = 0.9) -> np.ndarray:
    """Alias for :func:`scale_to_spectral_radius`."""
    return scale_to_spectral_radius(matrix, spectral_radius)


def w_res_from_graph(
    graph: nx.DiGraph,
    spectral_radius: float = 0.9,
    node_order: Sequence[str] | None = None,
) -> np.ndarray:
    """``log1p`` adjacency in j→i orientation, scaled to ``spectral_radius``. Never transposes."""
    adj = log1p_adjacency(graph, node_order)
    return scale_to_spectral_radius(adj, spectral_radius)


def random_input_weights(
    n_nodes: int,
    input_size: int,
    input_scale: float,
    seed: int,
) -> np.ndarray:
    """Dense ``W_in`` of shape ``[n_nodes, input_size]`` uniform in ``[-input_scale, input_scale]``."""
    rng = np.random.default_rng(int(seed))
    return rng.uniform(
        -float(input_scale),
        float(input_scale),
        size=(int(n_nodes), int(input_size)),
    ).astype(np.float64)


def save_reservoir_weights(
    path: Path,
    W_in: np.ndarray,
    W_res: np.ndarray,
    b_res: np.ndarray,
) -> Path:
    """Write frozen ``W_in``, ``W_res``, ``b_res``. Never transpose ``W_res``."""
    path = Path(path)
    buf = io.BytesIO()
    np.savez(
        buf,
        W_in=np.asarray(W_in),
        W_res=np.asarray(W_res),
        b_res=np.asarray(b_res),
    )
    atomic_write_bytes(path, buf.getvalue())
    return path


def load_reservoir_weights(path: Path) -> dict[str, np.ndarray]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        missing = [k for k in ("W_in", "W_res", "b_res") if k not in data.files]
        if missing:
            raise ValueError(f"{path} missing arrays: {missing}")
        return {
            "W_in": np.array(data["W_in"], copy=True),
            "W_res": np.array(data["W_res"], copy=True),
            "b_res": np.array(data["b_res"], copy=True),
        }
