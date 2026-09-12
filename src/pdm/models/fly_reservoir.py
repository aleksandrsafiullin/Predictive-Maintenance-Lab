from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import networkx as nx
import numpy as np
import torch

from pdm.connectome.graph import as_node_id
from pdm.connectome.provenance import (
    GRAPH_MODE_SYNTHETIC,
    SYNTHETIC_DISCLAIMER,
    build_provenance,
    hash_graph,
)
from pdm.connectome.weights import (
    ordered_node_ids,
    random_input_weights,
    scale_to_spectral_radius,
    w_res_from_graph,
)
from pdm.models.reservoir import LeakyESN


class FlyConnectomeReservoir(LeakyESN):
    """Connectome-structured leaky ESN. Frozen W_in / W_res / b_res; trainable readout.

    Accepts an ``nx.DiGraph`` or a square adjacency matrix already in j→i orientation.
    ``run_training`` still raises until subtask 02c; this class is for unit tests and
    direct construction.
    """

    def __init__(
        self,
        graph: nx.DiGraph | np.ndarray,
        input_size: int,
        *,
        head: str = "rul",
        leak: float = 0.2,
        spectral_radius: float = 0.9,
        input_scale: float = 0.1,
        seed: int = 42,
        state_mode: str = "window_reset",
        time_scale_s: float = 1.0,
        node_order: Sequence[str] | None = None,
        n_nodes: int | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> None:
        w_in, w_res, b_res, meta = build_fly_reservoir_weights(
            graph,
            input_size=input_size,
            spectral_radius=spectral_radius,
            input_scale=input_scale,
            seed=seed,
            node_order=node_order,
            n_nodes=n_nodes,
        )
        super().__init__(
            torch.tensor(w_in, dtype=torch.float32),
            torch.tensor(w_res, dtype=torch.float32),
            torch.tensor(b_res, dtype=torch.float32),
            alpha=float(leak),
            state_mode=state_mode,
            head=head,
            time_scale_s=time_scale_s,
        )
        self.seed = int(seed)
        self.spectral_radius = float(spectral_radius)
        self.input_scale = float(input_scale)
        self.node_order = list(meta["node_order"])
        self.graph = meta["graph"]
        self.graph_hash = meta["graph_hash"]
        self.graph_mode = str((provenance or {}).get("graph_mode") or GRAPH_MODE_SYNTHETIC)
        self.parent_graph_hash = (provenance or {}).get("parent_graph_hash")
        self.is_synthetic = bool((provenance or {}).get("is_synthetic", True))
        if provenance is not None:
            self.provenance = dict(provenance)
        else:
            n_edges = int(self.graph.number_of_edges()) if self.graph is not None else 0
            self.provenance = build_provenance(
                source="in_memory",
                graph_mode=GRAPH_MODE_SYNTHETIC,
                n_nodes=self.n_nodes,
                n_edges=n_edges,
                disclaimer=SYNTHETIC_DISCLAIMER,
                seed=int(seed),
                parent_graph_hash=None,
                extra={"is_synthetic": True, "graph_hash": self.graph_hash},
            )
            self.graph_mode = GRAPH_MODE_SYNTHETIC
            self.is_synthetic = True
        if "graph_hash" not in self.provenance:
            self.provenance["graph_hash"] = self.graph_hash
        if "parent_graph_hash" not in self.provenance:
            self.provenance["parent_graph_hash"] = self.parent_graph_hash


def build_fly_reservoir_weights(
    graph: nx.DiGraph | np.ndarray,
    *,
    input_size: int,
    spectral_radius: float = 0.9,
    input_scale: float = 0.1,
    seed: int = 42,
    node_order: Sequence[str] | None = None,
    n_nodes: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Build frozen ``W_in``, ``W_res``, ``b_res``. ``W_res`` is never transposed."""
    nx_graph: nx.DiGraph | None
    if isinstance(graph, nx.DiGraph):
        nx_graph = graph
        order = ordered_node_ids(graph, node_order)
        n_graph = len(order)
        if n_nodes is not None and int(n_nodes) != n_graph:
            raise ValueError(
                f"n_nodes={int(n_nodes)} does not match provided graph size {n_graph} "
                "(artifact mismatch is not synthetic clamp)"
            )
        w_res = w_res_from_graph(graph, spectral_radius=spectral_radius, node_order=order)
        graph_id = hash_graph(graph)
    else:
        nx_graph = None
        adj = np.asarray(graph, dtype=np.float64)
        if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
            raise ValueError(f"adjacency matrix must be square, got shape {adj.shape}")
        n_graph = int(adj.shape[0])
        if n_nodes is not None and int(n_nodes) != n_graph:
            raise ValueError(
                f"n_nodes={int(n_nodes)} does not match provided adjacency size {n_graph} "
                "(artifact mismatch is not synthetic clamp)"
            )
        order = [as_node_id(n) for n in (node_order if node_order is not None else range(n_graph))]
        if len(order) != n_graph:
            raise ValueError("node_order length must match adjacency size")
        w_res = scale_to_spectral_radius(adj, spectral_radius)
        graph_id = None
    w_in = random_input_weights(n_graph, int(input_size), input_scale, seed)
    b_res = np.zeros(n_graph, dtype=np.float64)
    meta = {
        "node_order": order,
        "graph": nx_graph,
        "graph_hash": graph_id,
    }
    return w_in, w_res, b_res, meta
