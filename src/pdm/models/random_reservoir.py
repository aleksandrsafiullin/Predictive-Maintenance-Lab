from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import networkx as nx

from pdm.connectome.provenance import (
    GRAPH_MODE_RANDOM_REWIRE,
    GRAPH_MODE_SYNTHETIC,
    RANDOM_REWIRE_DISCLAIMER,
    SYNTHETIC_DISCLAIMER,
    build_provenance,
    hash_graph,
)
from pdm.connectome.sampling import rewire_directed
from pdm.connectome.weights import ordered_node_ids
from pdm.models.fly_reservoir import FlyConnectomeReservoir


class RandomReservoir(FlyConnectomeReservoir):
    """Degree-preserving directed rewiring of a parent connectome graph.

    Same leaky update, readout, ``state_mode``, leak, spectral radius, input
    scale, and seed protocol as :class:`FlyConnectomeReservoir`. ``W_res`` is
    rebuilt from the rewired ``log1p`` adjacency (not copied from the parent).
    Provenance always records ``graph_mode='random_rewire'`` and
    ``parent_graph_hash``. If the parent was synthetic, both disclaimers apply.
    """

    def __init__(
        self,
        graph: nx.DiGraph,
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
        parent_provenance: dict[str, Any] | None = None,
        n_swaps_multiplier: int = 10,
        frozen_weights: tuple | None = None,
    ) -> None:
        if frozen_weights is not None:
            # Saved artifacts are already rewired; do not rebuild or re-rewire from seed.
            if graph is not None and not isinstance(graph, nx.DiGraph):
                raise TypeError("RandomReservoir frozen load expects an nx.DiGraph or None")
            super().__init__(
                graph,
                input_size,
                head=head,
                leak=leak,
                spectral_radius=spectral_radius,
                input_scale=input_scale,
                seed=seed,
                state_mode=state_mode,
                time_scale_s=time_scale_s,
                node_order=node_order,
                n_nodes=n_nodes,
                provenance=provenance,
                frozen_weights=frozen_weights,
            )
            self.graph_mode = GRAPH_MODE_RANDOM_REWIRE
            self.provenance["graph_mode"] = GRAPH_MODE_RANDOM_REWIRE
            self.parent_graph_hash = (provenance or {}).get("parent_graph_hash")
            self.provenance["parent_graph_hash"] = self.parent_graph_hash
            self.architecture = "random_reservoir"
            return
        if not isinstance(graph, nx.DiGraph):
            raise TypeError("RandomReservoir requires a parent nx.DiGraph")
        parent_hash = hash_graph(graph)
        order = ordered_node_ids(graph, node_order)
        if n_nodes is not None and int(n_nodes) != len(order):
            raise ValueError(
                f"n_nodes={int(n_nodes)} does not match provided graph size {len(order)} "
                "(artifact mismatch is not synthetic clamp)"
            )
        rewired = rewire_directed(graph, seed=int(seed), n_swaps_multiplier=int(n_swaps_multiplier))
        parent_meta = parent_provenance or provenance
        parent_is_synthetic = _parent_is_synthetic(parent_meta)
        disclaimer = _rewire_disclaimer(parent_is_synthetic)
        source = str((parent_meta or {}).get("source") or "in_memory")
        rec = build_provenance(
            source=source,
            graph_mode=GRAPH_MODE_RANDOM_REWIRE,
            n_nodes=rewired.number_of_nodes(),
            n_edges=rewired.number_of_edges(),
            disclaimer=disclaimer,
            url=(parent_meta or {}).get("url"),
            local_path=(parent_meta or {}).get("local_path"),
            file_hash=(parent_meta or {}).get("file_hash"),
            column_names_read=(parent_meta or {}).get("column_names_read"),
            seed=int(seed),
            parent_graph_hash=parent_hash,
            extra={
                "is_synthetic": parent_is_synthetic,
                "parent_graph_mode": (parent_meta or {}).get("graph_mode"),
                "graph_hash": hash_graph(rewired),
            },
        )
        super().__init__(
            rewired,
            input_size,
            head=head,
            leak=leak,
            spectral_radius=spectral_radius,
            input_scale=input_scale,
            seed=seed,
            state_mode=state_mode,
            time_scale_s=time_scale_s,
            node_order=order,
            n_nodes=n_nodes,
            provenance=rec,
        )
        self.parent_graph_hash = parent_hash
        self.graph_mode = GRAPH_MODE_RANDOM_REWIRE
        self.provenance["graph_mode"] = GRAPH_MODE_RANDOM_REWIRE
        self.provenance["parent_graph_hash"] = parent_hash
        self.is_synthetic = parent_is_synthetic
        self.architecture = "random_reservoir"


def _parent_is_synthetic(parent_meta: dict[str, Any] | None) -> bool:
    if parent_meta is None:
        return True
    if bool(parent_meta.get("is_synthetic")):
        return True
    mode = str(parent_meta.get("graph_mode") or "").strip().lower()
    return mode == GRAPH_MODE_SYNTHETIC or mode == ""


def _rewire_disclaimer(parent_is_synthetic: bool) -> str:
    if parent_is_synthetic:
        return f"{SYNTHETIC_DISCLAIMER}; {RANDOM_REWIRE_DISCLAIMER}"
    return RANDOM_REWIRE_DISCLAIMER
