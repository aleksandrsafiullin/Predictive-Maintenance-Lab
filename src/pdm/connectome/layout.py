from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import networkx as nx

from pdm.connectome.graph import as_node_id


def _anatomical_coords(graph: nx.DiGraph) -> dict[str, list[float]] | None:
    coords: dict[str, list[float]] = {}
    for node, data in graph.nodes(data=True):
        payload: Mapping[str, Any] = data or {}
        if all(k in payload for k in ("x", "y", "z")):
            coords[as_node_id(node)] = [float(payload["x"]), float(payload["y"]), float(payload["z"])]
            continue
        pos = payload.get("pos")
        if isinstance(pos, (list, tuple)) and len(pos) >= 3:
            coords[as_node_id(node)] = [float(pos[0]), float(pos[1]), float(pos[2])]
            continue
        return None
    if len(coords) != graph.number_of_nodes():
        return None
    return coords


def layout_positions(
    graph: nx.DiGraph,
    *,
    seed: int = 42,
    dim: int = 2,
) -> dict[str, list[float]]:
    """Anatomical 3D coordinates when present; otherwise a seeded spring layout."""
    anatomical = _anatomical_coords(graph)
    if anatomical is not None:
        return anatomical
    pos = nx.spring_layout(graph, dim=int(dim), seed=int(seed))
    return {as_node_id(n): [float(v) for v in xy] for n, xy in pos.items()}
