from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import networkx as nx

from pdm.connectome.graph import as_node_id

LAYOUT_METHOD_SPRING = "spring"
LAYOUT_METHOD_ANATOMICAL = "anatomical"


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


def layout_method(graph: nx.DiGraph) -> str:
    """Tag for new layout.json files. Existing files without ``method`` are spring."""
    return LAYOUT_METHOD_ANATOMICAL if _anatomical_coords(graph) is not None else LAYOUT_METHOD_SPRING


def is_anatomical_positions(
    layout: Mapping[str, Any] | None = None,
    *,
    graph: nx.DiGraph | None = None,
) -> bool:
    """True only for soma/template xyz, never for ``layout_positions()`` spring.

    Inspects raw ``layout.json`` as written. Missing ``method`` ⇒ spring (including
    3D spring with a Z span). Do not infer anatomy from Z. Graph node ``x,y,z``
    attrs from a train-time soma join count as anatomical when present on every node.
    """
    if isinstance(layout, Mapping):
        method = layout.get("method")
        if method is None:
            method = layout.get("source")
        if method is not None and str(method).strip() != "":
            return str(method).strip().lower() == LAYOUT_METHOD_ANATOMICAL
        # Missing method: existing train artifacts are spring. Z-span is irrelevant.
    if graph is not None:
        return _anatomical_coords(graph) is not None
    return False


def layout_positions(
    graph: nx.DiGraph,
    *,
    seed: int = 42,
    dim: int = 2,
) -> dict[str, list[float]]:
    """Anatomical 3D coordinates when present; otherwise a seeded spring layout.

    2D and 3D spring results are both non-anatomical for explorer merge.
    """
    anatomical = _anatomical_coords(graph)
    if anatomical is not None:
        return anatomical
    pos = nx.spring_layout(graph, dim=int(dim), seed=int(seed))
    return {as_node_id(n): [float(v) for v in xy] for n, xy in pos.items()}
