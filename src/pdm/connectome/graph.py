from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import networkx as nx


def as_node_id(value: Any) -> str:
    return str(value)


def graph_from_edges(
    edges: Iterable[Mapping[str, Any]],
    nodes: Iterable[Any] | None = None,
    *,
    src_key: str = "src",
    dst_key: str = "dst",
    weight_key: str = "weight",
) -> nx.DiGraph:
    """Build a directed graph. Every ``node_id`` is stored as ``str``."""
    graph = nx.DiGraph()
    if nodes is not None:
        graph.add_nodes_from(as_node_id(n) for n in nodes)
    for rec in edges:
        src = as_node_id(rec[src_key])
        dst = as_node_id(rec[dst_key])
        weight = rec.get(weight_key, rec.get("synapse_count", 1))
        syn = float(weight)
        graph.add_node(src)
        graph.add_node(dst)
        graph.add_edge(src, dst, synapse_count=syn, weight=syn)
    return graph


def graph_from_payload(payload: Mapping[str, Any]) -> nx.DiGraph:
    """Rebuild a graph. Prefer stored ``node_order`` (W_res row indices) over insertion order."""
    nodes = payload.get("node_order") or payload.get("nodes")
    return graph_from_edges(payload.get("edges") or [], nodes)


def graph_to_payload(
    graph: nx.DiGraph,
    node_order: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Serialize graph. ``node_order`` is the W_res row index list when provided."""
    if node_order is not None:
        nodes = [as_node_id(n) for n in node_order]
        seen = set(nodes)
        for n in graph.nodes():
            nid = as_node_id(n)
            if nid not in seen:
                nodes.append(nid)
                seen.add(nid)
    else:
        nodes = [as_node_id(n) for n in graph.nodes()]
    edges = []
    for src, dst, data in graph.edges(data=True):
        weight = data.get("synapse_count", data.get("weight", 1.0))
        edges.append({"src": as_node_id(src), "dst": as_node_id(dst), "weight": float(weight)})
    payload: dict[str, Any] = {"nodes": nodes, "edges": edges}
    if node_order is not None:
        payload["node_order"] = [as_node_id(n) for n in node_order]
    return payload


def node_ids(graph: nx.DiGraph) -> list[str]:
    return [as_node_id(n) for n in graph.nodes()]
