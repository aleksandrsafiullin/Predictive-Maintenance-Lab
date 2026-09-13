from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from pdm.connectome.anatomy import SomaTable
from pdm.connectome.graph import as_node_id
from pdm.connectome.layout import LAYOUT_METHOD_ANATOMICAL, is_anatomical_positions
from pdm.connectome.provenance import (
    GRAPH_MODE_RANDOM_REWIRE,
    GRAPH_MODE_REAL,
    GRAPH_MODE_SYNTHETIC,
)
from pdm.io_util import read_json

CONTEXT_CAP = 80_000
SOMA_CONTEXT_CAPTION = "Soma context downsampled for display"
XYZ_DECIMALS = 6
_EPS = 1e-12

HULL_MODE_ANATOMY = "malecns_anatomy"
HULL_MODE_SCHEMATIC = "schematic_cns"


def _as_xyz(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) < 2:
        return None
    try:
        x = float(value[0])
        y = float(value[1])
        z = float(value[2]) if len(value) > 2 else 0.0
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
        return None
    return [x, y, z]


def _round_xyz(xyz: Sequence[float], ndigits: int = XYZ_DECIMALS) -> list[float]:
    return [round(float(xyz[0]), ndigits), round(float(xyz[1]), ndigits), round(float(xyz[2]), ndigits)]


def _whole_string_is_integer(nid: str) -> bool:
    s = str(nid).strip()
    if not s:
        return False
    if s[0] in "+-":
        return s[1:].isdigit()
    return s.isdigit()


def body_id_sort_key(nid: str) -> tuple[int, int | str]:
    """Numeric body-id when the whole string is an integer; else lexicographic."""
    s = str(nid)
    if _whole_string_is_integer(s):
        return (0, int(s))
    return (1, s)


def downsample_context_ids(ids: Sequence[str], cap: int) -> tuple[list[str], bool]:
    """Sort by numeric body-id, then even stride. Deterministic; no RNG."""
    ordered = sorted((as_node_id(n) for n in ids), key=body_id_sort_key)
    n = len(ordered)
    limit = int(cap)
    if n <= limit or limit <= 0:
        return ordered, False
    idx = np.unique(np.linspace(0, n - 1, num=limit, dtype=np.int64))
    return [ordered[int(i)] for i in idx.tolist()], True


def _optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        rec = read_json(path)
    except Exception:  # noqa: BLE001
        return {}
    return rec if isinstance(rec, dict) else {}


def _positive_int(value: Any) -> int | None:
    try:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return None
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _graph_n_nodes(graph: Any) -> int | None:
    """Full graph width from an artifact or nx graph — not a display-capped list."""
    if graph is None:
        return None
    if isinstance(graph, Mapping):
        hit = _positive_int(graph.get("n_nodes"))
        if hit is not None:
            return hit
        nodes = graph.get("node_order") or graph.get("nodes") or []
        if nodes:
            return _positive_int(len(list(nodes)))
        return None
    number = getattr(graph, "number_of_nodes", None)
    if callable(number):
        try:
            return _positive_int(number())
        except TypeError:
            pass
    nodes_attr = getattr(graph, "nodes", None)
    if nodes_attr is None:
        return None
    try:
        n = len(nodes_attr)
    except TypeError:
        try:
            n = len(list(nodes_attr))
        except TypeError:
            return None
    return _positive_int(n)


def resolve_n_model(
    *,
    n_model: int | None = None,
    provenance: Mapping[str, Any] | None = None,
    run_row: Mapping[str, Any] | None = None,
    graph_payload: Mapping[str, Any] | None = None,
    n_nodes_full: int | None = None,
    run_dir: Path | str | None = None,
    snapshot: Mapping[str, Any] | None = None,
    graph: Any | None = None,
) -> int | None:
    """ESN width from run/provenance/checkpoint/graph — never a live 512-capped length."""
    hit = _positive_int(n_model)
    if hit is not None:
        return hit
    rdir = Path(run_dir) if run_dir is not None else None
    prov = dict(provenance) if isinstance(provenance, Mapping) else {}
    if rdir is not None and not prov:
        prov = _optional_json(rdir / "connectome" / "provenance.json")
    for rec in (run_row, prov):
        if not rec:
            continue
        hit = _positive_int(rec.get("n_nodes"))
        if hit is not None:
            return hit
    snap = dict(snapshot or {})
    if rdir is not None and not snap:
        snap = _optional_json(rdir / "experiment_snapshot.json")
    model = snap.get("model") if isinstance(snap.get("model"), dict) else {}
    reservoir = model.get("reservoir") if isinstance(model.get("reservoir"), dict) else {}
    hit = _positive_int(reservoir.get("n_nodes"))
    if hit is not None:
        return hit
    hit = _positive_int(n_nodes_full)
    if hit is not None:
        return hit
    payload = dict(graph_payload or {})
    if rdir is not None and not payload:
        payload = _optional_json(rdir / "connectome" / "graph.json")
    nodes = payload.get("node_order") or payload.get("nodes") or []
    if nodes:
        hit = _positive_int(len(list(nodes)))
        if hit is not None:
            return hit
    return _graph_n_nodes(graph)


def _layout_positions_map(layout: Mapping[str, Any] | None) -> dict[str, list[float]]:
    if not isinstance(layout, Mapping):
        return {}
    raw = layout.get("positions") if isinstance(layout.get("positions"), dict) else None
    if raw is None:
        raw = {k: v for k, v in layout.items() if k not in {"node_order", "method", "source", "positions"}}
    out: dict[str, list[float]] = {}
    for key, val in (raw or {}).items():
        xyz = _as_xyz(val)
        if xyz is not None:
            out[as_node_id(key)] = xyz
    return out


def _graph_anatomical_coords(graph: Any) -> dict[str, list[float]] | None:
    if graph is None:
        return None
    from pdm.connectome.layout import _anatomical_coords

    try:
        return _anatomical_coords(graph)
    except Exception:  # noqa: BLE001
        return None


def _is_synthetic_run(
    *,
    graph_mode: str,
    is_synthetic: bool | None,
    provenance: Mapping[str, Any] | None,
    scene: Mapping[str, Any] | None,
) -> bool:
    if is_synthetic is True:
        return True
    rec = provenance or {}
    sc = scene or {}
    if bool(sc.get("is_synthetic")) or bool(rec.get("is_synthetic")):
        return True
    for mode in (graph_mode, rec.get("graph_mode"), sc.get("graph_mode")):
        if str(mode or "").strip().lower() == GRAPH_MODE_SYNTHETIC:
            return True
    rewire_mode = str(graph_mode or rec.get("graph_mode") or sc.get("graph_mode") or "").strip().lower()
    if rewire_mode == GRAPH_MODE_RANDOM_REWIRE:
        parent_mode = str(rec.get("parent_graph_mode") or sc.get("parent_graph_mode") or "").strip().lower()
        if parent_mode == GRAPH_MODE_SYNTHETIC or bool(rec.get("parent_is_synthetic")):
            return True
    return False


def _anatomy_join_allowed(
    *,
    graph_mode: str,
    is_synthetic: bool,
    provenance: Mapping[str, Any] | None,
) -> bool:
    """Real connectome, or random_rewire of a real parent. Synthetic parent → False."""
    if is_synthetic:
        return False
    mode = str(graph_mode or "").strip().lower()
    if mode == GRAPH_MODE_SYNTHETIC or mode == "":
        return False
    if mode == GRAPH_MODE_REAL:
        return True
    if mode != GRAPH_MODE_RANDOM_REWIRE:
        return False
    rec = provenance or {}
    if bool(rec.get("is_synthetic")) or bool(rec.get("parent_is_synthetic")):
        return False
    parent_mode = str(rec.get("parent_graph_mode") or "").strip().lower()
    if parent_mode == GRAPH_MODE_SYNTHETIC:
        return False
    if parent_mode == GRAPH_MODE_REAL:
        return True
    # Rewire of real writes is_synthetic=False; missing parent_graph_mode still joins.
    return True


def _soma_aabb(soma_positions: Mapping[str, Sequence[float]]) -> tuple[np.ndarray, np.ndarray] | None:
    pts: list[list[float]] = []
    for raw in soma_positions.values():
        xyz = _as_xyz(raw)
        if xyz is not None:
            pts.append(xyz)
    if not pts:
        return None
    arr = np.asarray(pts, dtype=float)
    return arr.min(axis=0), arr.max(axis=0)


def _xyz_in_aabb(xyz: Sequence[float], lo: np.ndarray, hi: np.ndarray, pad: float = 0.02) -> bool:
    span = np.maximum(hi - lo, _EPS)
    p = np.asarray(xyz, dtype=float)
    return bool(np.all(p >= lo - pad * span) and np.all(p <= hi + pad * span))


def _join_reservoir_somas(
    nodes: Sequence[str],
    soma_positions: Mapping[str, Sequence[float]],
    payload_positions: Mapping[str, Sequence[float]] | None = None,
) -> tuple[dict[str, list[float]], int]:
    """Match reservoir ids to soma xyz. Never pile unmatched onto one centroid.

    Payload xyz is used only when it already sits in the soma AABB (same space).
    Spring / schematic coordinates fail that test and are omitted from the cloud.
    """
    out: dict[str, list[float]] = {}
    n_unmatched = 0
    payload = payload_positions or {}
    aabb = _soma_aabb(soma_positions)
    for nid in nodes:
        xyz = _as_xyz(soma_positions.get(nid))
        if xyz is None:
            alt = _as_xyz(payload.get(nid))
            if alt is not None and aabb is not None and _xyz_in_aabb(alt, aabb[0], aabb[1]):
                xyz = alt
        if xyz is None:
            n_unmatched += 1
            continue
        out[nid] = xyz
    return out, n_unmatched


def hull_polyline_from_somas(points: Sequence[Sequence[float]]) -> list[list[float]]:
    """2D XY convex hull with median Z. Empty if <3 points or Qhull fails."""
    xyz: list[list[float]] = []
    for raw in points:
        item = _as_xyz(raw)
        if item is not None:
            xyz.append(item)
    if len(xyz) < 3:
        return []
    arr = np.asarray(xyz, dtype=float)
    zmed = float(np.median(arr[:, 2]))
    try:
        from scipy.spatial import ConvexHull, QhullError
    except ImportError:  # pragma: no cover
        return []
    try:
        hull = ConvexHull(arr[:, :2])
    except QhullError:
        return []
    poly = [[float(arr[i, 0]), float(arr[i, 1]), zmed] for i in hull.vertices.tolist()]
    if poly and poly[0] != poly[-1]:
        poly.append(list(poly[0]))
    return poly


def normalize_shared_bbox(
    reservoir: Mapping[str, Sequence[float]],
    context: Sequence[Sequence[float]],
    hull: Sequence[Sequence[float]],
) -> tuple[dict[str, list[float]], list[list[float]], list[list[float]], float]:
    """Center/scale reservoir, context, and hull with one transform (~unit box)."""
    chunks: list[list[float]] = []
    for val in reservoir.values():
        xyz = _as_xyz(val)
        if xyz is not None:
            chunks.append(xyz)
    for val in context:
        xyz = _as_xyz(val)
        if xyz is not None:
            chunks.append(xyz)
    for val in hull:
        xyz = _as_xyz(val)
        if xyz is not None:
            chunks.append(xyz)
    if not chunks:
        return {}, [], [], 1.0
    arr = np.asarray(chunks, dtype=float)
    lo = arr.min(axis=0)
    hi = arr.max(axis=0)
    center = (lo + hi) * 0.5
    span = float(np.max(hi - lo))
    scale = span if span > _EPS else 1.0

    def _xf(xyz: Sequence[float]) -> list[float]:
        return _round_xyz(
            (
                (float(xyz[0]) - float(center[0])) / scale,
                (float(xyz[1]) - float(center[1])) / scale,
                (float(xyz[2]) - float(center[2])) / scale,
            )
        )

    pos = {nid: _xf(xyz) for nid, xyz in reservoir.items() if _as_xyz(xyz) is not None}
    ctx = [_xf(xyz) for xyz in context if _as_xyz(xyz) is not None]
    hull_out = [_xf(xyz) for xyz in hull if _as_xyz(xyz) is not None]
    return pos, ctx, hull_out, float(scale)


def _fallback_positions(nodes: Sequence[str]) -> dict[str, list[float]]:
    from pdm.visualization.explorer import fallback_positions

    return fallback_positions(nodes)


def _load_scene(run_dir: Path, node_order: Sequence[str] | None) -> dict[str, Any]:
    from pdm.visualization.explorer import load_scene_from_run

    return load_scene_from_run(run_dir, node_order=node_order)


def _soma_positions(soma: SomaTable | Mapping[str, Any] | None) -> dict[str, list[float]]:
    if soma is None:
        return {}
    raw = soma.positions if isinstance(soma, SomaTable) else soma.get("positions")
    out: dict[str, list[float]] = {}
    for key, val in (raw or {}).items():
        xyz = _as_xyz(val)
        if xyz is not None:
            out[as_node_id(key)] = xyz
    return out


def build_anatomy_scene(
    *,
    scene: Mapping[str, Any] | None = None,
    run_dir: Path | str | None = None,
    soma: SomaTable | Mapping[str, Any] | None = None,
    graph_mode: str | None = None,
    is_synthetic: bool | None = None,
    n_model: int | None = None,
    context_cap: int = CONTEXT_CAP,
    layout: Mapping[str, Any] | None = None,
    graph: Any | None = None,
    provenance: Mapping[str, Any] | None = None,
    run_row: Mapping[str, Any] | None = None,
    n_nodes_full: int | None = None,
    node_order: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Merge reservoir xyz with full soma context. Never expands ``states``.

    ``flags.n_model`` is ESN width from the run/provenance, not display-capped
    ``len(nodes)``. Context is empty for synthetic runs. Spring ``layout.json``
    (missing ``method``, including 3D spring) is non-anatomical; a real / rewired-real
    soma table wins over spring.
    """
    rdir = Path(run_dir) if run_dir is not None else None
    rec = dict(scene or {})
    graph_payload: dict[str, Any] = {}
    layout_rec = dict(layout) if isinstance(layout, Mapping) else {}
    prov = dict(provenance or {})
    if rdir is not None:
        cdir = rdir / "connectome"
        if not prov:
            prov = _optional_json(cdir / "provenance.json")
        graph_payload = _optional_json(cdir / "graph.json")
        if not layout_rec:
            layout_rec = _optional_json(cdir / "layout.json")
        if not rec:
            rec = _load_scene(rdir, node_order)
        if n_nodes_full is None:
            full_nodes = graph_payload.get("node_order") or graph_payload.get("nodes") or []
            if full_nodes:
                n_nodes_full = len(list(full_nodes))

    nodes = [as_node_id(n) for n in (node_order or rec.get("nodes") or [])]
    if not nodes and graph_payload:
        nodes = [as_node_id(n) for n in (graph_payload.get("node_order") or graph_payload.get("nodes") or [])]
    scene_pos: dict[str, list[float]] = {}
    for key, val in (rec.get("positions") or {}).items():
        xyz = _as_xyz(val)
        if xyz is not None:
            scene_pos[as_node_id(key)] = xyz
    layout_pos = _layout_positions_map(layout_rec)

    mode = str(graph_mode if graph_mode is not None else rec.get("graph_mode") or prov.get("graph_mode") or "")
    synthetic = _is_synthetic_run(
        graph_mode=mode,
        is_synthetic=is_synthetic if is_synthetic is not None else rec.get("is_synthetic"),
        provenance=prov,
        scene=rec,
    )
    if is_synthetic is None:
        is_synthetic = synthetic
    else:
        is_synthetic = bool(is_synthetic) or synthetic

    width = resolve_n_model(
        n_model=n_model,
        provenance=prov,
        run_row=run_row,
        graph_payload=graph_payload,
        n_nodes_full=n_nodes_full,
        run_dir=rdir,
        graph=graph,
    )
    n_display = len(nodes)

    join_ok = _anatomy_join_allowed(graph_mode=mode, is_synthetic=is_synthetic, provenance=prov)
    soma_pos = _soma_positions(soma) if join_ok else {}
    soma_usable = bool(soma_pos)

    anatomical_layout = is_anatomical_positions(layout_rec) if layout_rec else False
    graph_coords = _graph_anatomical_coords(graph)
    anatomical_graph = graph_coords is not None

    reservoir_pos: dict[str, list[float]] = {}
    n_unmatched = 0
    hull_mode = HULL_MODE_SCHEMATIC
    anatomy_missing = False
    context_ids: list[str] = []
    context_downsampled = False
    caption = ""

    if is_synthetic:
        hull_mode = HULL_MODE_SCHEMATIC
        anatomy_missing = False
        if anatomical_layout:
            reservoir_pos = {nid: layout_pos[nid] for nid in nodes if nid in layout_pos}
        elif anatomical_graph:
            reservoir_pos = {nid: graph_coords[nid] for nid in nodes if nid in graph_coords}
        else:
            reservoir_pos = {nid: scene_pos[nid] for nid in nodes if nid in scene_pos}
    elif anatomical_layout:
        reservoir_pos = {nid: layout_pos[nid] for nid in nodes if nid in layout_pos}
        if soma_usable:
            hull_mode = HULL_MODE_ANATOMY
            context_ids, context_downsampled = downsample_context_ids(soma_pos.keys(), context_cap)
        elif join_ok:
            anatomy_missing = True
    elif anatomical_graph:
        reservoir_pos = {nid: graph_coords[nid] for nid in nodes if nid in graph_coords}
        if soma_usable:
            hull_mode = HULL_MODE_ANATOMY
            context_ids, context_downsampled = downsample_context_ids(soma_pos.keys(), context_cap)
        elif join_ok:
            anatomy_missing = True
    elif join_ok and soma_usable:
        hull_mode = HULL_MODE_ANATOMY
        joined, n_unmatched = _join_reservoir_somas(nodes, soma_pos, payload_positions=scene_pos)
        reservoir_pos = joined
        context_ids, context_downsampled = downsample_context_ids(soma_pos.keys(), context_cap)
    else:
        if join_ok:
            anatomy_missing = True
        reservoir_pos = {nid: scene_pos[nid] for nid in nodes if nid in scene_pos}

    if hull_mode == HULL_MODE_ANATOMY:
        # Unmatched reservoir ids stay in nodes/states but are not drawn in a clump.
        reservoir_pos = {nid: reservoir_pos[nid] for nid in nodes if nid in reservoir_pos}
    elif not reservoir_pos:
        reservoir_pos = _fallback_positions(nodes)
    else:
        missing = [nid for nid in nodes if nid not in reservoir_pos]
        if missing:
            fb = _fallback_positions(nodes)
            for nid in missing:
                reservoir_pos[nid] = fb[nid]

    context_xyz: list[list[float]] = []
    hull: list[list[float]] = []
    if hull_mode == HULL_MODE_ANATOMY and context_ids:
        context_xyz = [list(soma_pos[i]) for i in context_ids if i in soma_pos]
        hull = hull_polyline_from_somas(context_xyz)
        caption = SOMA_CONTEXT_CAPTION if context_downsampled else ""
    else:
        context_xyz = []
        hull = []
        context_downsampled = False
        caption = ""

    reservoir_pos, context_xyz, hull, anatomy_scale = normalize_shared_bbox(
        {nid: reservoir_pos[nid] for nid in nodes if nid in reservoir_pos},
        context_xyz,
        hull,
    )

    n_viz = len(context_xyz)
    flags: dict[str, Any] = {
        "hull_mode": hull_mode,
        "n_nodes_display": int(n_display),
        "n_viz": int(n_viz),
        "context_n": int(n_viz),
        "context_downsampled": bool(context_downsampled),
        "anatomy_missing": bool(anatomy_missing),
        "n_unmatched_reservoir": int(n_unmatched),
        "anatomy_scale": float(anatomy_scale),
        "context_caption": caption,
        "graph_mode": mode,
        "is_synthetic": bool(is_synthetic),
    }
    if width is not None:
        flags["n_model"] = int(width)
    if hull_mode == HULL_MODE_SCHEMATIC:
        flags.update(
            {
                "n_viz": 0,
                "context_n": 0,
                "context_downsampled": False,
                "context_caption": "",
            }
        )
        context_xyz = []
        hull = []

    return {
        "nodes": nodes,
        "positions": reservoir_pos,
        "context_positions": context_xyz,
        "hull_polyline": hull,
        "flags": flags,
        "layout_method": LAYOUT_METHOD_ANATOMICAL if (anatomical_layout or anatomical_graph) else "spring",
    }
