from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pdm.connectome.provenance import (
    GRAPH_MODE_RANDOM_REWIRE,
    GRAPH_MODE_SYNTHETIC,
    SYNTHETIC_DISCLAIMER,
)
from pdm.io_util import read_json
from pdm.replay import slice_predictions_to_replay_time

EXPLORER_DISCLAIMER = (
    "Computational activity in a connectome-based reservoir. "
    "This is not a biophysical simulation or recorded activity of a living fly brain."
)
EXPLORER_MODES = (
    "Overview",
    "Equipment replay",
    "Inside prediction window",
    "Alert inspection",
)
INLINE_TRACE_MAX_NODES = 128
WORKER_BUSY_MESSAGE = "A heavy job is already running. Please wait."
RESERVOIR_REQUIRED_MESSAGE = "Neural Activity Explorer requires a reservoir run."
ANATOMY_MISSING_CAPTION = (
    "Anatomical soma coordinates are missing; showing a labeled schematic, not FlyEM MaleCNS anatomy."
)
SOMA_DOWNSAMPLE_CAPTION = "Soma context downsampled for display"
LIVE_CONTEXT_CAP = 20_000
_SOMA_TABLE_CACHE: dict[tuple[str, int, int], Any] = {}


def synthetic_banner_required(rec: Mapping[str, Any] | None, scene: Mapping[str, Any] | None = None) -> bool:
    """True when the selected run is a synthetic fixture (or labeled synthetic)."""
    rec = rec or {}
    scene = scene or {}
    if bool(rec.get("is_synthetic")) or bool(scene.get("is_synthetic")):
        return True
    if bool(rec.get("parent_is_synthetic")) or bool(scene.get("parent_is_synthetic")):
        return True
    for mode in (rec.get("graph_mode"), scene.get("graph_mode")):
        if str(mode or "").strip().lower() == GRAPH_MODE_SYNTHETIC:
            return True
    rewire = str(rec.get("graph_mode") or scene.get("graph_mode") or "").strip().lower()
    if rewire == GRAPH_MODE_RANDOM_REWIRE:
        for parent in (rec.get("parent_graph_mode"), scene.get("parent_graph_mode")):
            if str(parent or "").strip().lower() == GRAPH_MODE_SYNTHETIC:
                return True
    return False


def fallback_positions(nodes: Sequence[str]) -> dict[str, list[float]]:
    """Ring layout so the scene is never blank when xyz / layout.json are missing."""
    ids = [str(n) for n in nodes]
    n = max(len(ids), 1)
    out: dict[str, list[float]] = {}
    for i, nid in enumerate(ids):
        angle = 2.0 * math.pi * i / n
        out[nid] = [math.cos(angle), math.sin(angle), 0.0]
    return out


def _as_xyz(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    x = float(value[0])
    y = float(value[1])
    z = float(value[2]) if len(value) > 2 else 0.0
    return [x, y, z]


def load_scene_from_run(rdir: Path, node_order: Sequence[str] | None = None) -> dict[str, Any]:
    """Nodes, edges, and coordinates from connectome artifacts. Never a blank graph."""
    cdir = Path(rdir) / "connectome"
    nodes: list[str] = []
    edges: list[dict[str, Any]] = []
    positions: dict[str, list[float]] = {}
    graph_mode = ""
    is_synthetic = False
    parent_graph_mode = ""
    parent_is_synthetic = False

    graph_path = cdir / "graph.json"
    if graph_path.exists():
        payload = read_json(graph_path)
        nodes = [str(n) for n in (payload.get("node_order") or payload.get("nodes") or [])]
        edges = [dict(e) for e in (payload.get("edges") or [])]

    layout_path = cdir / "layout.json"
    if layout_path.exists():
        layout = read_json(layout_path)
        raw_pos = layout.get("positions") if isinstance(layout, dict) else None
        if not isinstance(raw_pos, dict):
            raw_pos = layout if isinstance(layout, dict) else {}
        for key, val in (raw_pos or {}).items():
            if key == "node_order":
                continue
            xyz = _as_xyz(val)
            if xyz is not None:
                positions[str(key)] = xyz
        if not nodes and layout.get("node_order"):
            nodes = [str(n) for n in layout["node_order"]]

    prov_path = cdir / "provenance.json"
    if prov_path.exists():
        prov = read_json(prov_path)
        graph_mode = str(prov.get("graph_mode") or "")
        is_synthetic = bool(prov.get("is_synthetic", graph_mode == GRAPH_MODE_SYNTHETIC))
        parent_graph_mode = str(prov.get("parent_graph_mode") or "")
        parent_is_synthetic = bool(prov.get("parent_is_synthetic"))

    if node_order:
        order = [str(n) for n in node_order]
        extra = [n for n in nodes if n not in order]
        nodes = order + extra
    if not nodes:
        nodes = list(positions.keys())
    if not positions:
        positions = fallback_positions(nodes)
    else:
        for nid in nodes:
            if nid not in positions:
                positions[nid] = [0.0, 0.0, 0.0]
    return {
        "nodes": nodes,
        "edges": edges,
        "positions": positions,
        "graph_mode": graph_mode,
        "is_synthetic": is_synthetic,
        "parent_graph_mode": parent_graph_mode,
        "parent_is_synthetic": parent_is_synthetic,
        "disclaimer": SYNTHETIC_DISCLAIMER if is_synthetic or graph_mode == GRAPH_MODE_SYNTHETIC else "",
    }


def subset_scene(scene: Mapping[str, Any], keep_nodes: Sequence[str]) -> dict[str, Any]:
    """Restrict reservoir nodes/edges/positions to ``keep_nodes``. Context is unchanged."""
    keep = [str(n) for n in keep_nodes]
    keep_set = set(keep)
    rec = dict(scene or {})
    positions = rec.get("positions") or {}
    context = rec.get("context_positions")
    hull = rec.get("hull_polyline")
    rec["nodes"] = keep
    rec["positions"] = {str(k): list(v) for k, v in positions.items() if str(k) in keep_set}
    rec["edges"] = [
        dict(e)
        for e in (rec.get("edges") or [])
        if str(e.get("src")) in keep_set and str(e.get("dst")) in keep_set
    ]
    if context is not None:
        rec["context_positions"] = [list(p) for p in context]
    if hull is not None:
        rec["hull_polyline"] = [list(p) for p in hull]
    flags = rec.get("flags")
    if isinstance(flags, Mapping):
        flag_out = dict(flags)
        flag_out["n_nodes_display"] = len(keep)
        rec["flags"] = flag_out
    return rec


def is_active_train_live(
    *,
    worker_alive: bool,
    worker_kind: object,
    live: Mapping[str, Any] | None,
) -> bool:
    """True only for an in-process train job with reservoir activity arrays."""
    if not worker_alive or str(worker_kind or "") != "train":
        return False
    if not live or str(live.get("status") or "") == "not_reservoir":
        return False
    return live.get("states") is not None


def explorer_overlay_clock(
    trace: Mapping[str, Any] | None,
    *,
    mode: str = "Overview",
    replay_step: int = 0,
    alert_ts: float | None = None,
    view: Mapping[str, Any] | None = None,
    live: Mapping[str, Any] | None = None,
    train_live: bool = False,
) -> float | None:
    """Selected-unit Now from the unit trace / replay meas. ``live`` is ignored."""
    del live, train_live
    if mode == "Equipment replay":
        meas = (view or {}).get("meas")
        if isinstance(meas, pd.DataFrame) and not meas.empty and "timestamp_s" in meas.columns:
            step = max(0, min(len(meas) - 1, int(replay_step)))
            val = meas.iloc[step].get("timestamp_s")
            if _finite_clock(val):
                return float(val)
    frames = list((trace or {}).get("frame_map") or [])
    if mode == "Alert inspection" and _finite_clock(alert_ts):
        return float(alert_ts)
    if not frames:
        return None
    if mode == "Equipment replay":
        idx = max(0, min(len(frames) - 1, int(replay_step)))
        ts = frames[idx].get("timestamp_s")
        return float(ts) if _finite_clock(ts) else None
    last = frames[-1]
    ts = last.get("window_end_timestamp_s")
    if not _finite_clock(ts):
        ts = last.get("timestamp_s")
    return float(ts) if _finite_clock(ts) else None


def selected_trace_predicted_rul(trace: Mapping[str, Any] | None) -> float | None:
    """Stored forecast for the selected unit. Never a live-training probe RUL."""
    if not trace:
        return None
    val = trace.get("predicted_rul_s")
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        out = float(val)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _finite_clock(val: Any) -> bool:
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return False
        return math.isfinite(float(val))
    except (TypeError, ValueError):
        return False


_ANATOMY_FLAG_KEYS = (
    "hull_mode",
    "n_model",
    "n_viz",
    "context_n",
    "n_nodes_display",
    "context_downsampled",
    "anatomy_missing",
    "n_unmatched_reservoir",
    "anatomy_scale",
    "context_caption",
)


def build_explorer_payload(
    *,
    nodes: Sequence[str],
    edges: Sequence[Mapping[str, Any]] | None,
    positions: Mapping[str, Sequence[float]] | None,
    states: Any,
    frame_map: Sequence[Mapping[str, Any]] | None,
    flags: Mapping[str, Any] | None,
    labels: Mapping[str, Any] | None = None,
    inputs: Any = None,
    predicted_rul_s: Any = None,
    anatomy: Mapping[str, Any] | None = None,
    soma: Any = None,
    n_model: int | None = None,
    run_dir: Path | None = None,
    graph: Any | None = None,
    context_cap: int | None = None,
    context_positions: Sequence[Sequence[float]] | None = None,
    hull_polyline: Sequence[Sequence[float]] | None = None,
) -> dict[str, Any]:
    """JSON-serializable args for the Neural Activity Explorer component.

    Consumes ``build_anatomy_scene`` when ``anatomy`` / ``soma`` / ``run_dir`` is
    given. Never expands ``states`` to N_viz. ``n_model`` is the full trained
    ESN width; ``flags.n_model`` is never ``len(nodes)`` after a live cap.
    """
    from pdm.visualization.scene import CONTEXT_CAP, build_anatomy_scene, resolve_n_model

    anatomy_rec: dict[str, Any] = dict(anatomy) if anatomy else {}
    if not anatomy_rec and (soma is not None or run_dir is not None):
        flag_in = dict(flags or {})
        anatomy_rec = build_anatomy_scene(
            scene={
                "nodes": [str(n) for n in nodes],
                "edges": [dict(e) for e in (edges or [])],
                "positions": dict(positions or {}),
                "graph_mode": flag_in.get("graph_mode"),
                "is_synthetic": flag_in.get("is_synthetic"),
            },
            soma=soma,
            n_model=n_model if n_model is not None else flag_in.get("n_model"),
            n_nodes_full=n_model,
            run_dir=run_dir,
            graph=graph,
            context_cap=int(context_cap) if context_cap is not None else CONTEXT_CAP,
            provenance=flag_in.get("provenance") if isinstance(flag_in.get("provenance"), Mapping) else None,
            run_row=flag_in.get("run_row") if isinstance(flag_in.get("run_row"), Mapping) else None,
        )
    n_nodes = len(nodes)
    if states is None:
        arr = np.zeros((0, n_nodes), dtype=float)
    else:
        arr = np.asarray(states, dtype=float)
        if arr.ndim == 1:
            arr = np.zeros((0, n_nodes), dtype=float) if arr.size == 0 else arr.reshape(1, -1)
        if arr.ndim != 2:
            arr = np.zeros((0, n_nodes), dtype=float)
    inp = np.asarray(inputs, dtype=float) if inputs is not None else np.zeros((0, 0), dtype=float)
    if inp.ndim == 1:
        inp = inp.reshape(1, -1)
    if inp.ndim != 2:
        inp = np.zeros((int(arr.shape[0]), 0), dtype=float)
    pos: dict[str, list[float]] = {}
    src_pos = anatomy_rec.get("positions") if anatomy_rec.get("positions") else positions
    for key, val in (src_pos or {}).items():
        xyz = _as_xyz(val)
        if xyz is not None:
            pos[str(key)] = xyz
    node_ids = [str(n) for n in nodes]
    if node_ids:
        pos = {nid: pos[nid] for nid in node_ids if nid in pos}
    ctx_raw = context_positions
    if ctx_raw is None:
        ctx_raw = anatomy_rec.get("context_positions") or []
    hull_raw = hull_polyline
    if hull_raw is None:
        hull_raw = anatomy_rec.get("hull_polyline") or []
    ctx = [xyz for xyz in (_as_xyz(p) for p in ctx_raw) if xyz is not None]
    hull = [xyz for xyz in (_as_xyz(p) for p in hull_raw) if xyz is not None]
    flag_out = dict(flags or {})
    aflags = anatomy_rec.get("flags") if isinstance(anatomy_rec.get("flags"), Mapping) else {}
    for key in _ANATOMY_FLAG_KEYS:
        if key in aflags:
            flag_out[key] = aflags[key]
    if n_model is not None:
        flag_out["n_model"] = int(n_model)
    flag_out.setdefault("hull_mode", "schematic_cns")
    flag_out.setdefault("downsampled", False)
    flag_out.setdefault("phase", "overview")
    flag_out.setdefault("context_downsampled", False)
    flag_out.setdefault("anatomy_missing", False)
    flag_out["n_nodes_display"] = n_nodes
    flag_out["n_viz"] = len(ctx)
    flag_out["context_n"] = len(ctx)
    if flag_out.get("n_model") is None:
        width = resolve_n_model(
            n_model=n_model,
            run_dir=run_dir,
            n_nodes_full=n_model,
            graph=graph,
            provenance=flag_out.get("provenance") if isinstance(flag_out.get("provenance"), Mapping) else None,
            run_row=flag_out.get("run_row") if isinstance(flag_out.get("run_row"), Mapping) else None,
        )
        if width is not None:
            flag_out["n_model"] = int(width)
        else:
            flag_out.pop("n_model", None)
    pred: float | list[float] | None
    if predicted_rul_s is None:
        pred = None
    elif isinstance(predicted_rul_s, (list, tuple, np.ndarray)):
        pred_arr = np.asarray(predicted_rul_s, dtype=float).reshape(-1)
        pred = [float(v) for v in pred_arr.tolist()]
    else:
        try:
            pred = float(predicted_rul_s)
            if not np.isfinite(pred):
                pred = None
        except (TypeError, ValueError):
            pred = None
    payload: dict[str, Any] = {
        "nodes": [str(n) for n in nodes],
        "edges": [dict(e) for e in (edges or [])],
        "positions": pos,
        "states": arr.tolist(),
        "inputs": inp.tolist(),
        "predicted_rul_s": pred,
        "frame_map": [dict(f) for f in (frame_map or [])],
        "flags": flag_out,
        "context_positions": ctx,
        "hull_polyline": hull,
    }
    if labels is not None:
        payload["labels"] = dict(labels)
    return payload


def _soma_cache_key(path: Path | str | None) -> tuple[str, int, int]:
    from pdm.connectome.anatomy import default_soma_dir, find_soma_table_path

    if path is not None:
        loc = Path(path).expanduser().resolve()
        if loc.is_file():
            st = loc.stat()
            return (str(loc), int(st.st_mtime_ns), int(st.st_size))
        if loc.is_dir():
            found = find_soma_table_path(loc)
            if found is not None and found.is_file():
                st = found.stat()
                return (str(found.resolve()), int(st.st_mtime_ns), int(st.st_size))
            return (str(loc), 0, -1)
    found = find_soma_table_path()
    if found is not None and found.is_file():
        st = found.stat()
        return (str(found.resolve()), int(st.st_mtime_ns), int(st.st_size))
    root = Path(default_soma_dir()).expanduser().resolve()
    return (str(root), 0, -1)


def clear_soma_table_cache() -> None:
    """Drop cached soma tables. Tests call this after monkeypatching the soma dir."""
    _SOMA_TABLE_CACHE.clear()


def cached_soma_table(path: Path | str | None = None) -> Any:
    """Load soma xyz once per resolved path + mtime. Re-raises schema ``ValueError``."""
    from pdm.connectome.anatomy import load_soma_table

    key = _soma_cache_key(path)
    hit = _SOMA_TABLE_CACHE.get(key)
    if hit is not None:
        if isinstance(hit, BaseException):
            raise hit
        return hit
    try:
        table = load_soma_table(path)
    except ValueError as exc:
        _SOMA_TABLE_CACHE[key] = exc
        raise
    _SOMA_TABLE_CACHE[key] = table
    return table


def soma_join_allowed(
    rec: Mapping[str, Any] | None = None,
    scene: Mapping[str, Any] | None = None,
) -> bool:
    """Real connectome, or random_rewire of a real parent. Synthetic never joins FlyEM xyz."""
    from pdm.visualization.scene import _anatomy_join_allowed, _is_synthetic_run

    rec = rec or {}
    scene = scene or {}
    mode = str(rec.get("graph_mode") or scene.get("graph_mode") or "")
    synthetic = _is_synthetic_run(
        graph_mode=mode,
        is_synthetic=rec.get("is_synthetic") if rec.get("is_synthetic") is not None else scene.get("is_synthetic"),
        provenance=rec,
        scene=scene,
    )
    prov = {
        "graph_mode": mode,
        "is_synthetic": synthetic,
        "parent_graph_mode": rec.get("parent_graph_mode") or scene.get("parent_graph_mode"),
        "parent_is_synthetic": rec.get("parent_is_synthetic") or scene.get("parent_is_synthetic"),
    }
    return _anatomy_join_allowed(graph_mode=mode, is_synthetic=bool(synthetic), provenance=prov)


def apply_anatomy_schema_fallback(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Schematic + anatomy_missing after a caught soma load failure. Empty context."""
    rec = dict(payload)
    flags = dict(rec.get("flags") or {})
    flags["hull_mode"] = "schematic_cns"
    flags["anatomy_missing"] = True
    flags["n_viz"] = 0
    flags["context_n"] = 0
    flags["context_downsampled"] = False
    flags["context_caption"] = ""
    rec["flags"] = flags
    rec["context_positions"] = []
    rec["hull_polyline"] = []
    return rec


def build_ui_explorer_payload(
    *,
    nodes: Sequence[str],
    edges: Sequence[Mapping[str, Any]] | None,
    positions: Mapping[str, Sequence[float]] | None,
    states: Any,
    frame_map: Sequence[Mapping[str, Any]] | None,
    flags: Mapping[str, Any] | None,
    labels: Mapping[str, Any] | None = None,
    inputs: Any = None,
    predicted_rul_s: Any = None,
    run_dir: Path | None = None,
    n_model: int | None = None,
    soma_path: Path | str | None = None,
    join_soma: bool = True,
    context_cap: int | None = None,
    graph: Any | None = None,
) -> tuple[dict[str, Any], str | None]:
    """UI-safe payload: cached soma, scene helper wins ``hull_mode``.

    Soma load failures (unknown-schema ``ValueError``, Arrow/OSError, …) become
    schematic + ``anatomy_missing`` with the exception text. Overlay / trace /
    processed-data loads are not swallowed here.
    """
    flag_in = dict(flags or {})
    flag_in.pop("hull_mode", None)
    schema_error: str | None = None
    soma = None
    if join_soma:
        try:
            soma = cached_soma_table(soma_path)
        except Exception as exc:  # noqa: BLE001
            schema_error = str(exc)
            soma = None
    payload = build_explorer_payload(
        nodes=nodes,
        edges=edges,
        positions=positions,
        states=states,
        frame_map=frame_map,
        flags=flag_in,
        labels=labels,
        inputs=inputs,
        predicted_rul_s=predicted_rul_s,
        soma=soma,
        n_model=n_model,
        run_dir=run_dir,
        graph=graph,
        context_cap=context_cap,
    )
    if schema_error is not None:
        payload = apply_anatomy_schema_fallback(payload)
    return payload, schema_error


def explorer_anatomy_captions(
    payload: Mapping[str, Any] | None,
    *,
    is_synthetic: bool,
    schema_error: str | None = None,
) -> list[str]:
    """Visible Streamlit captions. Synthetic never uses the FlyEM anatomy-missing line."""
    out: list[str] = []
    flags = dict((payload or {}).get("flags") or {})
    if is_synthetic:
        return out
    if schema_error:
        out.append(str(schema_error))
        out.append(ANATOMY_MISSING_CAPTION)
        return out
    if flags.get("anatomy_missing"):
        out.append(ANATOMY_MISSING_CAPTION)
        return out
    n_viz = flags.get("n_viz")
    n_model = flags.get("n_model")
    if flags.get("hull_mode") == "malecns_anatomy":
        if n_viz is not None and n_model is not None:
            out.append(f"N_viz={int(n_viz)} vs N_model={int(n_model)}")
        elif n_viz is not None:
            out.append(f"N_viz={int(n_viz)}")
    caption = str(flags.get("context_caption") or "")
    if flags.get("context_downsampled") and not caption:
        caption = SOMA_DOWNSAMPLE_CAPTION
    if caption:
        out.append(caption)
    return out


def collect_alert_rows(rdir: Path, unit_id: str | None) -> list[dict[str, Any]]:
    """Alert episode rows for Alert inspection. Empty list if none; never raises."""
    from pdm.experiments import (
        legacy_evaluation_artifacts,
        list_evaluations,
        resolve_evaluation_artifacts,
    )

    paths: list[Path] = []
    try:
        for ev in list_evaluations(rdir):
            art = resolve_evaluation_artifacts(rdir, ev.get("eval_id"))
            alerts = art.get("alerts")
            if alerts is not None:
                paths.append(Path(alerts))
        legacy = legacy_evaluation_artifacts(rdir).get("alerts")
        if legacy is not None:
            paths.append(Path(legacy))
    except Exception:  # noqa: BLE001
        return []

    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()
    for path in paths:
        try:
            frame = pd.read_csv(path)
        except Exception:  # noqa: BLE001
            continue
        if unit_id and "unit_id" in frame.columns:
            frame = frame[frame["unit_id"].astype(str) == str(unit_id)]
        for rec in frame.to_dict(orient="records"):
            key = (
                rec.get("unit_id"),
                rec.get("timestamp_s"),
                rec.get("type") or rec.get("alert_status"),
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(rec)
    return rows


def causal_prefix_rows(frame: pd.DataFrame, timestamp_s: float) -> pd.DataFrame:
    """Keep rows with timestamp_s ≤ T. No future measurements or predictions."""
    if frame is None or getattr(frame, "empty", True) or "timestamp_s" not in frame.columns:
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    ts = pd.to_numeric(frame["timestamp_s"], errors="coerce")
    return frame.loc[ts <= float(timestamp_s)].copy()


def alert_jump_target(episode: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Read-only unit + timestamp + stored predicted RUL from an alert episode."""
    if not episode:
        return None
    ts = episode.get("timestamp_s")
    if ts is None or (isinstance(ts, float) and pd.isna(ts)):
        return None
    rul = episode.get("predicted_rul_s")
    stored = None
    try:
        if rul is not None and not (isinstance(rul, float) and pd.isna(rul)):
            stored = float(rul)
    except (TypeError, ValueError):
        stored = None
    unit = episode.get("unit_id")
    return {
        "unit_id": None if unit is None else str(unit),
        "timestamp_s": float(ts),
        "predicted_rul_s": stored,
    }


def stored_alert_prediction(
    episode: Mapping[str, Any] | None,
    predictions: pd.DataFrame | None = None,
    *,
    unit_id: str | None = None,
    timestamp_s: float | None = None,
) -> float | None:
    """Stored predicted_rul_s at ≤ T. Never a rescore that uses future rows."""
    target = alert_jump_target(episode)
    if target and target.get("predicted_rul_s") is not None:
        return float(target["predicted_rul_s"])
    t = timestamp_s if timestamp_s is not None else (target or {}).get("timestamp_s")
    uid = unit_id if unit_id is not None else (target or {}).get("unit_id")
    if predictions is None or t is None:
        return None
    pred = predictions
    if uid is not None and "unit_id" in pred.columns:
        pred = pred[pred["unit_id"].astype(str) == str(uid)]
    sliced = slice_predictions_to_replay_time(pred, float(t))
    if sliced.empty or "predicted_rul_s" not in sliced.columns:
        return None
    val = sliced.iloc[-1].get("predicted_rul_s")
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        return float(val)
    except (TypeError, ValueError):
        return None


def slice_trace_to_alert(
    trace: Mapping[str, Any] | None,
    episode: Mapping[str, Any] | None,
    *,
    timestamp_s: float | None = None,
    predictions: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Restrict a saved trace to frames with timestamp_s ≤ alert time.

    Predicted RUL is the stored evaluation/alert value, not a rescore from future rows.
    """
    rec = dict(trace or {})
    target = alert_jump_target(episode) or {}
    t = timestamp_s if timestamp_s is not None else target.get("timestamp_s")
    frames = list(rec.get("frame_map") or [])
    if t is None:
        rec["frame_map"] = [dict(f) for f in frames]
        rec["predicted_rul_s"] = stored_alert_prediction(
            episode, predictions, timestamp_s=t, unit_id=target.get("unit_id")
        )
        return rec
    t = float(t)
    keep_idx = [
        i
        for i, frm in enumerate(frames)
        if frm.get("timestamp_s") is not None and float(frm["timestamp_s"]) <= t
    ]
    rec["frame_map"] = [dict(frames[i]) for i in keep_idx]
    states = rec.get("states")
    arr = np.asarray(states) if states is not None else None
    if arr is not None and arr.ndim == 2 and arr.shape[0] == len(frames):
        rec["states"] = arr[keep_idx] if keep_idx else arr[0:0]
    inputs = rec.get("inputs")
    inp = np.asarray(inputs) if inputs is not None else None
    if inp is not None and inp.ndim == 2 and inp.shape[0] == len(frames):
        rec["inputs"] = inp[keep_idx] if keep_idx else inp[0:0]
    rec["predicted_rul_s"] = stored_alert_prediction(
        episode,
        predictions,
        timestamp_s=t,
        unit_id=target.get("unit_id"),
    )
    return rec
