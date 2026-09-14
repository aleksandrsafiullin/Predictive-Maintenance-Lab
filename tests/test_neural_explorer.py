from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from pdm.connectome.provenance import SYNTHETIC_DISCLAIMER
from pdm.paths import project_root
from pdm.visualization.explorer import (
    ANATOMY_MISSING_CAPTION,
    EXPLORER_DISCLAIMER,
    RESERVOIR_REQUIRED_MESSAGE,
    SOMA_DOWNSAMPLE_CAPTION,
    build_explorer_payload,
    clear_soma_table_cache,
    explorer_anatomy_captions,
)

FRONTEND = project_root() / "src" / "pdm" / "visualization" / "component" / "frontend"
CDN_HOSTS = ("unpkg", "cdn.jsdelivr", "cdnjs", "googleapis")
SCRIPT_HTTPS_SRC = re.compile(
    r"""<script[^>]+(?:src|href)\s*=\s*["']https://""",
    re.IGNORECASE,
)
MODULE_HTTPS = re.compile(
    r"""(?:from|import)\s+["']https://""",
    re.IGNORECASE,
)


def _screen_radio(at):
    """Find the Screen radio by its options containing known screen names."""
    for radio in at.sidebar.radio:
        opts = list(radio.options)
        if "Data" in opts and "Train" in opts and "Test & Replay" in opts:
            return radio
    raise AssertionError("Screen radio not found")


def _app_text(at) -> str:
    parts: list[str] = []
    for attr in ("caption", "markdown", "info", "warning", "error", "title", "header"):
        for widget in getattr(at, attr, []):
            parts.append(str(getattr(widget, "value", widget)))
    return "\n".join(parts)


def _fake_processed_bundle(dataset_id, features, units, split):
    return {
        "features": features,
        "units": units,
        "split": split,
        "report": {
            "history_length_physical": {"note": "synthetic fixture, not a real dataset"},
            "dataset_version": "testver",
        },
        "dir": None,
        "fingerprint": {"dataset_version": "testver"},
        "dataset_version": "testver",
    }


def _explorer_harness(
    monkeypatch,
    tmp_path,
    tiny_bearing_tables,
    *,
    run_id="bearings_fly_explorer",
    architecture="fly_connectome_reservoir",
    graph_mode="synthetic_fixture",
    is_synthetic=True,
    n_nodes=8,
    worker_alive=False,
    captured=None,
    parent_graph_mode=None,
    parent_is_synthetic=None,
    disclaimer=None,
):
    from pdm.splits import bearings_split

    _patch_soma_dir(monkeypatch, tmp_path)
    features, units = tiny_bearing_tables
    split = bearings_split(units)
    bundle = _fake_processed_bundle("bearings", features, units, split)
    rdir = tmp_path / run_id
    rdir.mkdir(parents=True, exist_ok=True)
    cdir = rdir / "connectome"
    cdir.mkdir()
    provenance = {
        "graph_mode": graph_mode,
        "is_synthetic": is_synthetic,
        "disclaimer": SYNTHETIC_DISCLAIMER if disclaimer is None else disclaimer,
        "n_nodes": n_nodes,
    }
    if parent_graph_mode is not None:
        provenance["parent_graph_mode"] = parent_graph_mode
    if parent_is_synthetic is not None:
        provenance["parent_is_synthetic"] = parent_is_synthetic
    (cdir / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    fake_row = {
        "dataset_id": "bearings",
        "run_id": run_id,
        "path": str(rdir),
        "has_best": True,
        "has_last": True,
        "status": "completed",
        "architecture": architecture,
        "graph_mode": graph_mode,
        "is_synthetic": is_synthetic,
        "n_nodes": n_nodes,
    }
    if parent_graph_mode is not None:
        fake_row["parent_graph_mode"] = parent_graph_mode
    if parent_is_synthetic is not None:
        fake_row["parent_is_synthetic"] = parent_is_synthetic
    monkeypatch.setattr("pdm.data.prepare.processed_ready", lambda ds: ds == "bearings")
    monkeypatch.setattr("pdm.data.prepare.load_processed", lambda ds: bundle)
    monkeypatch.setattr("pdm.replay.bind_replay_to_run", lambda *args: {**bundle, "current_fingerprint": bundle["fingerprint"]})
    monkeypatch.setattr("pdm.experiments.list_runs", lambda ds=None: [fake_row])
    monkeypatch.setattr("pdm.experiments.run_dir", lambda ds, rid: rdir)
    monkeypatch.setattr("pdm.worker.worker_alive", lambda: worker_alive)
    monkeypatch.setattr("pdm.visualization.simulation_ui.worker_alive", lambda: worker_alive)
    # The operational screen opens a saved model directly, without a Build trace
    # prerequisite. Supply a small real checkpoint boundary for this UI harness.
    (rdir / "best.pt").write_bytes(b"ui fixture checkpoint")
    (cdir / "graph.json").write_text(json.dumps({"node_order": [str(i) for i in range(n_nodes)], "edges": []}))

    def _ui_model(*_args):
        import networkx as nx

        from pdm.models import FlyConnectomeReservoir
        from pdm.preprocessing import Preprocessor

        graph_doc = json.loads((cdir / "graph.json").read_text())
        nodes = graph_doc.get("node_order") or graph_doc.get("nodes") or []
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(zip(nodes, nodes[1:]), weight=1.0)
        model = FlyConnectomeReservoir(graph, input_size=1, seed=4, provenance=provenance)
        prep = Preprocessor(
            feature_names=["horizontal_rms"], log1p_features=[], scaler_mean=[0.0],
            scaler_scale=[1.0], time_scale_s=100.0, fill_values={"horizontal_rms": 0.0},
            dataset_id="bearings",
        )
        return model, prep, {"history_length": 5}

    monkeypatch.setattr("pdm.visualization.simulation_ui._simulation_model", _ui_model)
    if captured is not None:
        monkeypatch.setattr("pdm.cli.spawn_worker", lambda job: captured.update(job) or captured)
    return rdir, split, fake_row


def _open_explorer(at):
    _screen_radio(at).set_value("Neural Activity Explorer")
    at.run()
    assert not at.exception
    return at


def test_explorer_screen_present_in_app():
    """Neural Activity Explorer appears in the explicit 4-way screen radio."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)
    at.run()
    assert not at.exception
    radio = _screen_radio(at)
    opts = list(radio.options)
    assert opts == ["Data", "Train", "Test & Replay", "Neural Activity Explorer"]


def test_explorer_caption_present(monkeypatch, tmp_path, tiny_bearing_tables):
    """Both required captions are shown."""
    from streamlit.testing.v1 import AppTest

    _explorer_harness(monkeypatch, tmp_path, tiny_bearing_tables)
    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)
    at.run()
    assert not at.exception
    _open_explorer(at)
    text = _app_text(at)
    assert EXPLORER_DISCLAIMER in text
    assert SYNTHETIC_DISCLAIMER in text


def test_synthetic_banner_present(monkeypatch, tmp_path, tiny_bearing_tables):
    """Synthetic banner shown when is_synthetic=True."""
    from streamlit.testing.v1 import AppTest

    _explorer_harness(monkeypatch, tmp_path, tiny_bearing_tables, is_synthetic=True)
    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)
    at.run()
    _open_explorer(at)
    warnings = "\n".join(str(w.value) for w in at.warning)
    captions = "\n".join(str(w.value) for w in at.caption)
    assert SYNTHETIC_DISCLAIMER in warnings or SYNTHETIC_DISCLAIMER in captions
    assert SYNTHETIC_DISCLAIMER in _app_text(at)


def test_gru_run_does_not_show_fake_biological_activity(monkeypatch, tmp_path, tiny_bearing_tables):
    """GRU/LSTM runs show the reservoir-required note, not fake fly activity."""
    from streamlit.testing.v1 import AppTest

    inline = {"n": 0}

    def _no_inline(*_a, **_k):
        inline["n"] += 1
        raise AssertionError("GRU runs must not invent reservoir traces")

    _explorer_harness(
        monkeypatch,
        tmp_path,
        tiny_bearing_tables,
        architecture="gru",
        graph_mode="",
        is_synthetic=False,
    )
    monkeypatch.setattr("pdm.visualization.trace.predict_with_trace", _no_inline)
    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)
    at.run()
    _open_explorer(at)
    text = _app_text(at)
    assert EXPLORER_DISCLAIMER in text
    assert RESERVOIR_REQUIRED_MESSAGE in text
    assert not any(b.label == "Build trace" for b in at.button)
    assert not at.exception
    assert RESERVOIR_REQUIRED_MESSAGE in _app_text(at)
    assert inline["n"] == 0


def test_worker_busy_shows_error_not_inline(monkeypatch, tmp_path, tiny_bearing_tables):
    """A busy worker blocks operational inference and spawning another job."""
    from streamlit.testing.v1 import AppTest

    inline = {"n": 0}
    spawned = {"n": 0}

    def _no_inline(*_a, **_k):
        inline["n"] += 1
        raise AssertionError("predict_with_trace must not run while worker is busy")

    def _no_job(*_a, **_k):
        inline["n"] += 1
        raise AssertionError("run_trace_job must not run while worker is busy")

    def _no_spawn(*_a, **_k):
        spawned["n"] += 1
        raise AssertionError("spawn_worker must not run while worker is busy")

    _explorer_harness(monkeypatch, tmp_path, tiny_bearing_tables, worker_alive=True)
    monkeypatch.setattr("pdm.visualization.trace.predict_with_trace", _no_inline)
    monkeypatch.setattr("pdm.visualization.trace.run_trace_job", _no_job)
    monkeypatch.setattr("pdm.cli.spawn_worker", _no_spawn)
    monkeypatch.setattr("pdm.app.spawn_worker", _no_spawn)

    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)
    at.run()
    _open_explorer(at)
    assert not any(b.label == "Build trace" for b in at.button)
    assert not at.exception
    assert "A heavy job is already running. Please wait." in _app_text(at)
    assert inline["n"] == 0
    assert spawned["n"] == 0


def test_no_cdn_in_frontend():
    """No https:// script/module src in frontend HTML/JS, including the component bridge."""
    # Cartoon mapping (fitNodesIntoHull / polarToCns) is grepped in tests/test_explorer_webgl.py.
    html_files = list(FRONTEND.rglob("*.html"))
    js_files = list(FRONTEND.rglob("*.js"))
    assert html_files
    assert js_files
    vendor_bridge = FRONTEND / "vendor" / "streamlit-component-lib.js"
    assert vendor_bridge.is_file()
    for path in html_files + js_files:
        text = path.read_text(encoding="utf-8")
        for host in CDN_HOSTS:
            assert host not in text, f"{path} contains CDN host {host}"
        if "vendor" not in path.parts:
            assert SCRIPT_HTTPS_SRC.search(text) is None, f"{path} has https script src"
            assert MODULE_HTTPS.search(text) is None, f"{path} has https module import"
            assert 'src="https://' not in text
            assert "src='https://" not in text
            assert "src=`https://" not in text
        else:
            assert SCRIPT_HTTPS_SRC.search(text) is None, f"{path} has https script src"
            assert 'src="https://' not in text
            assert "src='https://" not in text
    main_js = (FRONTEND / "main.js").read_text(encoding="utf-8")
    assert "Math.random" not in main_js
    assert "requestAnimationFrame" in main_js
    html = (FRONTEND / "index.html").read_text(encoding="utf-8")
    assert "./vendor/three.min.js" in html
    assert "./vendor/streamlit-component-lib.js" in html
    assert "AdditiveBlending" in main_js or "blending: THREE.AdditiveBlending" in main_js


def test_component_payload_has_states(tiny_bearing_tables):
    """Component receives states array from predict_with_trace output."""
    from pdm.connectome.graph import graph_from_edges
    from pdm.models import FlyConnectomeReservoir
    from pdm.preprocessing import Preprocessor
    from pdm.visualization.trace import predict_with_trace
    from pdm.windows import BEARINGS_RAW_NUMERIC_COLUMNS

    features, _units = tiny_bearing_tables
    names = list(BEARINGS_RAW_NUMERIC_COLUMNS)
    prep = Preprocessor(
        feature_names=names,
        log1p_features=[],
        scaler_mean=[0.0] * len(names),
        scaler_scale=[1.0] * len(names),
        time_scale_s=1.0,
        fill_values={name: 0.0 for name in names},
        dataset_id="bearings",
    )
    n_nodes = 8
    nodes = [str(i) for i in range(n_nodes)]
    edges = [{"src": nodes[i], "dst": nodes[(i + 1) % n_nodes], "weight": 1.0} for i in range(n_nodes)]
    graph = graph_from_edges(edges, nodes)
    model = FlyConnectomeReservoir(
        graph,
        input_size=len(prep.feature_names),
        head="rul",
        seed=0,
        leak=0.2,
        time_scale_s=1.0,
        n_nodes=n_nodes,
    )
    uid = str(features["unit_id"].iloc[0])
    hist = features[features["unit_id"] == uid].sort_values("timestamp_s").iloc[:5].copy()
    hist.attrs["raw_features"] = True
    trace = predict_with_trace(hist, uid, model, prep, history_length=5)
    assert trace["status"] == "predicted"
    assert isinstance(trace["states"], np.ndarray)
    payload = build_explorer_payload(
        nodes=trace["node_order"],
        edges=edges,
        positions={nid: [float(i), 0.0, 0.0] for i, nid in enumerate(trace["node_order"])},
        states=trace["states"],
        frame_map=trace["frame_map"],
        flags={"mode": "Overview"},
        inputs=trace["inputs"],
        predicted_rul_s=trace["predicted_rul_s"],
    )
    assert payload["states"] == np.asarray(trace["states"], dtype=float).tolist()
    assert payload["inputs"] == np.asarray(trace["inputs"], dtype=float).tolist()
    assert len(payload["states"]) == trace["states"].shape[0]
    assert len(payload["states"][0]) == trace["states"].shape[1]
    assert payload["states"][0] != payload["states"][-1] or trace["states"].shape[0] == 1
    assert payload["predicted_rul_s"] == float(trace["predicted_rul_s"])


def test_screen_switch_by_label(monkeypatch, tmp_path):
    """AppTest screen switch uses label not index."""
    from streamlit.testing.v1 import AppTest

    _patch_soma_dir(monkeypatch, tmp_path)
    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)
    at.run()
    assert not at.exception
    _screen_radio(at).set_value("Neural Activity Explorer")
    at.run()
    assert not at.exception
    assert EXPLORER_DISCLAIMER in _app_text(at)
    _screen_radio(at).set_value("Train")
    at.run()
    assert not at.exception
    text = _app_text(at)
    assert "Train" in text or "Training mode" in text or "Prepare data" in text


def test_frontend_vendor_files_present():
    vendor = FRONTEND / "vendor"
    assert (vendor / "three.min.js").is_file()
    assert (vendor / "streamlit-component-lib.js").is_file()
    assert (vendor / "VENDOR.txt").is_file()
    three = (vendor / "three.min.js").read_text(encoding="utf-8", errors="ignore")
    assert "WebGLRenderer" in three
    bridge = (vendor / "streamlit-component-lib.js").read_text(encoding="utf-8")
    assert "setComponentReady" in bridge
    assert "onRender" in bridge
    assert Path(FRONTEND / "index.html").is_file()


def test_explorer_opens_test_controls_without_trace_or_demo(monkeypatch, tmp_path, tiny_bearing_tables):
    """The screen opens the practical test run without historical mode setup."""
    from streamlit.testing.v1 import AppTest

    _explorer_harness(monkeypatch, tmp_path, tiny_bearing_tables)
    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)
    at.run()
    _open_explorer(at)
    assert not at.exception
    labels = {b.label for b in at.button}
    assert {"Start test run", "Next measurement", "Reset test run"} <= labels
    assert "Build trace" not in labels
    assert "Load demo scenario" not in labels
    assert SYNTHETIC_DISCLAIMER in _app_text(at)


def test_explorer_plain_language_copy(monkeypatch, tmp_path, tiny_bearing_tables):
    from streamlit.testing.v1 import AppTest

    _explorer_harness(monkeypatch, tmp_path, tiny_bearing_tables)
    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)
    at.run()
    _open_explorer(at)
    text = _app_text(at)
    assert "remaining useful life" in text.lower()
    assert "computational units" in text.lower()
    assert EXPLORER_DISCLAIMER in text
    assert SYNTHETIC_DISCLAIMER in text
    assert "Choose a model and equipment unit" in text
    assert any(b.label == "Start test run" for b in at.button)


def test_overlay_show_gt_does_not_move_predicted_zone(tiny_bearing_tables):
    """show_gt is evaluator overlay only; predicted zone uses provided predicted_rul_s."""
    from pdm.visualization.overlay import build_work_overlay_figure

    features, units = tiny_bearing_tables
    uid = str(features["unit_id"].iloc[0])
    unit = features[features["unit_id"].astype(str) == uid].copy()
    meta = units[units["unit_id"].astype(str) == uid].iloc[0]
    now = float(unit["timestamp_s"].iloc[12])
    pred = 600.0
    fig_off = build_work_overlay_figure(
        dataset_id="bearings",
        unit_features=unit,
        now_timestamp_s=now,
        predicted_rul_s=pred,
        history_length=5,
        show_gt=False,
        event_time_s=float(meta["event_time_s"]),
        event_observed=True,
    )
    fig_on = build_work_overlay_figure(
        dataset_id="bearings",
        unit_features=unit.iloc[:13],
        now_timestamp_s=now,
        predicted_rul_s=pred,
        history_length=5,
        show_gt=True,
        event_time_s=float(meta["event_time_s"]),
        event_observed=True,
    )
    m0 = dict(fig_off.layout.meta or {})
    m1 = dict(fig_on.layout.meta or {})
    assert m0["predicted_zone_x0"] == m1["predicted_zone_x0"]
    assert m0["predicted_zone_x1"] == m1["predicted_zone_x1"]
    assert m0["predicted_event_x"] == m1["predicted_event_x"]
    assert m0["stored_predicted_rul_s"] == pred
    assert abs(m0["predicted_zone_x1"] - (now / 60.0 + pred / 60.0)) < 1e-9
    names = [getattr(tr, "name", None) for tr in fig_off.data]
    assert "recorded signal (full history)" in names
    assert "seen by the model" in names
    assert "predicted event" not in names
    assert not any(n and "RUL" in str(n) for n in names)


def test_live_writer_reservoir_has_2d_states(monkeypatch, tmp_path, tiny_bearing_tables):
    from pdm.connectome.graph import graph_from_edges
    from pdm.models import FlyConnectomeReservoir
    from pdm.visualization.live import load_live_activity, write_training_live_snapshot
    from pdm.windows import BEARINGS_RAW_NUMERIC_COLUMNS

    monkeypatch.setattr("pdm.visualization.live.worker_dir", lambda: tmp_path)
    features, _units = tiny_bearing_tables
    names = list(BEARINGS_RAW_NUMERIC_COLUMNS)
    n_nodes = 8
    nodes = [str(i) for i in range(n_nodes)]
    edges = [{"src": nodes[i], "dst": nodes[(i + 1) % n_nodes], "weight": 1.0} for i in range(n_nodes)]
    graph = graph_from_edges(edges, nodes)
    model = FlyConnectomeReservoir(
        graph,
        input_size=len(names),
        head="rul",
        seed=0,
        leak=0.2,
        time_scale_s=1.0,
        n_nodes=n_nodes,
    )
    uid = str(features["unit_id"].iloc[0])
    hist = features[features["unit_id"] == uid].sort_values("timestamp_s").iloc[:5]
    window = hist[names].to_numpy(dtype=np.float32)
    stamps = hist["timestamp_s"].to_numpy(dtype=np.float64)
    ok = write_training_live_snapshot(
        model,
        window,
        architecture="fly_connectome_reservoir",
        dataset_id="bearings",
        run_id="live_test",
        epoch=1,
        unit_id=uid,
        node_order=nodes,
        window_timestamps_s=stamps,
    )
    assert ok is True
    live = load_live_activity()
    assert live is not None
    assert live["status"] == "ok"
    assert live["unit_id"] == uid
    assert isinstance(live["states"], np.ndarray)
    assert live["states"].ndim == 2
    assert live["states"].shape[0] == 5
    assert live["states"].shape[1] == n_nodes
    assert [f["timestamp_s"] for f in live["frame_map"]] == [float(v) for v in stamps.tolist()]
    assert "hull_mode" not in live
    assert "context_positions" not in live
    meta = json.loads((tmp_path / "live_activity.json").read_text(encoding="utf-8"))
    assert "hull_mode" not in meta
    assert "context_positions" not in meta
    assert "n_nodes" not in meta
    assert meta["n_nodes_display"] == n_nodes


def test_live_writer_gru_has_no_fake_states(monkeypatch, tmp_path):
    import torch
    from torch import nn

    from pdm.visualization.live import load_live_activity, write_training_live_snapshot

    monkeypatch.setattr("pdm.visualization.live.worker_dir", lambda: tmp_path)

    class DummyGRU(nn.Module):
        architecture = "gru"

        def forward_states(self, x):
            raise AssertionError("GRU path must not invent reservoir states")

    model = DummyGRU()
    ok = write_training_live_snapshot(
        model,
        torch.zeros(4, 3),
        architecture="gru",
        dataset_id="bearings",
        run_id="gru_live",
        epoch=1,
    )
    assert ok is False
    live = load_live_activity()
    assert live is not None
    assert live["status"] == "not_reservoir"
    assert live.get("states") is None
    assert not (tmp_path / "live_activity.npz").exists()


def test_overlay_clock_ignores_live_fake_timestamps():
    from pdm.visualization.explorer import explorer_overlay_clock

    trace = {
        "frame_map": [
            {"timestamp_s": 1740.0, "window_end_timestamp_s": 1800.0},
            {"timestamp_s": 1800.0, "window_end_timestamp_s": 1800.0},
        ],
        "predicted_rul_s": 900.0,
    }
    live = {
        "phase": "training",
        "predicted_rul_s": 12.5,
        "frame_map": [{"timestamp_s": float(i), "window_end_timestamp_s": 4.0} for i in range(5)],
    }
    now = explorer_overlay_clock(trace, mode="Overview", live=live, train_live=True)
    assert now == 1800.0
    replay_now = explorer_overlay_clock(trace, mode="Equipment replay", replay_step=0, live=live)
    assert replay_now == 1740.0


def test_train_live_gate_requires_active_train_job():
    from pdm.visualization.explorer import is_active_train_live, selected_trace_predicted_rul

    live = {
        "phase": "training",
        "status": "ok",
        "states": [[0.2]],
        "predicted_rul_s": 12.5,
    }
    assert is_active_train_live(worker_alive=False, worker_kind="train", live=live) is False
    assert is_active_train_live(worker_alive=True, worker_kind="trace", live=live) is False
    assert is_active_train_live(worker_alive=True, worker_kind="evaluate", live=live) is False
    assert is_active_train_live(worker_alive=True, worker_kind="train", live=live) is True
    trace = {"predicted_rul_s": 900.0}
    assert selected_trace_predicted_rul(trace) == 900.0
    assert selected_trace_predicted_rul(None) is None


def test_idle_live_json_does_not_set_selected_rul(monkeypatch, tmp_path, tiny_bearing_tables):
    """Leftover phase=training file must not become selected-unit predicted RUL."""
    from streamlit.testing.v1 import AppTest

    _explorer_harness(monkeypatch, tmp_path, tiny_bearing_tables, worker_alive=False)
    leftover = {
        "phase": "training",
        "status": "ok",
        "architecture": "fly_connectome_reservoir",
        "states": [[0.4, 0.5]],
        "predicted_rul_s": 12.5,
        "unit_id": "TRAIN_PROBE",
        "dataset_id": "filters",
        "run_id": "other_run",
        "frame_map": [{"timestamp_s": 0.0, "window_end_timestamp_s": 4.0}],
    }
    monkeypatch.setattr("pdm.visualization.live.load_live_activity", leftover_loader := (lambda: leftover))
    monkeypatch.setattr("pdm.app.load_live_activity", leftover_loader)
    monkeypatch.setattr("pdm.worker.read_status", lambda: {"status": "completed", "kind": "train"})
    monkeypatch.setattr("pdm.app.read_status", lambda: {"status": "completed", "kind": "train"})
    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)
    at.run()
    _open_explorer(at)
    text = _app_text(at)
    assert "Live training — train-split window only" not in text
    assert "TRAIN_PROBE" not in text
    metrics = "\n".join(str(getattr(m, "value", m)) for m in at.metric)
    assert "12.5" not in metrics


def test_operational_screen_omits_architecture_comparison(monkeypatch, tmp_path, tiny_bearing_tables):
    from streamlit.testing.v1 import AppTest

    _explorer_harness(monkeypatch, tmp_path, tiny_bearing_tables)
    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)
    at.run()
    _open_explorer(at)
    md = [str(getattr(w, "value", w)) for w in at.markdown]
    exact = [v for v in md if str(v).replace("*", "").strip() == "Architecture comparison"]
    assert len(exact) == 0


def test_unit_window_dataset_probe_is_train_unit(tiny_bearing_tables):
    """Live probe comes from UnitWindowDataset built on split['train'] units only."""
    from pdm.splits import bearings_split
    from pdm.train import UnitWindowDataset
    from pdm.windows import BEARINGS_RAW_NUMERIC_COLUMNS, build_windows

    features, units = tiny_bearing_tables
    split = bearings_split(units)
    train_ids = {str(u) for u in split["train"]}
    held_out = {str(u) for u in list(split["validation"]) + list(split["test"])}
    train_feat = features[features["unit_id"].astype(str).isin(train_ids)]
    train_units = units[units["unit_id"].astype(str).isin(train_ids)]
    windows = build_windows(train_feat, train_units, history_length=5, dataset_id="bearings")
    ds = UnitWindowDataset(
        train_feat,
        windows,
        list(BEARINGS_RAW_NUMERIC_COLUMNS),
        "bearings",
        1.0,
    )
    assert len(ds)
    probe = ds[0]
    assert str(probe["unit_id"]) in train_ids
    assert str(probe["unit_id"]) not in held_out
    assert len(probe["timestamps_s"]) == int(probe["x"].shape[0])


def _patch_soma_dir(monkeypatch, tmp_path):
    monkeypatch.setattr("pdm.connectome.anatomy.default_soma_dir", lambda: tmp_path)
    clear_soma_table_cache()
    return tmp_path


def _write_feather(path, frame):
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(frame).to_feather(path)
    return path


def test_synthetic_somas_fixture_labeled_string_xyz():
    from pdm.connectome.anatomy import load_synthetic_somas
    from pdm.connectome.provenance import GRAPH_MODE_REAL

    table = load_synthetic_somas()
    assert table.is_synthetic is True
    assert SYNTHETIC_DISCLAIMER in table.provenance["disclaimer"]
    assert table.n_points == len(table.positions) <= 64
    assert table.n_points > 0
    assert table.provenance["n_nodes"] == 0
    assert table.provenance["n_points"] == table.n_points
    assert table.provenance["graph_mode"] != GRAPH_MODE_REAL
    for nid, xyz in table.positions.items():
        assert isinstance(nid, str)
        assert len(xyz) == 3
        assert all(isinstance(v, float) for v in xyz)


def test_soma_allowlist_miss_empty_unavailable(monkeypatch, tmp_path):
    from pdm.connectome.anatomy import SOMA_ALLOWLIST, load_soma_table
    from pdm.connectome.provenance import GRAPH_MODE_REAL
    from pdm.connectome.sources import MALEMCNS_FILENAME

    _patch_soma_dir(monkeypatch, tmp_path)
    (tmp_path / MALEMCNS_FILENAME).write_bytes(b"not-a-soma")
    (tmp_path / "syn-points-dump.feather").write_bytes(b"not-a-soma")
    (tmp_path / "random.feather").write_bytes(b"not-a-soma")
    table = load_soma_table()
    assert table.positions == {}
    assert table.n_points == 0
    assert table.is_synthetic is False
    assert table.provenance["source"] == "unavailable"
    assert table.provenance["graph_mode"] == "unavailable"
    assert table.provenance["graph_mode"] != GRAPH_MODE_REAL
    assert table.provenance["n_points"] == 0
    assert table.provenance["n_nodes"] == 0
    assert SOMA_ALLOWLIST[0] == "body-annotations-male-cns-v1.0-minconf-0.5.feather"


def test_soma_allowlisted_bodyid_xyz_loads(monkeypatch, tmp_path):
    from pdm.connectome.anatomy import SOMA_ALLOWLIST, load_soma_table
    from pdm.connectome.provenance import GRAPH_MODE_REAL

    _patch_soma_dir(monkeypatch, tmp_path)
    loc = tmp_path / SOMA_ALLOWLIST[0]
    _write_feather(
        loc,
        {
            "bodyId": [1001, 1002, 1003],
            "x": [1.0, 2.0, 3.0],
            "y": [4.0, 5.0, 6.0],
            "z": [7.0, 8.0, 9.0],
        },
    )
    table = load_soma_table()
    assert table.is_synthetic is False
    assert table.provenance["graph_mode"] == "viz_soma_xyz"
    assert table.provenance["graph_mode"] != GRAPH_MODE_REAL
    assert table.n_points == 3
    assert table.provenance["n_points"] == 3
    assert table.provenance["n_nodes"] == 0
    assert table.provenance["role"] == "viz_soma_xyz"
    assert table.provenance["n_model_unaffected"] is True
    assert table.provenance["license"] == "CC-BY"
    assert table.provenance["file_hash"]
    assert set(table.positions) == {"1001", "1002", "1003"}
    assert table.positions["1001"] == [1.0, 4.0, 7.0]


def test_soma_explicit_unknown_columns_raise(monkeypatch, tmp_path):
    import pytest

    from pdm.connectome.anatomy import load_soma_table

    _patch_soma_dir(monkeypatch, tmp_path)
    loc = tmp_path / "junk-annotations.feather"
    _write_feather(loc, {"foo": [1], "bar": [2]})
    with pytest.raises(ValueError, match="(?i)columns found"):
        load_soma_table(loc)


def test_soma_explicit_weights_filename_raises(monkeypatch, tmp_path):
    import pytest

    from pdm.connectome.anatomy import load_soma_table
    from pdm.connectome.sources import MALEMCNS_FILENAME

    _patch_soma_dir(monkeypatch, tmp_path)
    loc = tmp_path / MALEMCNS_FILENAME
    _write_feather(
        loc,
        {
            "body_pre": [1],
            "body_post": [2],
            "weight": [3],
        },
    )
    with pytest.raises(ValueError):
        load_soma_table(loc)


def test_soma_syn_points_skipped_by_name_and_explicit_raises(monkeypatch, tmp_path):
    import pytest

    from pdm.connectome.anatomy import SOMA_ALLOWLIST, load_soma_table

    _patch_soma_dir(monkeypatch, tmp_path)
    syn_name = "syn-points-male-cns.feather"
    monkeypatch.setattr(
        "pdm.connectome.anatomy.SOMA_ALLOWLIST",
        (syn_name, SOMA_ALLOWLIST[0]),
    )
    syn = tmp_path / syn_name
    _write_feather(
        syn,
        {
            "bodyId": [9],
            "x": [1.0],
            "y": [2.0],
            "z": [3.0],
        },
    )
    empty = load_soma_table()
    assert empty.positions == {}
    assert empty.provenance["graph_mode"] == "unavailable"

    loc = tmp_path / SOMA_ALLOWLIST[0]
    _write_feather(
        loc,
        {
            "bodyId": [42],
            "x": [4.0],
            "y": [5.0],
            "z": [6.0],
        },
    )
    hit = load_soma_table()
    assert hit.positions == {"42": [4.0, 5.0, 6.0]}
    assert hit.n_points == 1
    assert "9" not in hit.positions

    with pytest.raises(ValueError):
        load_soma_table(syn)


def test_soma_explicit_x_pre_table_raises(monkeypatch, tmp_path):
    import pytest

    from pdm.connectome.anatomy import load_soma_table

    _patch_soma_dir(monkeypatch, tmp_path)
    loc = tmp_path / "pre-xyz.feather"
    _write_feather(
        loc,
        {
            "bodyId": [1],
            "x_pre": [0.0],
            "y_pre": [0.0],
            "z_pre": [0.0],
        },
    )
    with pytest.raises(ValueError, match="(?i)columns found"):
        load_soma_table(loc)


def test_soma_load_twice_deterministic(monkeypatch, tmp_path):
    from pdm.connectome.anatomy import SOMA_ALLOWLIST, load_soma_table

    _patch_soma_dir(monkeypatch, tmp_path)
    loc = tmp_path / SOMA_ALLOWLIST[0]
    _write_feather(
        loc,
        {
            "bodyId": [7, 3, 9],
            "somaLocation": [[1.5, 2.5, 3.5], [0.0, 1.0, 2.0], [9.0, 8.0, 7.0]],
        },
    )
    first = load_soma_table()
    second = load_soma_table(loc)
    assert first.positions == second.positions == {
        "7": [1.5, 2.5, 3.5],
        "3": [0.0, 1.0, 2.0],
        "9": [9.0, 8.0, 7.0],
    }
    assert first.n_points == second.n_points == 3
    assert first.provenance["graph_mode"] == second.provenance["graph_mode"] == "viz_soma_xyz"


def _in_memory_somas(positions: dict, *, is_synthetic: bool = False):
    from pdm.connectome.anatomy import SomaTable

    pos = {
        str(k): [float(v[0]), float(v[1]), float(v[2])]
        for k, v in positions.items()
    }
    return SomaTable(
        positions=pos,
        provenance={
            "graph_mode": "viz_soma_xyz",
            "n_nodes": 0,
            "n_points": len(pos),
            "is_synthetic": is_synthetic,
        },
        is_synthetic=is_synthetic,
        n_points=len(pos),
    )


def _write_run_connectome(rdir, *, nodes, layout, provenance, edges=None):
    cdir = Path(rdir) / "connectome"
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "graph.json").write_text(
        json.dumps(
            {
                "nodes": list(nodes),
                "node_order": list(nodes),
                "edges": list(edges or []),
            }
        ),
        encoding="utf-8",
    )
    (cdir / "layout.json").write_text(json.dumps(layout), encoding="utf-8")
    (cdir / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    return cdir


def test_synthetic_anatomy_scene_is_schematic_empty_context():
    from pdm.visualization.explorer import build_explorer_payload
    from pdm.visualization.scene import build_anatomy_scene

    nodes = [str(i) for i in range(8)]
    soma = _in_memory_somas({nid: [float(i), 0.0, 0.0] for i, nid in enumerate(nodes + ["99", "100"])})
    scene = {
        "nodes": nodes,
        "positions": {nid: [float(i), 1.0, 0.0] for i, nid in enumerate(nodes)},
        "graph_mode": "synthetic_fixture",
        "is_synthetic": True,
    }
    anatomy = build_anatomy_scene(scene=scene, soma=soma, n_model=8, graph_mode="synthetic_fixture", is_synthetic=True)
    assert anatomy["flags"]["hull_mode"] == "schematic_cns"
    assert anatomy["context_positions"] == []
    assert anatomy["hull_polyline"] == []
    assert anatomy["flags"]["n_viz"] == 0
    assert anatomy["flags"]["context_n"] == 0
    assert anatomy["flags"]["anatomy_missing"] is False
    payload = build_explorer_payload(
        nodes=nodes,
        edges=[],
        positions=scene["positions"],
        states=np.zeros((2, 8)),
        frame_map=[],
        flags={"mode": "Overview", "graph_mode": "synthetic_fixture", "is_synthetic": True},
        anatomy=anatomy,
    )
    assert payload["context_positions"] == []
    assert payload["flags"]["hull_mode"] == "schematic_cns"
    json.dumps(payload)


def test_fake_somas_payload_n_model_not_display_cap():
    from pdm.visualization.explorer import build_explorer_payload, subset_scene
    from pdm.visualization.scene import build_anatomy_scene

    nodes = [str(100 + i) for i in range(8)]
    extra = {str(200 + i): [float(i), float(i) * 0.5, 1.0] for i in range(12)}
    soma_pos = {nid: [float(i) * 10.0, float(i), 2.0] for i, nid in enumerate(nodes)}
    soma_pos.update(extra)
    soma = _in_memory_somas(soma_pos)
    spring = {nid: [0.0, 0.0, float(i)] for i, nid in enumerate(nodes)}
    scene = {
        "nodes": nodes,
        "positions": spring,
        "graph_mode": "real_connectome",
        "is_synthetic": False,
        "context_positions": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        "hull_polyline": [[0.0, 0.0, 0.0]],
        "flags": {"n_model": 8},
    }
    display = subset_scene(scene, nodes[:4])
    assert display["context_positions"] == scene["context_positions"]
    assert len(display["nodes"]) == 4
    anatomy = build_anatomy_scene(
        scene=display,
        soma=soma,
        n_model=8,
        graph_mode="real_connectome",
        is_synthetic=False,
        layout={"positions": spring, "node_order": nodes},
        n_nodes_full=8,
    )
    states = np.arange(8, dtype=float).reshape(2, 4)
    payload = build_explorer_payload(
        nodes=display["nodes"],
        edges=[],
        positions=display["positions"],
        states=states,
        frame_map=[],
        flags={"mode": "Overview", "graph_mode": "real_connectome", "is_synthetic": False, "downsampled": True},
        anatomy=anatomy,
        n_model=8,
    )
    flags = payload["flags"]
    assert flags["hull_mode"] == "malecns_anatomy"
    assert payload["context_positions"]
    assert flags["n_viz"] > flags["n_model"]
    assert flags["n_model"] == 8
    assert flags["n_nodes_display"] == 4
    assert len(payload["states"][0]) == 4
    assert len(payload["states"][0]) == flags["n_nodes_display"]
    assert flags["n_viz"] == flags["context_n"] == len(payload["context_positions"])
    assert flags["downsampled"] is True
    assert flags["context_downsampled"] is False
    json.dumps(payload)


def test_context_downsample_numeric_id_stride():
    from pdm.visualization.scene import (
        SOMA_CONTEXT_CAPTION,
        build_anatomy_scene,
        downsample_context_ids,
    )

    ids = [str(i) for i in range(100)]
    a, down_a = downsample_context_ids(ids, 10)
    b, down_b = downsample_context_ids(list(reversed(ids)), 10)
    assert down_a is True and down_b is True
    assert a == b
    assert len(a) == 10
    assert a == [ids[i] for i in np.unique(np.linspace(0, 99, num=10, dtype=np.int64)).tolist()]

    soma_pos = {str(i): [float(i), float(i % 7), 1.0] for i in range(100)}
    soma = _in_memory_somas(soma_pos)
    nodes = ["0", "1", "2"]
    anatomy = build_anatomy_scene(
        scene={
            "nodes": nodes,
            "positions": {n: [0.0, 0.0, 0.0] for n in nodes},
            "graph_mode": "real_connectome",
            "is_synthetic": False,
        },
        soma=soma,
        n_model=3,
        graph_mode="real_connectome",
        is_synthetic=False,
        context_cap=10,
        layout={"positions": {n: [9.0, 9.0, 9.0] for n in nodes}},
    )
    assert anatomy["flags"]["context_downsampled"] is True
    assert anatomy["flags"]["n_viz"] == 10
    assert len(anatomy["context_positions"]) == 10
    assert anatomy["flags"]["context_caption"] == SOMA_CONTEXT_CAPTION
    assert anatomy["flags"]["hull_mode"] == "malecns_anatomy"


def test_real_connectome_missing_soma_anatomy_missing():
    from pdm.visualization.explorer import build_explorer_payload
    from pdm.visualization.scene import build_anatomy_scene

    nodes = ["10", "20", "30"]
    anatomy = build_anatomy_scene(
        scene={
            "nodes": nodes,
            "positions": {n: [float(i), 0.0, 0.0] for i, n in enumerate(nodes)},
            "graph_mode": "real_connectome",
            "is_synthetic": False,
        },
        soma=None,
        n_model=3,
        graph_mode="real_connectome",
        is_synthetic=False,
    )
    assert anatomy["flags"]["hull_mode"] == "schematic_cns"
    assert anatomy["flags"]["anatomy_missing"] is True
    assert anatomy["context_positions"] == []
    payload = build_explorer_payload(
        nodes=nodes,
        edges=[],
        positions=anatomy["positions"],
        states=np.zeros((1, 3)),
        frame_map=[],
        flags={"graph_mode": "real_connectome"},
        anatomy=anatomy,
    )
    assert payload["flags"]["anatomy_missing"] is True
    assert payload["context_positions"] == []


def test_3d_spring_layout_not_anatomical_soma_wins():
    from pdm.connectome.graph import graph_from_edges
    from pdm.connectome.layout import is_anatomical_positions, layout_positions
    from pdm.visualization.scene import build_anatomy_scene

    nodes = ["10", "20", "30"]
    edges = [{"src": "10", "dst": "20", "weight": 1.0}, {"src": "20", "dst": "30", "weight": 1.0}]
    graph = graph_from_edges(edges, nodes)
    spring3 = layout_positions(graph, dim=3, seed=42)
    zs = [float(spring3[n][2]) if len(spring3[n]) > 2 else 0.0 for n in nodes]
    layout = {"positions": spring3, "node_order": nodes}
    assert is_anatomical_positions(layout) is False
    tagged = {"method": "spring", "positions": {n: [float(i), 0.0, float(i) + 1.0] for i, n in enumerate(nodes)}}
    assert max(tagged["positions"][n][2] for n in nodes) - min(tagged["positions"][n][2] for n in nodes) > 0
    assert is_anatomical_positions(tagged) is False
    if max(zs) - min(zs) == 0:
        layout = tagged

    soma = _in_memory_somas(
        {
            "10": [0.0, 0.0, 0.0],
            "20": [10.0, 0.0, 0.0],
            "30": [0.0, 10.0, 0.0],
            "40": [0.0, 0.0, 10.0],
            "50": [5.0, 5.0, 5.0],
        }
    )
    anatomy = build_anatomy_scene(
        scene={
            "nodes": nodes,
            "positions": spring3,
            "graph_mode": "real_connectome",
            "is_synthetic": False,
        },
        soma=soma,
        n_model=3,
        graph_mode="real_connectome",
        is_synthetic=False,
        layout=layout,
        graph=graph,
    )
    assert is_anatomical_positions(layout, graph=graph) is False
    assert anatomy["flags"]["hull_mode"] == "malecns_anatomy"
    p10 = anatomy["positions"]["10"]
    p20 = anatomy["positions"]["20"]
    dx = abs(p20[0] - p10[0])
    dz = abs(p20[2] - p10[2])
    assert dx > dz
    assert anatomy["flags"]["n_viz"] > anatomy["flags"]["n_model"]
    assert anatomy["hull_polyline"]


def test_planar_spring_layout_not_anatomical():
    from pdm.connectome.graph import graph_from_edges
    from pdm.connectome.layout import is_anatomical_positions, layout_positions

    nodes = ["1", "2", "3", "4"]
    edges = [{"src": nodes[i], "dst": nodes[(i + 1) % 4], "weight": 1.0} for i in range(4)]
    graph = graph_from_edges(edges, nodes)
    pos = layout_positions(graph, dim=2, seed=42)
    layout = {"positions": pos, "node_order": nodes}
    assert is_anatomical_positions(layout) is False
    assert is_anatomical_positions({"method": "spring", "positions": pos}) is False
    assert is_anatomical_positions(graph=graph) is False


def test_graph_xyz_attrs_are_anatomical():
    from pdm.connectome.graph import graph_from_edges
    from pdm.connectome.layout import is_anatomical_positions

    nodes = ["1", "2"]
    graph = graph_from_edges([{"src": "1", "dst": "2", "weight": 1.0}], nodes)
    graph.nodes["1"]["x"] = 1.0
    graph.nodes["1"]["y"] = 2.0
    graph.nodes["1"]["z"] = 3.0
    graph.nodes["2"]["x"] = 4.0
    graph.nodes["2"]["y"] = 5.0
    graph.nodes["2"]["z"] = 6.0
    assert is_anatomical_positions({"positions": {"1": [0.0, 0.0, 9.0]}}, graph=graph) is True


def test_unmatched_reservoir_keeps_nodes_omits_centroid():
    from pdm.visualization.explorer import build_explorer_payload
    from pdm.visualization.scene import build_anatomy_scene

    nodes = ["10", "99"]
    soma = _in_memory_somas({"10": [2.0, 4.0, 6.0], "20": [8.0, 0.0, 0.0], "30": [0.0, 8.0, 0.0]})
    anatomy = build_anatomy_scene(
        scene={"nodes": nodes, "positions": {"10": [9.0, 9.0, 9.0], "99": [8.0, 8.0, 8.0]}, "graph_mode": "real_connectome"},
        soma=soma,
        n_model=2,
        graph_mode="real_connectome",
        is_synthetic=False,
        layout={"positions": {"10": [9.0, 9.0, 9.0], "99": [8.0, 8.0, 8.0]}},
    )
    assert anatomy["flags"]["n_unmatched_reservoir"] == 1
    assert "10" in anatomy["positions"]
    assert "99" not in anatomy["positions"]
    states = [[0.1, 0.2], [0.3, 0.4]]
    payload = build_explorer_payload(
        nodes=nodes,
        edges=[],
        positions=anatomy["positions"],
        states=states,
        frame_map=[],
        flags={"graph_mode": "real_connectome"},
        anatomy=anatomy,
    )
    assert payload["nodes"] == nodes
    assert "99" not in payload["positions"]
    assert len(payload["states"][0]) == 2
    assert len(payload["states"][0]) != payload["flags"]["n_viz"]


def test_rewire_synthetic_parent_stays_schematic():
    from pdm.visualization.scene import build_anatomy_scene

    nodes = ["0", "1", "2"]
    soma = _in_memory_somas({n: [float(i), 0.0, 0.0] for i, n in enumerate(nodes + ["9"])})
    anatomy = build_anatomy_scene(
        scene={"nodes": nodes, "positions": {n: [0.0, 0.0, 0.0] for n in nodes}},
        soma=soma,
        n_model=3,
        graph_mode="random_rewire",
        is_synthetic=False,
        provenance={"graph_mode": "random_rewire", "is_synthetic": True, "parent_graph_mode": "synthetic_fixture"},
    )
    assert anatomy["flags"]["hull_mode"] == "schematic_cns"
    assert anatomy["context_positions"] == []


def test_n_model_from_run_provenance_not_graph_cap(tmp_path):
    from pdm.visualization.explorer import build_explorer_payload, load_scene_from_run
    from pdm.visualization.scene import build_anatomy_scene

    nodes = [str(1000 + i) for i in range(8)]
    layout = {"positions": {n: [float(i), 0.0, float(i)] for i, n in enumerate(nodes)}, "node_order": nodes}
    _write_run_connectome(
        tmp_path,
        nodes=nodes,
        layout=layout,
        provenance={
            "graph_mode": "real_connectome",
            "is_synthetic": False,
            "n_nodes": 8,
        },
    )
    scene = load_scene_from_run(tmp_path)
    assert scene["nodes"] == nodes
    assert scene["positions"]["1000"] == [0.0, 0.0, 0.0]
    soma = _in_memory_somas(
        {n: [float(i), float(i) * 2.0, 1.0] for i, n in enumerate(nodes)}
        | {str(9000 + i): [float(i), 3.0, 0.0] for i in range(6)}
    )
    display_nodes = nodes[:4]
    anatomy = build_anatomy_scene(
        scene={**scene, "nodes": display_nodes},
        run_dir=tmp_path,
        soma=soma,
        graph_mode="real_connectome",
        is_synthetic=False,
        layout=layout,
    )
    payload = build_explorer_payload(
        nodes=display_nodes,
        edges=[],
        positions=scene["positions"],
        states=np.zeros((3, 4)),
        frame_map=[],
        flags={"downsampled": True, "graph_mode": "real_connectome"},
        anatomy=anatomy,
    )
    assert payload["flags"]["n_model"] == 8
    assert payload["flags"]["n_nodes_display"] == 4
    assert len(payload["states"][0]) == 4
    assert payload["flags"]["n_viz"] > 8
    assert payload["flags"]["hull_mode"] == "malecns_anatomy"


def test_payload_n_model_from_run_not_live_cap(tmp_path):
    from pdm.visualization.explorer import build_explorer_payload
    from pdm.visualization.live import LIVE_NODE_CAP

    nodes = [str(1000 + i) for i in range(8)]
    layout = {"positions": {n: [float(i), 0.0, float(i)] for i, n in enumerate(nodes)}, "node_order": nodes}
    _write_run_connectome(
        tmp_path,
        nodes=nodes,
        layout=layout,
        provenance={
            "graph_mode": "real_connectome",
            "is_synthetic": False,
            "n_nodes": 8,
        },
    )
    soma = _in_memory_somas(
        {n: [float(i), float(i) * 2.0, 1.0] for i, n in enumerate(nodes)}
        | {str(9000 + i): [float(i), 3.0, 0.0] for i in range(4)}
    )
    display = [str(i) for i in range(LIVE_NODE_CAP)]
    payload = build_explorer_payload(
        nodes=display,
        edges=[],
        positions={nid: [0.0, 0.0, 0.0] for nid in display},
        states=np.zeros((2, LIVE_NODE_CAP)),
        frame_map=[],
        flags={"downsampled": True, "graph_mode": "real_connectome"},
        soma=soma,
        run_dir=tmp_path,
        n_model=None,
    )
    assert payload["flags"]["n_model"] == 8
    assert payload["flags"]["n_model"] != LIVE_NODE_CAP
    assert payload["flags"]["n_model"] != len(display)
    assert payload["flags"]["n_nodes_display"] == LIVE_NODE_CAP
    assert len(payload["states"][0]) == LIVE_NODE_CAP


def test_payload_omits_n_model_when_width_unknown():
    from pdm.visualization.explorer import build_explorer_payload
    from pdm.visualization.live import LIVE_NODE_CAP
    from pdm.visualization.scene import resolve_n_model

    display = [str(i) for i in range(LIVE_NODE_CAP)]
    payload = build_explorer_payload(
        nodes=display,
        edges=[],
        positions={nid: [0.0, 0.0, 0.0] for nid in display},
        states=np.zeros((1, LIVE_NODE_CAP)),
        frame_map=[],
        flags={"downsampled": True},
        n_model=None,
    )
    assert payload["flags"].get("n_model") is None
    assert "n_model" not in payload["flags"] or payload["flags"]["n_model"] is None
    assert payload["flags"]["n_nodes_display"] == LIVE_NODE_CAP
    assert resolve_n_model() is None


def test_empty_soma_table_real_is_anatomy_missing():
    from pdm.connectome.anatomy import SomaTable
    from pdm.visualization.scene import build_anatomy_scene

    empty = SomaTable(positions={}, provenance={"graph_mode": "unavailable", "n_nodes": 0, "n_points": 0}, is_synthetic=False, n_points=0)
    anatomy = build_anatomy_scene(
        scene={"nodes": ["1", "2"], "positions": {"1": [0.0, 0.0, 0.0], "2": [1.0, 0.0, 0.0]}, "graph_mode": "real_connectome"},
        soma=empty,
        n_model=2,
        graph_mode="real_connectome",
        is_synthetic=False,
    )
    assert anatomy["flags"]["anatomy_missing"] is True
    assert anatomy["flags"]["hull_mode"] == "schematic_cns"
    assert anatomy["context_positions"] == []


def test_soma_downsample_caption_matches_scene():
    from pdm.visualization.scene import SOMA_CONTEXT_CAPTION

    assert SOMA_DOWNSAMPLE_CAPTION == SOMA_CONTEXT_CAPTION
    assert SOMA_DOWNSAMPLE_CAPTION == "Soma context downsampled for display"


def test_cached_soma_table_not_reread_until_cleared(monkeypatch, tmp_path):
    from pdm.connectome.anatomy import SOMA_ALLOWLIST, load_soma_table
    from pdm.visualization.explorer import cached_soma_table

    _patch_soma_dir(monkeypatch, tmp_path)
    _write_feather(
        tmp_path / SOMA_ALLOWLIST[0],
        {"bodyId": [1], "x": [1.0], "y": [2.0], "z": [3.0]},
    )
    calls = {"n": 0}
    real = load_soma_table

    def counted(path=None):
        calls["n"] += 1
        return real(path)

    monkeypatch.setattr("pdm.connectome.anatomy.load_soma_table", counted)
    clear_soma_table_cache()
    first = cached_soma_table()
    second = cached_soma_table()
    assert calls["n"] == 1
    assert first.positions == second.positions == {"1": [1.0, 2.0, 3.0]}
    clear_soma_table_cache()
    cached_soma_table()
    assert calls["n"] == 2


def test_ui_helper_overwrites_leftover_hull_mode(monkeypatch, tmp_path):
    from pdm.connectome.anatomy import SOMA_ALLOWLIST
    from pdm.visualization.explorer import LIVE_CONTEXT_CAP, build_ui_explorer_payload

    _patch_soma_dir(monkeypatch, tmp_path)
    nodes = ["10", "20", "30"]
    extras = [str(100 + i) for i in range(8)]
    ids = [int(n) for n in nodes + extras]
    _write_feather(
        tmp_path / SOMA_ALLOWLIST[0],
        {
            "bodyId": ids,
            "x": [float(i) for i in range(len(ids))],
            "y": [0.5 * i for i in range(len(ids))],
            "z": [1.0] * len(ids),
        },
    )
    _write_run_connectome(
        tmp_path,
        nodes=nodes,
        layout={"positions": {n: [9.0, 9.0, 9.0] for n in nodes}, "node_order": nodes},
        provenance={"graph_mode": "real_connectome", "is_synthetic": False, "n_nodes": 3},
    )
    payload, err = build_ui_explorer_payload(
        nodes=nodes,
        edges=[],
        positions={n: [0.0, 0.0, 0.0] for n in nodes},
        states=np.zeros((2, 3)),
        frame_map=[],
        flags={
            "hull_mode": "schematic_cns",
            "graph_mode": "real_connectome",
            "is_synthetic": False,
        },
        run_dir=tmp_path,
        n_model=3,
        join_soma=True,
        context_cap=LIVE_CONTEXT_CAP,
    )
    assert err is None
    assert payload["flags"]["hull_mode"] == "malecns_anatomy"
    assert payload["context_positions"]
    assert payload["flags"]["n_model"] == 3
    assert payload["flags"]["n_nodes_display"] == 3
    assert len(payload["states"][0]) == 3
    assert payload["flags"]["n_viz"] == len(payload["context_positions"])
    caps = explorer_anatomy_captions(payload, is_synthetic=False, schema_error=None)
    assert any("N_viz=" in line and "N_model=" in line for line in caps)


def test_ui_helper_schema_error_forces_schematic(monkeypatch, tmp_path):
    from pdm.connectome.anatomy import SOMA_ALLOWLIST
    from pdm.visualization.explorer import build_ui_explorer_payload

    _patch_soma_dir(monkeypatch, tmp_path)
    _write_feather(tmp_path / SOMA_ALLOWLIST[0], {"foo": [1], "bar": [2]})
    payload, err = build_ui_explorer_payload(
        nodes=["1", "2"],
        edges=[],
        positions={"1": [0.0, 0.0, 0.0], "2": [1.0, 0.0, 0.0]},
        states=np.zeros((1, 2)),
        frame_map=[],
        flags={"graph_mode": "real_connectome", "hull_mode": "malecns_anatomy"},
        join_soma=True,
        n_model=2,
    )
    assert err is not None
    assert "Columns found" in err
    assert payload["flags"]["hull_mode"] == "schematic_cns"
    assert payload["flags"]["anatomy_missing"] is True
    assert payload["context_positions"] == []
    caps = explorer_anatomy_captions(payload, is_synthetic=False, schema_error=err)
    assert any("Columns found" in line for line in caps)
    assert ANATOMY_MISSING_CAPTION in caps


def test_ui_helper_soma_oserror_forces_schematic(monkeypatch):
    from pdm.visualization.explorer import build_ui_explorer_payload

    def _boom(_path=None):
        raise OSError("arrow failed")

    monkeypatch.setattr("pdm.visualization.explorer.cached_soma_table", _boom)
    payload, err = build_ui_explorer_payload(
        nodes=["1", "2"],
        edges=[],
        positions={"1": [0.0, 0.0, 0.0], "2": [1.0, 0.0, 0.0]},
        states=np.zeros((1, 2)),
        frame_map=[],
        flags={"graph_mode": "real_connectome", "hull_mode": "malecns_anatomy"},
        join_soma=True,
        n_model=2,
    )
    assert err is not None
    assert "arrow failed" in err
    assert payload["flags"]["hull_mode"] == "schematic_cns"
    assert payload["flags"]["anatomy_missing"] is True
    assert payload["context_positions"] == []
    caps = explorer_anatomy_captions(payload, is_synthetic=False, schema_error=err)
    assert any("arrow failed" in line for line in caps)
    assert ANATOMY_MISSING_CAPTION in caps


def test_explorer_has_one_operational_flow(monkeypatch, tmp_path, tiny_bearing_tables):
    from streamlit.testing.v1 import AppTest

    _explorer_harness(monkeypatch, tmp_path, tiny_bearing_tables)
    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)
    at.run()
    _open_explorer(at)
    assert not any(r.label == "Mode" for r in at.radio)
    assert not any(b.label in {"Build trace", "Load demo scenario"} for b in at.button)
    assert any(b.label == "Start test run" for b in at.button)


def _assert_schematic_cns_payload(captured: list[dict]) -> dict:
    """Dropped soma feather must not claim MaleCNS anatomy on synthetic ids."""
    assert captured
    payload = next(
        (c for c in captured if c.get("key") == "neural_activity_explorer"),
        captured[0],
    )
    flags = payload.get("flags") or {}
    assert flags.get("hull_mode") == "schematic_cns"
    assert flags.get("hull_mode") != "malecns_anatomy"
    assert not (payload.get("context_positions") or [])
    assert not flags.get("anatomy_missing")
    return payload


def test_synthetic_harness_not_malecns_anatomy(monkeypatch, tmp_path, tiny_bearing_tables):
    from streamlit.testing.v1 import AppTest

    from pdm.connectome.anatomy import SOMA_ALLOWLIST

    captured: list[dict] = []

    def _cap(*_args, **kwargs):
        captured.append(kwargs)
        return None

    rdir, _, _ = _explorer_harness(monkeypatch, tmp_path, tiny_bearing_tables, is_synthetic=True)
    nodes = ["10", "20", "30", "40"]
    (rdir / "connectome" / "graph.json").write_text(
        json.dumps({"nodes": nodes, "node_order": nodes, "edges": []}),
        encoding="utf-8",
    )
    _write_feather(
        tmp_path / SOMA_ALLOWLIST[0],
        {
            "bodyId": [10, 20, 30, 40],
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [0.0, 0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0, 0.0],
        },
    )
    monkeypatch.setattr("pdm.visualization.component.neural_activity_explorer", _cap)
    monkeypatch.setattr("pdm.visualization.simulation_ui.neural_activity_explorer", _cap)
    monkeypatch.setattr("pdm.app.neural_activity_explorer", _cap)
    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)
    monkeypatch.setattr("pdm.app.neural_activity_explorer", _cap)
    at.run()
    _open_explorer(at)
    text = _app_text(at)
    assert SYNTHETIC_DISCLAIMER in text
    assert EXPLORER_DISCLAIMER in text
    assert ANATOMY_MISSING_CAPTION not in text
    assert "registered MaleCNS" not in text
    payload = _assert_schematic_cns_payload(captured)
    assert set(str(n) for n in (payload.get("nodes") or [])) == set(nodes)


def test_real_connectome_without_soma_shows_anatomy_missing(monkeypatch, tmp_path, tiny_bearing_tables):
    from streamlit.testing.v1 import AppTest

    _explorer_harness(
        monkeypatch,
        tmp_path,
        tiny_bearing_tables,
        graph_mode="real_connectome",
        is_synthetic=False,
        disclaimer="",
    )
    captured = []
    monkeypatch.setattr("pdm.visualization.simulation_ui.neural_activity_explorer", lambda **kw: captured.append(kw))
    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)
    at.run()
    _open_explorer(at)
    assert not at.exception
    assert "cannot show every computing neuron" in _app_text(at)
    assert captured == []


def test_explorer_junk_soma_schema_caption_no_exception(monkeypatch, tmp_path, tiny_bearing_tables):
    from streamlit.testing.v1 import AppTest

    from pdm.connectome.anatomy import SOMA_ALLOWLIST

    _explorer_harness(
        monkeypatch,
        tmp_path,
        tiny_bearing_tables,
        graph_mode="real_connectome",
        is_synthetic=False,
        disclaimer="",
    )
    _write_feather(tmp_path / SOMA_ALLOWLIST[0], {"foo": [1], "bar": [2]})
    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)
    at.run()
    _open_explorer(at)
    assert not at.exception
    text = _app_text(at)
    assert "Columns found" in text
    assert "cannot show every computing neuron" in text


def test_explorer_soma_arrow_oserror_no_exception(monkeypatch, tmp_path, tiny_bearing_tables):
    from streamlit.testing.v1 import AppTest

    _explorer_harness(
        monkeypatch,
        tmp_path,
        tiny_bearing_tables,
        graph_mode="real_connectome",
        is_synthetic=False,
        disclaimer="",
    )

    def _boom(_path=None):
        raise OSError("arrow failed")

    monkeypatch.setattr("pdm.visualization.explorer.cached_soma_table", _boom)
    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)
    at.run()
    _open_explorer(at)
    assert not at.exception
    text = _app_text(at)
    assert "arrow failed" in text
    assert "cannot show every computing neuron" in text


def test_explorer_explicit_weights_path_caption_no_exception(monkeypatch, tmp_path, tiny_bearing_tables):
    from streamlit.testing.v1 import AppTest

    from pdm.connectome.anatomy import load_soma_table
    from pdm.connectome.sources import MALEMCNS_FILENAME

    _explorer_harness(
        monkeypatch,
        tmp_path,
        tiny_bearing_tables,
        graph_mode="real_connectome",
        is_synthetic=False,
        disclaimer="",
    )
    loc = tmp_path / MALEMCNS_FILENAME
    _write_feather(loc, {"body_pre": [1], "body_post": [2], "weight": [3]})

    def _explicit(path=None):
        return load_soma_table(loc)

    monkeypatch.setattr("pdm.visualization.explorer.cached_soma_table", _explicit)
    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)
    at.run()
    _open_explorer(at)
    assert not at.exception
    text = _app_text(at)
    assert "is not a soma anatomy table" in text
    assert "cannot show every computing neuron" in text


def test_random_rewire_synthetic_parent_not_flyem_anatomy_missing(
    monkeypatch, tmp_path, tiny_bearing_tables
):
    from streamlit.testing.v1 import AppTest

    from pdm.connectome.anatomy import SOMA_ALLOWLIST

    captured: list[dict] = []

    def _cap(*_args, **kwargs):
        captured.append(kwargs)
        return None

    rdir, _, _ = _explorer_harness(
        monkeypatch,
        tmp_path,
        tiny_bearing_tables,
        graph_mode="random_rewire",
        is_synthetic=True,
        parent_graph_mode="synthetic_fixture",
        parent_is_synthetic=True,
    )
    nodes = ["1", "2", "3", "4"]
    (rdir / "connectome" / "graph.json").write_text(
        json.dumps({"nodes": nodes, "node_order": nodes, "edges": []}),
        encoding="utf-8",
    )
    _write_feather(
        tmp_path / SOMA_ALLOWLIST[0],
        {
            "bodyId": [1, 2, 3, 4],
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [0.0, 0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0, 0.0],
        },
    )
    monkeypatch.setattr("pdm.visualization.component.neural_activity_explorer", _cap)
    monkeypatch.setattr("pdm.visualization.simulation_ui.neural_activity_explorer", _cap)
    monkeypatch.setattr("pdm.app.neural_activity_explorer", _cap)
    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)
    monkeypatch.setattr("pdm.app.neural_activity_explorer", _cap)
    at.run()
    _open_explorer(at)
    text = _app_text(at)
    assert SYNTHETIC_DISCLAIMER in text
    assert ANATOMY_MISSING_CAPTION not in text
    assert not at.exception
    payload = _assert_schematic_cns_payload(captured)
    assert set(str(n) for n in (payload.get("nodes") or [])) == set(nodes)


def test_random_rewire_real_parent_without_soma_anatomy_missing(
    monkeypatch, tmp_path, tiny_bearing_tables
):
    from streamlit.testing.v1 import AppTest

    _explorer_harness(
        monkeypatch,
        tmp_path,
        tiny_bearing_tables,
        graph_mode="random_rewire",
        is_synthetic=False,
        parent_graph_mode="real_connectome",
        parent_is_synthetic=False,
        disclaimer="",
    )
    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)
    at.run()
    _open_explorer(at)
    assert not at.exception
    assert "cannot show every computing neuron" in _app_text(at)


def test_live_snapshot_source_does_not_stamp_hull_or_context():
    import inspect

    from pdm.visualization.live import write_training_live_snapshot

    src = inspect.getsource(write_training_live_snapshot)
    assert "hull_mode" not in src
    assert "context_positions" not in src


def test_explorer_train_live_compact_key_does_not_hijack_overlay(
    monkeypatch, tmp_path, tiny_bearing_tables
):
    from streamlit.testing.v1 import AppTest

    captured: list[dict] = []

    def _cap(*_args, **kwargs):
        captured.append(kwargs)
        return None

    _explorer_harness(
        monkeypatch,
        tmp_path,
        tiny_bearing_tables,
        worker_alive=True,
        graph_mode="real_connectome",
        is_synthetic=False,
        disclaimer="",
        n_nodes=8,
    )
    live = {
        "status": "ok",
        "architecture": "fly_connectome_reservoir",
        "states": [[0.2, 0.3]],
        "inputs": [[0.0]],
        "node_order": ["0", "1"],
        "predicted_rul_s": 12.5,
        "unit_id": "TRAIN_PROBE",
        "dataset_id": "bearings",
        "run_id": "bearings_fly_explorer",
        "frame_map": [{"timestamp_s": 0.0, "window_end_timestamp_s": 4.0}],
        "downsampled": True,
        "hull_mode": "schematic_cns",
    }
    monkeypatch.setattr("pdm.visualization.live.load_live_activity", lambda: live)
    monkeypatch.setattr("pdm.app.load_live_activity", lambda: live)
    monkeypatch.setattr("pdm.worker.read_status", lambda: {"status": "training", "kind": "train"})
    monkeypatch.setattr("pdm.app.read_status", lambda: {"status": "training", "kind": "train"})
    monkeypatch.setattr("pdm.visualization.component.neural_activity_explorer", _cap)
    monkeypatch.setattr("pdm.visualization.simulation_ui.neural_activity_explorer", _cap)
    monkeypatch.setattr("pdm.app.neural_activity_explorer", _cap)
    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)
    monkeypatch.setattr("pdm.app.neural_activity_explorer", _cap)
    at.run()
    _open_explorer(at)
    assert not at.exception
    keys = [c.get("key") for c in captured]
    assert "explorer_train_live" in keys
    live_payload = next(c for c in captured if c.get("key") == "explorer_train_live")
    assert live_payload["flags"]["n_nodes_display"] == 2
    assert len(live_payload["states"][0]) == 2
    assert live_payload["flags"].get("n_model") == 8
    assert live_payload["flags"].get("n_model") != 2
    text = _app_text(at)
    assert ANATOMY_MISSING_CAPTION in text
    metrics = "\n".join(str(getattr(m, "value", m)) for m in at.metric)
    assert "12.5" not in metrics
    assert "TRAIN_PROBE" in text


def test_cartoon_mapping_only_when_schematic_cns_lives_in_webgl_suite():
    """JS fitNodesIntoHull / polarToCns: tests/test_explorer_webgl.py (do not duplicate greps)."""
    text = (project_root() / "tests" / "test_explorer_webgl.py").read_text(encoding="utf-8")
    assert "def test_cartoon_mapping_only_when_schematic_cns" in text
    assert "fitNodesIntoHull" in text
    assert "polarToCns" in text


def test_soma_search_is_allowlist_not_glob():
    import inspect

    from pdm.connectome import anatomy
    from pdm.connectome.anatomy import SOMA_ALLOWLIST, find_soma_table_path

    src = inspect.getsource(find_soma_table_path)
    assert "glob(" not in src
    assert "rglob" not in src
    assert "*.feather" not in src
    assert "SOMA_ALLOWLIST" in src
    assert SOMA_ALLOWLIST[0] == "body-annotations-male-cns-v1.0-minconf-0.5.feather"
    module_src = inspect.getsource(anatomy)
    assert "write_provenance" not in module_src
    assert "write_graph_artifact" not in module_src


def test_soma_viz_provenance_does_not_overwrite_run_provenance(monkeypatch, tmp_path):
    from pdm.connectome.anatomy import SOMA_ALLOWLIST, load_soma_table

    _patch_soma_dir(monkeypatch, tmp_path)
    run_dir = tmp_path / "run"
    orig = {
        "graph_mode": "real_connectome",
        "is_synthetic": False,
        "n_nodes": 8,
        "source": "local",
    }
    _write_run_connectome(
        run_dir,
        nodes=["1001", "1002"],
        layout={"positions": {"1001": [0.0, 0.0, 0.0], "1002": [1.0, 0.0, 0.0]}},
        provenance=orig,
    )
    prov_path = run_dir / "connectome" / "provenance.json"
    before = prov_path.read_text(encoding="utf-8")
    _write_feather(
        tmp_path / SOMA_ALLOWLIST[0],
        {
            "bodyId": [1001, 1002, 1003],
            "x": [1.0, 2.0, 3.0],
            "y": [4.0, 5.0, 6.0],
            "z": [7.0, 8.0, 9.0],
        },
    )
    table = load_soma_table()
    assert table.provenance["graph_mode"] == "viz_soma_xyz"
    assert table.provenance["n_nodes"] == 0
    assert table.provenance["n_points"] == 3
    after = prov_path.read_text(encoding="utf-8")
    assert after == before
    rec = json.loads(after)
    assert rec["graph_mode"] == "real_connectome"
    assert rec["n_nodes"] == 8
    assert rec.get("n_points") is None


def test_explorer_docs_single_flow_anatomy_history_and_validation():
    text = (project_root() / "docs" / "neural_activity_explorer.md").read_text(encoding="utf-8")
    assert EXPLORER_DISCLAIMER in text
    assert SYNTHETIC_DISCLAIMER in text
    assert RESERVOIR_REQUIRED_MESSAGE in text
    assert "A heavy job is already running. Please wait." in text
    assert "one scenario" in text
    assert "no Modes selector or trace-building prerequisite" in text
    assert "166,700 classified MaleCNS" in text
    assert "All neurons compute" in text
    assert "Missing cells are not placed at invented coordinates" in text
    assert "Coordinates, morphology and states share body-ID order" in text
    assert "persistent chronological state" in text
    assert "interval_profile.json" in text
    assert "dataset fingerprint" in text
    assert "Ground-truth visibility does not change inference" in text
    assert "Payload tests do not certify appearance" in text
    assert "desktop and mobile" in text
    assert "browser console" in text
    fly = (project_root() / "docs" / "fly_connectome.md").read_text(encoding="utf-8")
    assert "Soma xyz (viz only)" in fly
    assert "malecns_visualization.md" in fly
    assert "layout_positions()" in fly
    assert "500–2000" in fly
    assert "W_res[i, j] = edge j → i" in fly
    validation = (project_root() / "docs" / "fly_connectome_validation.md").read_text(encoding="utf-8")
    assert "AppTest does not certify WebGL look" in validation
    assert "pytest greps + payload flags are the merge gate" in validation
    assert "AppTest is the merge gate" not in validation
    assert RESERVOIR_REQUIRED_MESSAGE in validation


def test_malecns_visualization_doc_reference_urls_and_allowlist():
    text = (project_root() / "docs" / "malecns_visualization.md").read_text(encoding="utf-8")
    assert "https://neuroglancer-demo.appspot.com/#!gs://flyem-male-cns/v1.0/male-cns-v1.0.json" in text
    assert "https://male-cns.janelia.org/explore/" in text
    assert "https://male-cns.janelia.org/download/" in text
    assert "CC-BY" in text
    assert "body-annotations-male-cns-v1.0-minconf-0.5.feather" in text
    assert "No `*.feather` glob" in text
    assert "syn-points-" in text
    assert "no xyz" in text.lower()
    assert "166k" in text
    assert "viz_soma_xyz" in text
    assert "unavailable" in text
    assert "provenance.json" in text
    assert "gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/" in text
    assert "https://neuprint.janelia.org/" in text
    assert "n_nodes" in text
    assert "iframe" in text.lower()




def test_anatomy_never_uses_spring_coordinates_even_inside_soma_bounds():
    from pdm.connectome.anatomy import SomaTable

    soma = SomaTable(positions={"a": [0, 0, 0], "b": [2, 2, 2]}, provenance={}, is_synthetic=False, n_points=2)
    payload = build_explorer_payload(
        nodes=["fragment"], edges=[], positions={"fragment": [1, 1, 1]},
        states=np.ones((1, 1)), inputs=np.ones((1, 1)), frame_map=[],
        flags={"graph_mode": "real_connectome"}, soma=soma, n_model=1,
    )
    assert payload["positions"] == {}
    assert payload["flags"]["n_with_soma"] == 0
    assert payload["flags"]["n_unmatched_reservoir"] == 1
    assert payload["states"] == [[1.0]]


def test_equipment_simulation_clock_ground_truth_and_busy(monkeypatch, tmp_path, tiny_bearing_tables):
    from streamlit.testing.v1 import AppTest

    from pdm.connectome.sources import load_synthetic_fixture
    from pdm.models import FlyConnectomeReservoir
    from pdm.preprocessing import Preprocessor
    from pdm.visualization import simulation_ui

    rdir, _, _ = _explorer_harness(monkeypatch, tmp_path, tiny_bearing_tables)
    (rdir / "best.pt").write_bytes(b"test checkpoint")
    prep = Preprocessor(
        feature_names=["horizontal_rms"], log1p_features=[], scaler_mean=[0.0],
        scaler_scale=[1.0], time_scale_s=100.0, fill_values={"horizontal_rms": 0.0},
        dataset_id="bearings",
    )
    model = FlyConnectomeReservoir(load_synthetic_fixture().graph, input_size=1, seed=4)
    nodes = list(model.node_order)
    (rdir / "connectome" / "graph.json").write_text(json.dumps({"node_order": nodes, "edges": []}))
    monkeypatch.setattr(simulation_ui, "_simulation_model", lambda *args: (model, prep, {"history_length": 5}))
    monkeypatch.setattr(simulation_ui, "worker_alive", lambda: False)
    captured = {}
    monkeypatch.setattr(simulation_ui, "neural_activity_explorer", lambda **kw: captured.update(kw))
    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=30)
    at.run()
    _open_explorer(at)
    assert not at.exception
    assert not at.error
    assert "Collecting history" in _app_text(at)
    slider = next(w for w in at.slider if w.label == "Measurement")
    slider.set_value(7).run()
    assert not at.exception
    assert not at.error
    assert len(captured["states"]) == 1
    assert captured["flags"]["synchronized"] is True
    now = captured["frame_map"][0]["timestamp_s"]
    pred = captured["predicted_rul_s"]
    chart = at.get("plotly_chart")[0]
    chart_meta = json.loads(chart.proto.spec)["layout"]["meta"]
    assert chart_meta["now_x"] == now / 60
    assert chart_meta["stored_predicted_rul_s"] == pred
    next(w for w in at.checkbox if w.label == "Show ground truth").set_value(False).run()
    assert captured["frame_map"][0]["timestamp_s"] == now
    assert captured["predicted_rul_s"] == pred
    next(w for w in at.button if w.label == "Next measurement").click().run()
    assert captured["frame_map"][0]["timestamp_s"] > now
    next(w for w in at.button if w.label == "Start test run").click().run()
    assert not at.exception
    next(w for w in at.button if w.label == "Pause test run").click().run()
    assert not at.exception
    frozen = captured["frame_map"][0]["timestamp_s"]
    at.run()
    assert captured["frame_map"][0]["timestamp_s"] == frozen
    next(w for w in at.button if w.label == "Reset test run").click().run()
    assert not at.exception
    assert captured["predicted_rul_s"] is None
    assert captured["states"] == []
    monkeypatch.setattr(simulation_ui, "worker_alive", lambda: True)
    monkeypatch.setattr(simulation_ui, "simulate_step", lambda *args: (_ for _ in ()).throw(AssertionError("busy inference")))
    next(w for w in at.button if w.label == "Next measurement").click().run()
    assert not at.exception
    assert "A heavy job is already running. Please wait." in _app_text(at)


def test_equipment_simulation_shorter_data_resets_cursor(monkeypatch, tmp_path, tiny_bearing_tables):
    from streamlit.testing.v1 import AppTest

    _explorer_harness(monkeypatch, tmp_path, tiny_bearing_tables)
    from pdm.replay import bind_replay_to_run

    bound = bind_replay_to_run("bearings", "bearings_fly_explorer")
    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=20)
    at.run()
    _open_explorer(at)
    next(w for w in at.slider if w.label == "Measurement").set_value(7).run()
    assert not at.exception
    shorter = {
        **bound,
        "features": bound["features"].groupby("unit_id", sort=False).head(3).copy(),
        "current_fingerprint": {"dataset_version": "replaced-data"},
    }
    monkeypatch.setattr("pdm.replay.bind_replay_to_run", lambda *args: shorter)
    at.run()
    assert not at.exception
    assert not at.error
    assert next(w for w in at.slider if w.label == "Measurement").value == 0
