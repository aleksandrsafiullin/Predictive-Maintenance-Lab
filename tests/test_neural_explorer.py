from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from pdm.connectome.provenance import SYNTHETIC_DISCLAIMER
from pdm.paths import project_root
from pdm.visualization.explorer import EXPLORER_DISCLAIMER, build_explorer_payload

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
):
    from pdm.splits import bearings_split

    features, units = tiny_bearing_tables
    split = bearings_split(units)
    bundle = _fake_processed_bundle("bearings", features, units, split)
    rdir = tmp_path / run_id
    rdir.mkdir(parents=True, exist_ok=True)
    cdir = rdir / "connectome"
    cdir.mkdir()
    (cdir / "provenance.json").write_text(
        json.dumps(
            {
                "graph_mode": graph_mode,
                "is_synthetic": is_synthetic,
                "disclaimer": SYNTHETIC_DISCLAIMER,
            }
        ),
        encoding="utf-8",
    )
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
    monkeypatch.setattr("pdm.data.prepare.processed_ready", lambda ds: ds == "bearings")
    monkeypatch.setattr("pdm.data.prepare.load_processed", lambda ds: bundle)
    monkeypatch.setattr("pdm.experiments.list_runs", lambda ds=None: [fake_row])
    monkeypatch.setattr("pdm.experiments.run_dir", lambda ds, rid: rdir)
    monkeypatch.setattr("pdm.worker.worker_alive", lambda: worker_alive)
    if captured is not None:
        monkeypatch.setattr("pdm.cli.spawn_worker", lambda job: captured.update(job) or captured)
    return rdir, split, fake_row


def _open_explorer(at):
    _screen_radio(at).set_value("Neural Activity Explorer")
    at.run()
    assert not at.exception
    return at


def test_explorer_screen_present_in_app():
    """Neural Activity Explorer appears in screen radio options."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)
    at.run()
    assert not at.exception
    radio = _screen_radio(at)
    opts = list(radio.options)
    assert "Neural Activity Explorer" in opts
    assert "Data" in opts
    assert "Train" in opts
    assert "Test & Replay" in opts


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


def test_worker_busy_shows_error_not_inline(monkeypatch, tmp_path, tiny_bearing_tables):
    """When worker busy, Build trace shows error caption, not inline trace."""
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
    monkeypatch.setattr("pdm.app.run_trace_job", _no_job)
    monkeypatch.setattr("pdm.cli.spawn_worker", _no_spawn)
    monkeypatch.setattr("pdm.app.spawn_worker", _no_spawn)

    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)
    at.run()
    _open_explorer(at)
    btn = next(b for b in at.button if "Build trace" in b.label)
    btn.click()
    at.run()
    assert not at.exception
    errors = "\n".join(str(w.value) for w in at.error)
    assert "A heavy job is already running. Please wait." in errors
    assert inline["n"] == 0
    assert spawned["n"] == 0


def test_no_cdn_in_frontend():
    """No https:// script/module src in frontend HTML/JS."""
    html_files = list(FRONTEND.rglob("*.html"))
    js_files = [p for p in FRONTEND.rglob("*.js") if "vendor" not in p.parts]
    assert html_files
    assert js_files
    for path in html_files + js_files:
        text = path.read_text(encoding="utf-8")
        for host in CDN_HOSTS:
            assert host not in text, f"{path} contains CDN host {host}"
        assert SCRIPT_HTTPS_SRC.search(text) is None, f"{path} has https script src"
        assert MODULE_HTTPS.search(text) is None, f"{path} has https module import"
        assert 'src="https://' not in text
        assert "src='https://" not in text
        assert "src=`https://" not in text
    main_js = (FRONTEND / "main.js").read_text(encoding="utf-8")
    assert "Math.random" not in main_js
    assert "requestAnimationFrame" in main_js


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
    )
    assert payload["states"] == np.asarray(trace["states"], dtype=float).tolist()
    assert len(payload["states"]) == trace["states"].shape[0]
    assert len(payload["states"][0]) == trace["states"].shape[1]
    assert payload["states"][0] != payload["states"][-1] or trace["states"].shape[0] == 1


def test_screen_switch_by_label():
    """AppTest screen switch uses label not index."""
    from streamlit.testing.v1 import AppTest

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
