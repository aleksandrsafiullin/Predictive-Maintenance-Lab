from __future__ import annotations

import numpy as np
import pytest
import torch

from pdm.config import load_dataset_config, model_defaults
from pdm.connectome.graph import graph_from_edges
from pdm.connectome.sampling import resolve_n_nodes, sample_connected_subgraph
from pdm.connectome.sources import load_malemcns, load_synthetic_fixture
from pdm.connectome.weights import log1p_adjacency
from pdm.paths import configs_root
from pdm.train import run_training

SYNTHETIC_LABEL = "Synthetic test graph — not a biological connectome"


def test_synthetic_label_in_metadata():
    src = load_synthetic_fixture()
    assert src.is_synthetic is True
    assert src.payload.get("is_synthetic") is True
    assert src.label == SYNTHETIC_LABEL
    assert src.payload.get("label") == SYNTHETIC_LABEL
    assert src.provenance.get("disclaimer") == SYNTHETIC_LABEL
    assert src.provenance.get("graph_mode") == "synthetic_fixture"
    assert src.graph.number_of_nodes() >= 50


def test_synthetic_n_nodes_clamps_not_raises():
    src = load_synthetic_fixture()
    available = src.graph.number_of_nodes()
    assert resolve_n_nodes("synthetic_fixture", 1000, available) == available
    assert resolve_n_nodes("synthetic_fixture", 8, available) == 8
    assert resolve_n_nodes("synthetic_fixture", available * 10, available) == available


def test_real_connectome_n_nodes_out_of_range_raises():
    with pytest.raises(ValueError, match="real_connectome"):
        resolve_n_nodes("real_connectome", 400, 10_000)
    with pytest.raises(ValueError, match="real_connectome"):
        resolve_n_nodes("real_connectome", 2500, 10_000)
    with pytest.raises(ValueError, match="exceeds available"):
        resolve_n_nodes("real_connectome", 800, 600)


def test_sampling_determinism():
    src = load_synthetic_fixture()
    n = resolve_n_nodes("synthetic_fixture", 8, src.graph.number_of_nodes())
    a = sample_connected_subgraph(src.graph, n, seed=42)
    b = sample_connected_subgraph(src.graph, n, seed=42)
    assert a == b
    assert len(a) == 8
    assert all(isinstance(nid, str) for nid in a)


def test_adjacency_orientation():
    graph = graph_from_edges([{"src": "0", "dst": "1", "weight": 3}], nodes=["0", "1"])
    adj = log1p_adjacency(graph, ["0", "1"])
    assert adj[1, 0] != 0
    assert adj[0, 1] == 0
    np.testing.assert_allclose(adj[1, 0], np.log1p(3.0))


def test_missing_malemens_fallback_does_not_raise(tmp_path):
    missing = tmp_path / "no-such-connectome.feather"
    got = load_malemcns(missing)
    assert got.provenance.get("source") == "unavailable"
    assert got.is_synthetic is True
    assert got.provenance.get("graph_mode") == "synthetic_fixture"
    assert got.label == SYNTHETIC_LABEL
    assert got.graph.number_of_nodes() >= 50


def test_yaml_reservoir_defaults_parse():
    cfg = load_dataset_config("bearings")
    mcfg = model_defaults(cfg)
    res = mcfg["reservoir"]
    assert mcfg["architecture"] == "gru"
    assert res["n_nodes"] == 1000
    assert res["leak"] == 0.2
    assert res["spectral_radius"] == 0.9
    assert res["input_scale"] == 0.1
    assert res["ridge_alpha"] == 0.001
    assert res["seed"] == 42
    assert res["state_mode"] == "window_reset"
    assert res["graph_mode"] == "synthetic_fixture"


def test_filters_yaml_no_readout_ridge():
    text = (configs_root() / "filters.yaml").read_text(encoding="utf-8")
    assert "readout: ridge" not in text
    cfg = load_dataset_config("filters")
    reservoir = (cfg.get("model") or {}).get("reservoir") or {}
    assert "readout" not in reservoir


def test_model_defaults_sets_readout_from_dataset_id():
    bearings = model_defaults(load_dataset_config("bearings"))
    filters = model_defaults(load_dataset_config("filters"))
    assert bearings["reservoir"]["readout"] == "ridge"
    assert filters["reservoir"]["readout"] == "gradient"
    bad = load_dataset_config("filters")
    bad.setdefault("model", {}).setdefault("reservoir", {})["readout"] = "ridge"
    with pytest.raises(ValueError, match="readout=ridge"):
        model_defaults(bad)


def test_reservoir_arch_raises_in_run_training(monkeypatch):
    created = {"opt": False}

    class _SpyAdamW:
        def __init__(self, *args, **kwargs):
            created["opt"] = True
            raise AssertionError("AdamW must not be constructed for reservoir architectures")

    monkeypatch.setattr(torch.optim, "AdamW", _SpyAdamW)
    with pytest.raises(NotImplementedError, match="Reservoir training not yet implemented in run_training"):
        run_training(dataset_id="bearings", architecture="fly_connectome_reservoir")
    assert created["opt"] is False


def test_run_connectome_and_traces_dirs():
    from pdm.paths import dataset_runs, run_connectome_dir, run_traces_dir

    assert run_connectome_dir("bearings", "run1") == dataset_runs("bearings") / "run1" / "connectome"
    assert run_traces_dir("filters", "run2") == dataset_runs("filters") / "run2" / "traces"
    assert run_traces_dir("filters", "run2", "U1") == dataset_runs("filters") / "run2" / "traces" / "U1"


def test_build_model_reservoir_does_not_call_recurrent_encoder(monkeypatch):
    from pdm.models import RecurrentEncoder, build_model

    def _boom(*_a, **_k):
        raise AssertionError("RecurrentEncoder must not see reservoir architecture strings")

    monkeypatch.setattr("pdm.models.RecurrentEncoder", _boom)
    with pytest.raises(NotImplementedError, match="Reservoir training not yet implemented"):
        build_model(architecture="random_reservoir", input_size=4)
    monkeypatch.setattr("pdm.models.RecurrentEncoder", RecurrentEncoder)
    net = build_model(architecture="gru", input_size=4, hidden_size=8)
    assert net.encoder.architecture == "gru"
