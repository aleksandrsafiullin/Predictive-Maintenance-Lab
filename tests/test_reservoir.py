from __future__ import annotations

import numpy as np
import pytest
import torch

from pdm.config import load_dataset_config, model_defaults
from pdm.connectome.graph import graph_from_edges
from pdm.connectome.provenance import GRAPH_MODE_RANDOM_REWIRE, hash_graph
from pdm.connectome.sampling import resolve_n_nodes, sample_connected_subgraph
from pdm.connectome.sources import load_malemcns, load_synthetic_fixture
from pdm.connectome.weights import log1p_adjacency
from pdm.models import FlyConnectomeReservoir, LeakyESN, RandomReservoir
from pdm.models.readout import fit_ridge
from pdm.paths import configs_root
from pdm.train import run_training

SYNTHETIC_LABEL = "Synthetic test graph — not a biological connectome"


def _cycle_graph(n: int = 8):
    """Tiny directed cycle plus chords. Explicit test size, never the 50-node fixture."""
    nodes = [str(i) for i in range(int(n))]
    edges = [
        {"src": nodes[i], "dst": nodes[(i + 1) % n], "weight": float(2 + (i % 3))} for i in range(n)
    ]
    for i in range(0, n, 2):
        edges.append({"src": nodes[i], "dst": nodes[(i + 3) % n], "weight": 1.0})
    return graph_from_edges(edges, nodes)


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


def test_graph_orientation():
    graph = graph_from_edges([{"src": "src", "dst": "dst", "weight": 4.0}], nodes=["src", "dst"])
    model = FlyConnectomeReservoir(
        graph,
        input_size=1,
        seed=0,
        node_order=["src", "dst"],
        spectral_radius=0.9,
    )
    src_i = model.node_order.index("src")
    dst_i = model.node_order.index("dst")
    w = model.W_res.detach()
    assert float(w[dst_i, src_i]) != 0.0
    assert float(w[src_i, dst_i]) == 0.0


def test_state_update_hand_calculation():
    """Hand-computed leaky step. Expected values come from numpy, not the production kernel.

    Implementation notes: x[0] = [1, 0], W_res[1, 0] = c, alpha = 1, W_in = 0, b = 0
    ⇒ x[1] = tanh(W_res @ x[0]) = [0, tanh(c)].
    """
    c = 2.0
    alpha = 1.0
    w_res = np.zeros((2, 2), dtype=np.float64)
    w_res[1, 0] = c
    w_in = np.zeros((2, 1), dtype=np.float64)
    b_res = np.zeros(2, dtype=np.float64)
    x0 = np.array([1.0, 0.0], dtype=np.float64)
    u = np.zeros(1, dtype=np.float64)
    expected = (1.0 - alpha) * x0 + alpha * np.tanh(w_res @ x0 + w_in @ u + b_res)
    np.testing.assert_allclose(expected, np.array([0.0, np.tanh(c)]))

    esn = LeakyESN(
        W_in=torch.tensor(w_in, dtype=torch.float32),
        W_res=torch.tensor(w_res, dtype=torch.float32),
        b_res=torch.tensor(b_res, dtype=torch.float32),
        alpha=alpha,
    )
    states = esn.forward_states(torch.zeros(1, 1), x0=torch.tensor(x0, dtype=torch.float32))
    np.testing.assert_allclose(states[0].detach().cpu().numpy(), expected, rtol=1e-6, atol=1e-6)


def test_seed_reproducibility():
    graph = _cycle_graph(8)
    a = FlyConnectomeReservoir(graph, input_size=3, seed=42, spectral_radius=0.9)
    b = FlyConnectomeReservoir(graph, input_size=3, seed=42, spectral_radius=0.9)
    np.testing.assert_allclose(
        a.W_res.detach().cpu().numpy(),
        b.W_res.detach().cpu().numpy(),
    )
    np.testing.assert_allclose(
        a.W_in.detach().cpu().numpy(),
        b.W_in.detach().cpu().numpy(),
    )


def test_forward_bearings_returns_1d_nonneg_norm_rul():
    graph = _cycle_graph(8)
    model = FlyConnectomeReservoir(graph, input_size=3, head="rul", seed=1)
    assert not model.W_in.requires_grad
    assert not model.W_res.requires_grad
    assert not model.b_res.requires_grad
    assert model.readout.W_x.requires_grad
    out = model(torch.randn(4, 5, 3))
    assert out.shape == (4,)
    assert out.dim() == 1
    assert torch.all(out >= 0)


def test_forward_filters_returns_lam_k_tuple():
    graph = _cycle_graph(8)
    model = FlyConnectomeReservoir(graph, input_size=3, head="weibull", seed=1)
    out = model(torch.randn(4, 5, 3))
    assert isinstance(out, tuple) and len(out) == 2
    lam, k = out
    assert lam.shape == (4,)
    assert k.shape == (4,)
    assert torch.all(lam > 0)
    assert torch.all(k > 0)


def test_ridge_bearings_no_double_softplus():
    graph = _cycle_graph(8)
    model = FlyConnectomeReservoir(graph, input_size=2, head="rul", seed=0)
    x = torch.randn(24, 4, 2)
    with torch.no_grad():
        states = model.forward_states(x)
    x_t = states[:, -1, :].detach().cpu().numpy().astype(np.float64)
    u_t = x[:, -1, :].detach().cpu().numpy().astype(np.float64)
    z = np.concatenate([x_t, u_t, np.ones((x_t.shape[0], 1), dtype=np.float64)], axis=1)
    rng = np.random.default_rng(0)
    w_true = rng.normal(scale=0.05, size=z.shape[1])
    y_norm = np.abs(z @ w_true) + 0.1
    w, residual_mean_sq = fit_ridge(z, y_norm, alpha=1e-8)
    assert residual_mean_sq < 1e-3
    model.readout.load_ridge_vector(w)
    with torch.no_grad():
        pred = model(x)
        raw = model.forward_raw(states[:, -1, :], x[:, -1, :])
    relu_raw = torch.relu(raw).squeeze(-1)
    assert torch.all(pred >= 0)
    np.testing.assert_allclose(
        pred.detach().cpu().numpy(),
        relu_raw.detach().cpu().numpy(),
        atol=1e-5,
    )


def test_random_reservoir_parent_graph_hash():
    parent = _cycle_graph(8)
    parent_h = hash_graph(parent)
    model = RandomReservoir(parent, input_size=2, seed=3)
    assert model.parent_graph_hash == parent_h
    assert model.provenance["parent_graph_hash"] == parent_h
    assert model.provenance["graph_mode"] == GRAPH_MODE_RANDOM_REWIRE
    assert model.graph_mode == GRAPH_MODE_RANDOM_REWIRE
    assert SYNTHETIC_LABEL in str(model.provenance["disclaimer"])
    assert parent.number_of_nodes() == model.graph.number_of_nodes()
    assert parent.number_of_edges() == model.graph.number_of_edges()
    for node in parent.nodes():
        assert parent.in_degree(node) == model.graph.in_degree(node)
        assert parent.out_degree(node) == model.graph.out_degree(node)
