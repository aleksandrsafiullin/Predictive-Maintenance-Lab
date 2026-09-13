from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from pdm.config import load_dataset_config, model_defaults
from pdm.connectome.graph import graph_from_edges, graph_from_payload
from pdm.connectome.provenance import GRAPH_MODE_RANDOM_REWIRE, hash_graph
from pdm.connectome.sampling import prepare_run_graph, resolve_n_nodes, sample_connected_subgraph
from pdm.connectome.sources import load_malemcns, load_synthetic_fixture
from pdm.connectome.weights import log1p_adjacency
from pdm.losses import weibull_nll
from pdm.models import FlyConnectomeReservoir, LeakyESN, RandomReservoir
from pdm.models.readout import fit_ridge
from pdm.paths import configs_root
from pdm.predict import Predictor
from pdm.preprocessing import Preprocessor
from pdm.train import UnitWindowDataset, checkpoints_compatible, load_trained_model, run_training
from pdm.visualization.export import load_trace, save_trace
from pdm.visualization.trace import predict_with_trace
from pdm.windows import BEARINGS_RAW_NUMERIC_COLUMNS

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


def test_synthetic_fixture_label():
    """Grep-able acceptance test 14: fixture is labeled synthetic, not biology."""
    src = load_synthetic_fixture()
    assert src.is_synthetic is True
    assert src.label == SYNTHETIC_LABEL
    assert src.payload.get("is_synthetic") is True
    assert src.payload.get("label") == SYNTHETIC_LABEL


def test_synthetic_n_nodes_clamps_not_raises():
    src = load_synthetic_fixture()
    available = src.graph.number_of_nodes()
    assert resolve_n_nodes("synthetic_fixture", 1000, available) == available
    assert resolve_n_nodes("synthetic_fixture", 8, available) == 8
    assert resolve_n_nodes("synthetic_fixture", available * 10, available) == available
    graph, prov, resolved = prepare_run_graph(
        architecture="fly_connectome_reservoir",
        graph_mode="synthetic_fixture",
        n_nodes=1000,
        seed=0,
    )
    assert resolved == available
    assert graph.number_of_nodes() == available
    assert prov.get("graph_mode") == "synthetic_fixture"
    assert prov.get("is_synthetic") is True


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
    graph, prov, resolved = prepare_run_graph(
        architecture="fly_connectome_reservoir",
        graph_mode="real_connectome",
        n_nodes=500,  # changed from 8 to be in valid range
        seed=1,
        source_path=missing,
    )
    assert prov.get("graph_mode") == "synthetic_fixture"
    assert resolved >= 1
    assert graph.number_of_nodes() >= 1  # clamped to fixture size


def _make_tiny_feather(tmp_path: Path, n_nodes: int = 30, n_edges: int = 200, seed: int = 0) -> Path:
    """Create a tiny feather with body_pre/body_post/weight columns for unit tests."""
    rng = np.random.default_rng(seed)
    nodes = list(range(n_nodes))
    src = rng.choice(nodes, size=n_edges).astype(np.int64)
    dst = rng.choice(nodes, size=n_edges).astype(np.int64)
    weight = rng.integers(1, 10, size=n_edges).astype(np.int32)
    df = pd.DataFrame({"body_pre": src, "body_post": dst, "weight": weight})
    p = tmp_path / "tiny_connectome.feather"
    df.to_feather(p)
    return p


def test_load_malemcns_subgraph_tiny_feather(tmp_path):
    """Sampling logic on tiny temp feather — does not use the real 1GB file."""
    from pdm.connectome.sources import load_malemcns_subgraph

    p = _make_tiny_feather(tmp_path)
    result = load_malemcns_subgraph(p, n_nodes=10, seed=7)
    assert result.is_synthetic is False
    assert result.provenance["source"] == "local"
    assert result.provenance["graph_mode"] == "real_connectome"
    assert result.provenance["full_graph_materialized"] is False
    assert result.provenance["sampling_method"] == "seeded_bfs"
    assert result.graph.number_of_nodes() <= 10
    assert result.provenance["n_nodes"] == result.graph.number_of_nodes()
    assert result.provenance["seed"] == 7
    assert result.provenance["file_hash"] is not None
    assert result.provenance["column_names_read"] == ["body_pre", "body_post", "weight"]


def test_load_malemcns_subgraph_determinism(tmp_path):
    from pdm.connectome.sources import load_malemcns_subgraph

    p = _make_tiny_feather(tmp_path)
    a = load_malemcns_subgraph(p, n_nodes=8, seed=42)
    b = load_malemcns_subgraph(p, n_nodes=8, seed=42)
    assert sorted(a.graph.nodes()) == sorted(b.graph.nodes())
    assert sorted(a.graph.edges()) == sorted(b.graph.edges())
    # Different seed → different subgraph (usually)
    c = load_malemcns_subgraph(p, n_nodes=8, seed=99)
    # Not guaranteed to differ for tiny graphs but check it runs
    assert c.graph.number_of_nodes() <= 8


def test_load_malemcns_subgraph_missing_fallback(tmp_path):
    from pdm.connectome.sources import load_malemcns_subgraph

    missing = tmp_path / "no-such-file.feather"
    result = load_malemcns_subgraph(missing, n_nodes=10, seed=0)
    assert result.is_synthetic is True
    assert result.provenance["source"] == "unavailable"
    assert result.provenance["graph_mode"] == "synthetic_fixture"


def test_prepare_run_graph_real_uses_subgraph_bfs(tmp_path):
    """prepare_run_graph with real_connectome uses subgraph BFS (not full NetworkX)."""
    p = _make_tiny_feather(tmp_path, n_nodes=50, n_edges=500)
    graph, prov, resolved = prepare_run_graph(
        architecture="fly_connectome_reservoir",
        graph_mode="real_connectome",
        n_nodes=500,  # valid real_connectome range; tiny file has fewer nodes
        seed=5,
        source_path=p,
    )
    assert graph.number_of_nodes() <= 500
    assert prov["graph_mode"] == "real_connectome"
    assert prov["source"] == "local"
    assert prov["full_graph_materialized"] is False
    assert prov["seed"] == 5
    assert resolved == graph.number_of_nodes()


def test_bfs_subgraph_from_df_determinism(tmp_path):
    from pdm.connectome.sources import _bfs_subgraph_from_df

    p = _make_tiny_feather(tmp_path)
    df = pd.read_feather(p)
    a, _ = _bfs_subgraph_from_df(df, "body_pre", "body_post", n_nodes=8, seed=3)
    b, _ = _bfs_subgraph_from_df(df, "body_pre", "body_post", n_nodes=8, seed=3)
    assert a == b
    assert len(a) <= 8
    assert all(isinstance(nid, str) for nid in a)


def test_bfs_subgraph_weight_threshold(tmp_path):
    from pdm.connectome.sources import load_malemcns_subgraph

    p = _make_tiny_feather(tmp_path)
    result = load_malemcns_subgraph(p, n_nodes=10, seed=0, weight_threshold=5)
    assert result.provenance["weight_threshold"] == 5
    assert "weight_threshold=5" in result.provenance["weight_threshold_note"]


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


def test_reservoir_arch_does_not_raise_not_implemented():
    from pdm.models import FlyConnectomeReservoir, build_model

    graph = _cycle_graph(8)
    model = build_model(
        architecture="fly_connectome_reservoir",
        input_size=3,
        graph=graph,
        n_nodes=8,
        seed=0,
    )
    assert isinstance(model, FlyConnectomeReservoir)


def test_run_connectome_and_traces_dirs():
    from pdm.paths import dataset_runs, run_connectome_dir, run_traces_dir

    assert run_connectome_dir("bearings", "run1") == dataset_runs("bearings") / "run1" / "connectome"
    assert run_traces_dir("filters", "run2") == dataset_runs("filters") / "run2" / "traces"
    assert run_traces_dir("filters", "run2", "U1") == dataset_runs("filters") / "run2" / "traces" / "U1"


def test_build_model_reservoir_does_not_call_recurrent_encoder(monkeypatch):
    from pdm.models import FlyConnectomeReservoir, RecurrentEncoder, build_model

    def _boom(*_a, **_k):
        raise AssertionError("RecurrentEncoder must not see reservoir architecture strings")

    monkeypatch.setattr("pdm.models.recurrent.RecurrentEncoder", _boom)
    graph = _cycle_graph(8)
    model = build_model(
        architecture="random_reservoir",
        input_size=4,
        graph=graph,
        n_nodes=8,
        seed=0,
    )
    assert model.n_nodes == 8
    monkeypatch.setattr("pdm.models.recurrent.RecurrentEncoder", RecurrentEncoder)
    net = build_model(architecture="gru", input_size=4, hidden_size=8)
    assert net.encoder.architecture == "gru"
    fly = build_model(
        architecture="fly_connectome_reservoir",
        input_size=4,
        graph=graph,
        n_nodes=8,
        seed=1,
    )
    assert isinstance(fly, FlyConnectomeReservoir)


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
    # y_norm = z @ w_true is exactly in the column space; ridge with tiny alpha should recover w_true closely
    y_norm = z @ w_true + 0.5  # positive offset so relu doesn't mask all output
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


def _prepare_tiny_bearings(tmp_path, monkeypatch, tiny_bearing_tables):
    from pdm.data.prepare import write_processed_version
    from pdm.splits import bearings_split

    features, units = tiny_bearing_tables
    split = bearings_split(units)
    processed_root = tmp_path / "processed" / "bearings"
    runs_root = tmp_path / "runs"
    monkeypatch.setattr("pdm.data.prepare.dataset_processed", lambda _id: processed_root)
    monkeypatch.setattr("pdm.paths.dataset_runs", lambda _id: runs_root / _id)
    monkeypatch.setattr("pdm.train.dataset_runs", lambda _id: runs_root / _id)
    write_processed_version(
        "bearings", features, units, split, sensor_note="n", cfg={}, processed_root=processed_root
    )
    return features, units, split, runs_root


def _train_tiny_reservoir(tmp_path, monkeypatch, tiny_bearing_tables, **kwargs):
    _prepare_tiny_bearings(tmp_path, monkeypatch, tiny_bearing_tables)
    params = {
        "dataset_id": "bearings",
        "architecture": "fly_connectome_reservoir",
        "smoke": True,
        "max_epochs": 1,
        "history_length": 3,
        "device_pref": "cpu",
        "max_windows_per_unit": 8,
        "n_nodes": 8,
        "readout": "ridge",
    }
    params.update(kwargs)
    return run_training(**params)


def test_frozen_reservoir_weights_not_updated():
    graph = _cycle_graph(8)
    model = FlyConnectomeReservoir(graph, input_size=3, head="rul", seed=1)
    w_before = model.W_res.detach().cpu().clone()
    win_before = model.W_in.detach().cpu().clone()
    b_before = model.b_res.detach().cpu().clone()
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable
    assert not model.W_res.requires_grad
    opt = torch.optim.AdamW(trainable, lr=0.05)
    x = torch.randn(4, 5, 3)
    y = torch.rand(4)
    pred = model(x)
    loss = torch.nn.functional.smooth_l1_loss(pred, y)
    loss.backward()
    opt.step()
    assert torch.equal(model.W_res.detach().cpu(), w_before)
    assert torch.equal(model.W_in.detach().cpu(), win_before)
    assert torch.equal(model.b_res.detach().cpu(), b_before)
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in trainable)


def test_filters_ridge_raises_before_targets(monkeypatch):
    consumed = {"targets": False, "processed": False}

    def _no_processed(*_a, **_k):
        consumed["processed"] = True
        raise AssertionError("load_processed must not run before the filters+ridge raise")

    class _SpyDataset(UnitWindowDataset):
        def __getitem__(self, i):
            consumed["targets"] = True
            return super().__getitem__(i)

    monkeypatch.setattr("pdm.train.load_processed", _no_processed)
    monkeypatch.setattr("pdm.train.UnitWindowDataset", _SpyDataset)
    with pytest.raises(ValueError, match="Ridge readout is not supported for filters"):
        run_training(
            "filters",
            architecture="fly_connectome_reservoir",
            readout="ridge",
            n_nodes=8,
        )
    assert consumed["targets"] is False
    assert consumed["processed"] is False


def test_filters_censoring_not_rul_zero():
    lam = torch.tensor([1.2, 1.2])
    k = torch.tensor([1.8, 1.8])
    duration = torch.tensor([10.0, 10.0])
    nll_censored = weibull_nll(duration, torch.tensor([0.0, 0.0]), lam, k, 1.0)
    nll_zero_event = weibull_nll(torch.zeros_like(duration), torch.tensor([1.0, 1.0]), lam, k, 1.0)
    assert not torch.allclose(nll_censored, nll_zero_event)
    windows = pd.DataFrame(
        {
            "unit_id": ["U1", "U1"],
            "start_index": [0, 1],
            "end_index": [2, 3],
            "target_rul_s": [float("nan"), float("nan")],
            "duration_s": [12.0, 18.0],
            "event": [0, 0],
        }
    )
    features = pd.DataFrame(
        {
            "unit_id": ["U1"] * 5,
            "timestamp_s": [0.0, 1.0, 2.0, 3.0, 4.0],
            "f0": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    ds = UnitWindowDataset(features, windows, ["f0"], "filters", time_scale_s=1.0)
    for i in range(len(ds)):
        item = ds[i]
        assert int(item["event"].item()) == 0
        assert not torch.isfinite(item["target"])
        assert float(item["duration"]) > 0


def test_ridge_uses_forward_states_kernel(tmp_path, monkeypatch, tiny_bearing_tables):
    import pdm.models.reservoir as res_mod

    calls = {"n": 0}
    orig = res_mod.forward_states

    def _wrapped(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(res_mod, "forward_states", _wrapped)
    rec = _train_tiny_reservoir(tmp_path, monkeypatch, tiny_bearing_tables)
    assert rec["status"] == "completed"
    assert calls["n"] > 0


def test_ridge_no_backward(tmp_path, monkeypatch, tiny_bearing_tables):
    called = {"n": 0}
    orig = torch.Tensor.backward

    def _spy(self, *args, **kwargs):
        called["n"] += 1
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "backward", _spy)
    rec = _train_tiny_reservoir(tmp_path, monkeypatch, tiny_bearing_tables)
    assert rec["status"] == "completed"
    assert called["n"] == 0


def test_ridge_writes_artifacts(tmp_path, monkeypatch, tiny_bearing_tables):
    rec = _train_tiny_reservoir(tmp_path, monkeypatch, tiny_bearing_tables)
    rdir = Path(rec["dir"])
    assert (rdir / "best.pt").exists()
    assert (rdir / "last.pt").exists()
    assert (rdir / "status.json").exists()
    assert (rdir / "experiment_snapshot.json").exists()
    cdir = rdir / "connectome"
    assert (cdir / "provenance.json").exists()
    assert (cdir / "graph.json").exists()
    assert (cdir / "weights.npz").exists()
    weights = np.load(cdir / "weights.npz")
    assert "W_in" in weights.files and "W_res" in weights.files and "b_res" in weights.files
    snap = json.loads((rdir / "experiment_snapshot.json").read_text())
    assert snap["model"]["architecture"] == "fly_connectome_reservoir"
    assert snap["model"]["reservoir"]["n_nodes"] == 8
    status = json.loads((rdir / "status.json").read_text())
    assert status["status"] == "completed"
    rec_rand = _train_tiny_reservoir(
        tmp_path,
        monkeypatch,
        tiny_bearing_tables,
        architecture="random_reservoir",
    )
    prov = json.loads(
        (Path(rec_rand["dir"]) / "connectome" / "provenance.json").read_text()
    )
    assert prov["graph_mode"] == GRAPH_MODE_RANDOM_REWIRE
    assert prov.get("parent_graph_hash")


def test_ridge_stop_flag_sets_cancelled(tmp_path, monkeypatch, tiny_bearing_tables):
    rec = _train_tiny_reservoir(
        tmp_path,
        monkeypatch,
        tiny_bearing_tables,
        should_stop=lambda: True,
    )
    assert rec["status"] == "cancelled"
    status = json.loads((Path(rec["dir"]) / "status.json").read_text())
    assert status["status"] == "cancelled"
    assert status["status"] not in {"completed", "stopped"}


def test_load_trained_model_missing_weights_npz_raises(tmp_path, monkeypatch, tiny_bearing_tables):
    rec = _train_tiny_reservoir(tmp_path, monkeypatch, tiny_bearing_tables)
    rdir = Path(rec["dir"])
    (rdir / "connectome" / "weights.npz").unlink()
    with pytest.raises(FileNotFoundError, match="will not rebuild from seed"):
        load_trained_model(rdir, device="cpu")


def test_n_nodes_mismatch_vs_weights_npz_raises(tmp_path, monkeypatch, tiny_bearing_tables):
    rec = _train_tiny_reservoir(tmp_path, monkeypatch, tiny_bearing_tables, n_nodes=16)
    rdir = Path(rec["dir"])
    n8 = 8
    np.savez(
        rdir / "connectome" / "weights.npz",
        W_in=np.zeros((n8, 2), dtype=np.float32),
        W_res=np.eye(n8, dtype=np.float32),
        b_res=np.zeros(n8, dtype=np.float32),
    )
    with pytest.raises(ValueError, match="n_nodes mismatch"):
        load_trained_model(rdir, device="cpu")


def test_dataset_checkpoint_isolation():
    base = {
        "dataset_id": "bearings",
        "architecture": "fly_connectome_reservoir",
        "history_length": 20,
        "hidden_size": 64,
        "recurrent_layers": 1,
        "head": "rul",
        "feature_names": ["horizontal_rms"],
        "time_scale_s": 60.0,
        "feature_pipeline_version": "v2_raw_first",
        "categorical_maps_fingerprint": "abc",
        "split_hash": "deadbeefdeadbeef",
        "n_nodes": 8,
        "graph_mode": "synthetic_fixture",
        "graph_hash": "aaa",
        "state_mode": "window_reset",
        "leak": 0.2,
        "spectral_radius": 0.9,
        "input_scale": 0.1,
        "seed": 42,
        "readout": "ridge",
    }
    rand = {**base, "architecture": "random_reservoir", "graph_mode": "random_rewire"}
    gru = {**base, "architecture": "gru"}
    filters = {**base, "dataset_id": "filters", "head": "weibull"}
    assert not checkpoints_compatible(base, rand)
    assert not checkpoints_compatible(base, gru)
    assert not checkpoints_compatible(base, filters)
    other_hash = {**base, "graph_hash": "bbb"}
    assert not checkpoints_compatible(base, other_hash)
    assert checkpoints_compatible(base, dict(base))


def test_split_preprocess_isolation(tmp_path, monkeypatch, tiny_bearing_tables):
    from pdm.preprocessing import fit_preprocessor

    features, units, split, _runs = _prepare_tiny_bearings(tmp_path, monkeypatch, tiny_bearing_tables)
    rec = run_training(
        "bearings",
        architecture="fly_connectome_reservoir",
        smoke=True,
        max_epochs=1,
        history_length=3,
        device_pref="cpu",
        max_windows_per_unit=8,
        n_nodes=8,
        readout="ridge",
        seed=42,
    )
    rdir = Path(rec["dir"])
    w_in = np.load(rdir / "connectome" / "weights.npz")["W_in"].copy()
    model, prep, _meta = load_trained_model(rdir, device="cpu")
    np.testing.assert_allclose(model.W_in.detach().cpu().numpy(), w_in)
    cfg = load_dataset_config("bearings")
    mutated = features.copy()
    mutated.loc[mutated["unit_id"].isin(split["test"]), "horizontal_rms"] = 999.0
    prep2, _ = fit_preprocessor("bearings", mutated, units, split, cfg)
    np.testing.assert_allclose(prep.scaler_mean, prep2.scaler_mean)
    rec_b = run_training(
        "bearings",
        architecture="fly_connectome_reservoir",
        smoke=True,
        max_epochs=1,
        history_length=3,
        device_pref="cpu",
        max_windows_per_unit=8,
        n_nodes=8,
        readout="ridge",
        seed=42,
    )
    w_b = np.load(Path(rec_b["dir"]) / "connectome" / "weights.npz")["W_res"]
    np.testing.assert_allclose(np.load(rdir / "connectome" / "weights.npz")["W_res"], w_b)


def test_predictor_works_with_reservoir(tmp_path, monkeypatch, tiny_bearing_tables):
    rec = _train_tiny_reservoir(tmp_path, monkeypatch, tiny_bearing_tables)
    rdir = Path(rec["dir"])
    model, prep, meta = load_trained_model(rdir, device="cpu")
    predictor = Predictor(model, prep, history_length=int(meta["history_length"]), device="cpu")
    features, units = tiny_bearing_tables
    uid = units["unit_id"].iloc[0]
    hist = features[features["unit_id"] == uid].sort_values("timestamp_s").iloc[: int(meta["history_length"])]
    hist = hist.copy()
    hist.attrs["raw_features"] = True
    out = predictor.predict_from_history(hist)
    assert out["status"] in {"ok", "Collecting history"}
    if out["status"] == "ok":
        assert out["predicted_rul_s"] is not None
        assert out["predicted_rul_s"] >= 0


def _bearings_prep(feature_names=None, time_scale_s: float = 1.0) -> Preprocessor:
    names = list(feature_names or BEARINGS_RAW_NUMERIC_COLUMNS)
    return Preprocessor(
        feature_names=names,
        log1p_features=[],
        scaler_mean=[0.0] * len(names),
        scaler_scale=[1.0] * len(names),
        time_scale_s=float(time_scale_s),
        fill_values={name: 0.0 for name in names},
        dataset_id="bearings",
    )


def _unit_history(features: pd.DataFrame, n: int, unit_id: str | None = None) -> tuple[pd.DataFrame, str]:
    uid = str(unit_id or features["unit_id"].iloc[0])
    hist = features[features["unit_id"] == uid].sort_values("timestamp_s").iloc[: int(n)].copy()
    hist.attrs["raw_features"] = True
    return hist, uid


def _tiny_fly(prep: Preprocessor, *, n_nodes: int = 8, head: str = "rul", seed: int = 0, leak: float = 0.2, **kwargs):
    graph = _cycle_graph(n_nodes)
    return FlyConnectomeReservoir(
        graph,
        input_size=len(prep.feature_names),
        head=head,
        seed=seed,
        leak=leak,
        time_scale_s=float(prep.time_scale_s),
        n_nodes=n_nodes,
        **kwargs,
    )


def test_graph_json_stores_node_order(tmp_path):
    from pdm.io_util import read_json
    from pdm.train import _saved_node_order, _write_connectome_artifacts

    order = ["c", "a", "b"]
    edges = [
        {"src": "c", "dst": "a", "weight": 2.0},
        {"src": "a", "dst": "b", "weight": 1.0},
        {"src": "b", "dst": "c", "weight": 1.0},
    ]
    graph = graph_from_edges(edges, order)
    model = FlyConnectomeReservoir(graph, input_size=2, seed=0, node_order=order)
    _write_connectome_artifacts(tmp_path, model)
    payload = read_json(tmp_path / "connectome" / "graph.json")
    assert payload["node_order"] == order
    assert payload["nodes"][:3] == order
    assert _saved_node_order(tmp_path, payload) == order
    restored = FlyConnectomeReservoir(
        graph_from_payload(payload),
        input_size=2,
        frozen_weights=(
            model.W_in.detach().cpu().numpy(),
            model.W_res.detach().cpu().numpy(),
            model.b_res.detach().cpu().numpy(),
        ),
        node_order=_saved_node_order(tmp_path, payload),
        n_nodes=3,
        provenance=model.provenance,
    )
    assert restored.node_order == order
    np.testing.assert_allclose(
        restored.W_res.detach().cpu().numpy(),
        model.W_res.detach().cpu().numpy(),
    )


def test_predict_trace_parity(tiny_bearing_tables):
    features, _units = tiny_bearing_tables
    prep = _bearings_prep(time_scale_s=4.0)
    model = _tiny_fly(prep, seed=1)
    hist, uid = _unit_history(features, 5)
    trace = predict_with_trace(hist, uid, model, prep, history_length=5)
    assert trace["status"] == "predicted"
    x = torch.from_numpy(np.ascontiguousarray(trace["inputs"])).unsqueeze(0)
    with torch.no_grad():
        pred = model.predicted_rul_s(x)
        states = model.forward_states(x)
        raw = model.forward_raw(states[:, -1, :], x[:, -1, :])
    np.testing.assert_allclose(
        float(trace["predicted_rul_s"]),
        float(pred.detach().cpu().reshape(-1)[0]),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(trace["raw_prediction"]).reshape(-1),
        raw.detach().cpu().numpy().reshape(-1),
        rtol=1e-6,
        atol=1e-6,
    )
    live = Predictor(model, prep, history_length=5).predict_from_history(hist)
    np.testing.assert_allclose(
        float(trace["predicted_rul_s"]),
        float(live["predicted_rul_s"]),
        rtol=1e-6,
        atol=1e-6,
    )


def test_predict_and_trace_share_update_function(tiny_bearing_tables, monkeypatch):
    from pdm.models.reservoir import forward_states
    from pdm.visualization import trace as trace_mod

    assert trace_mod.leaky_forward_states is forward_states
    source = Path(trace_mod.__file__).read_text(encoding="utf-8")
    assert "tanh" not in source
    assert "forward_states" in inspect.getsource(trace_mod.predict_with_trace)
    features, _units = tiny_bearing_tables
    prep = _bearings_prep()
    model = _tiny_fly(prep, seed=2)
    hist, uid = _unit_history(features, 5)
    calls = {"n": 0}
    orig = LeakyESN.forward_states

    def _wrapped(self, *args, **kwargs):
        calls["n"] += 1
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(LeakyESN, "forward_states", _wrapped)
    trace = predict_with_trace(hist, uid, model, prep, history_length=5)
    assert calls["n"] >= 1
    x = torch.from_numpy(np.ascontiguousarray(trace["inputs"])).unsqueeze(0)
    with torch.no_grad():
        states = orig(model, x)
    np.testing.assert_array_equal(
        trace["states"],
        states.squeeze(0).detach().cpu().numpy().astype(np.float32),
    )


def test_no_future_frames(tiny_bearing_tables):
    features, _units = tiny_bearing_tables
    prep = _bearings_prep()
    model = _tiny_fly(prep, seed=3)
    hist, uid = _unit_history(features, 8)
    prefix = hist.iloc[:5].copy()
    prefix.attrs["raw_features"] = True
    p1 = predict_with_trace(prefix, uid, model, prep, history_length=5)
    mutated = hist.copy()
    mutated.loc[mutated.index[5:], "horizontal_rms"] = 9999.0
    still_prefix = mutated.iloc[:5].copy()
    still_prefix.attrs["raw_features"] = True
    p2 = predict_with_trace(still_prefix, uid, model, prep, history_length=5)
    np.testing.assert_allclose(p1["states"], p2["states"], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(p1["contributions"]["raw"], p2["contributions"]["raw"], atol=1e-6)
    np.testing.assert_allclose(p1["predicted_rul_s"], p2["predicted_rul_s"], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(p1["raw_prediction"], p2["raw_prediction"], atol=1e-6)


def test_window_reset(tiny_bearing_tables):
    features, _units = tiny_bearing_tables
    prep = _bearings_prep()
    model = _tiny_fly(prep, seed=4, leak=0.2)
    hist, uid = _unit_history(features, 8)
    w1 = hist.iloc[:4].copy()
    w1.attrs["raw_features"] = True
    both = hist.iloc[:8].copy()
    both.attrs["raw_features"] = True
    t1 = predict_with_trace(w1, uid, model, prep, history_length=4)
    t2 = predict_with_trace(both, uid, model, prep, history_length=4)
    cont = predict_with_trace(both, uid, model, prep, history_length=8)
    x0 = torch.from_numpy(np.ascontiguousarray(t2["inputs"][:1])).unsqueeze(0)
    with torch.no_grad():
        from_zeros = model.forward_states(x0).squeeze(0)[0].detach().cpu().numpy()
    np.testing.assert_allclose(t2["states"][0], from_zeros, rtol=1e-5, atol=1e-5)
    assert not np.allclose(t2["states"][0], cont["states"][4], atol=1e-4)
    assert not np.allclose(t2["states"][0], t1["states"][-1], atol=1e-4)


def test_edge_drive_previous_state(tiny_bearing_tables):
    features, _units = tiny_bearing_tables
    prep = _bearings_prep(feature_names=["horizontal_rms"])
    graph = graph_from_edges([{"src": "src", "dst": "dst", "weight": 4.0}], nodes=["src", "dst"])
    model = FlyConnectomeReservoir(
        graph,
        input_size=1,
        seed=0,
        node_order=["src", "dst"],
        leak=1.0,
        time_scale_s=1.0,
    )
    with torch.no_grad():
        model.W_in.zero_()
        model.W_in[0, 0] = 1.0
        model.b_res.zero_()
    hist, uid = _unit_history(features, 2)
    hist = hist.copy()
    hist["horizontal_rms"] = [1.0, 0.0]
    hist.attrs["raw_features"] = True
    trace = predict_with_trace(hist, uid, model, prep, history_length=2)
    src_i = model.node_order.index("src")
    dst_i = model.node_order.index("dst")
    w = float(model.W_res[dst_i, src_i])
    assert w != 0.0
    assert abs(float(trace["inputs"][1, 0])) < 1e-8
    assert abs(float(trace["states"][0, dst_i])) < 1e-5
    expected = float(np.tanh(w * float(trace["states"][0, src_i])))
    np.testing.assert_allclose(float(trace["states"][1, dst_i]), expected, rtol=1e-5, atol=1e-5)


def test_contribution_sum(tiny_bearing_tables):
    features, _units = tiny_bearing_tables
    prep = _bearings_prep()
    hist, uid = _unit_history(features, 5)
    for head in ("rul", "weibull"):
        model = _tiny_fly(prep, head=head, seed=5)
        trace = predict_with_trace(hist, uid, model, prep, history_length=5)
        contrib = trace["contributions"]
        recon = (
            contrib["intercept"][None, :]
            + contrib["input"].sum(axis=1)
            + contrib["neuron"].sum(axis=1)
        )
        np.testing.assert_allclose(recon, contrib["raw"], atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(
            recon[-1],
            np.asarray(trace["raw_prediction"]).reshape(-1),
            atol=1e-5,
            rtol=1e-5,
        )


def test_raw_vs_display_postprocess(tiny_bearing_tables):
    features, _units = tiny_bearing_tables
    prep = _bearings_prep(time_scale_s=1.0)
    model = _tiny_fly(prep, seed=6)
    with torch.no_grad():
        model.readout.W_x.zero_()
        model.readout.W_u.zero_()
        model.readout.b.fill_(1.5)
    hist, uid = _unit_history(features, 5)
    model.time_scale_s = 2.0
    t1 = predict_with_trace(hist, uid, model, prep, history_length=5)
    raw = float(np.asarray(t1["raw_prediction"]).reshape(-1)[0])
    np.testing.assert_allclose(t1["predicted_rul_s"], max(0.0, raw) * 2.0, rtol=1e-5, atol=1e-5)
    contrib = t1["contributions"]
    recon = contrib["intercept"] + contrib["input"][-1].sum(axis=0) + contrib["neuron"][-1].sum(axis=0)
    np.testing.assert_allclose(recon.reshape(-1), t1["raw_prediction"].reshape(-1), atol=1e-5)
    model.time_scale_s = 8.0
    t2 = predict_with_trace(hist, uid, model, prep, history_length=5)
    np.testing.assert_allclose(t2["raw_prediction"], t1["raw_prediction"], atol=1e-6)
    np.testing.assert_allclose(t2["predicted_rul_s"], max(0.0, raw) * 8.0, rtol=1e-5, atol=1e-5)
    assert t2["predicted_rul_s"] != t1["predicted_rul_s"]


def test_trace_artifact_reload(tiny_bearing_tables, tmp_path):
    features, _units = tiny_bearing_tables
    prep = _bearings_prep(time_scale_s=3.0)
    model = _tiny_fly(prep, seed=7)
    hist, uid = _unit_history(features, 5)
    trace = predict_with_trace(hist, uid, model, prep, history_length=5)
    dest = tmp_path / "runs" / "bearings" / "run1" / "traces" / uid
    save_trace(
        dest,
        unit_id=uid,
        run_id="run1",
        dataset_id="bearings",
        architecture=model.architecture,
        graph_hash=model.graph_hash,
        n_nodes=model.n_nodes,
        history_length=5,
        graph_mode=model.graph_mode,
        is_synthetic=bool(model.is_synthetic),
        states=trace["states"],
        inputs=trace["inputs"],
        contributions=trace["contributions"],
        frame_map=trace["frame_map"],
        node_order=trace["node_order"],
        predicted_rul_s=trace["predicted_rul_s"],
        raw_prediction=trace["raw_prediction"],
        status=trace["status"],
        time_scale_s=model.time_scale_s,
        head=model.head_type,
    )
    loaded = load_trace(dest)
    np.testing.assert_allclose(loaded["states"], trace["states"], atol=1e-6)
    np.testing.assert_allclose(loaded["inputs"], trace["inputs"], atol=1e-6)
    np.testing.assert_allclose(loaded["contributions"]["raw"], trace["contributions"]["raw"], atol=1e-6)
    np.testing.assert_allclose(loaded["contributions"]["neuron"], trace["contributions"]["neuron"], atol=1e-6)
    np.testing.assert_allclose(loaded["raw_prediction"], trace["raw_prediction"], atol=1e-6)
    live = Predictor(model, prep, history_length=5).predict_from_history(hist)
    np.testing.assert_allclose(
        float(loaded["predicted_rul_s"]),
        float(live["predicted_rul_s"]),
        rtol=1e-5,
        atol=1e-5,
    )
    meta = json.loads((dest / "meta.json").read_text(encoding="utf-8"))
    assert meta["is_synthetic"] is True
    assert meta["n_nodes"] == 8


def test_predict_with_trace_gru_note(tiny_bearing_tables):
    from pdm.models import PDMNet

    features, _units = tiny_bearing_tables
    prep = _bearings_prep()
    hist, uid = _unit_history(features, 5)
    gru = PDMNet(len(prep.feature_names), hidden_size=8, architecture="gru", head="rul", time_scale_s=1.0)
    out = predict_with_trace(hist, uid, gru, prep, history_length=5)
    assert out["status"] == "traces require reservoir model"
    pred = Predictor(gru, prep, history_length=5).predict_from_history(hist, with_trace=True)
    assert pred.get("trace_note") == "traces require reservoir model"
    assert pred["status"] in {"ok", "Collecting history", "No valid prediction"}
    assert "states" not in pred or pred["status"] != "predicted"


def test_lazy_skips_per_neuron(tiny_bearing_tables):
    features, _units = tiny_bearing_tables
    prep = _bearings_prep()
    model = _tiny_fly(prep, seed=8)
    hist, uid = _unit_history(features, 5)
    full = predict_with_trace(hist, uid, model, prep, history_length=5, lazy=False)
    lazy = predict_with_trace(hist, uid, model, prep, history_length=5, lazy=True)
    assert full["contributions"]["neuron"].shape[1] == model.n_nodes
    assert lazy["contributions"]["neuron"].shape[1] == 0
    np.testing.assert_allclose(full["predicted_rul_s"], lazy["predicted_rul_s"], rtol=1e-6, atol=1e-6)


def test_trace_job_stop_flag_sets_cancelled(monkeypatch, tmp_path):
    from pdm.worker import read_status, run_job, stop_path

    def fake_trace(*_a, **_k):
        stop_path().write_text("stop\n", encoding="utf-8")
        return {"status": "cancelled"}

    wdir = tmp_path / "worker"
    wdir.mkdir()
    monkeypatch.setattr("pdm.worker.worker_dir", lambda: wdir)
    monkeypatch.setattr("pdm.visualization.trace.run_trace_job", fake_trace)
    run_job({"kind": "trace", "dataset_id": "bearings", "run_id": "r1", "unit_id": "U1"})
    status = read_status()["status"]
    assert status == "cancelled"
    assert status not in {"completed", "stopped"}


def _comparison_run(**overrides) -> dict:
    row = {
        "run_id": "fly_real",
        "architecture": "fly_connectome_reservoir",
        "graph_mode": "real_connectome",
        "is_synthetic": False,
        "parent_graph_hash": None,
        "graph_hash": "flyhash",
        "n_nodes": 8,
        "split_hash": "splitA",
        "dataset_id": "bearings",
        "best_metric": 1.0,
        "head": "rul",
        "smoke": False,
        "leak": 0.2,
        "spectral_radius": 0.9,
        "state_mode": "window_reset",
        "history_length": 5,
    }
    row.update(overrides)
    return row


def test_synthetic_excluded_from_biological_comparison():
    from pdm.visualization.comparison import filter_synthetic_runs

    real = _comparison_run(run_id="real", is_synthetic=False, graph_mode="real_connectome")
    synth = _comparison_run(
        run_id="synth",
        is_synthetic=True,
        graph_mode="synthetic_fixture",
        graph_hash="synthhash",
    )
    real_runs, synthetic_runs = filter_synthetic_runs([real, synth])
    real_ids = {r["run_id"] for r in real_runs}
    synth_ids = {r["run_id"] for r in synthetic_runs}
    assert "synth" not in real_ids
    assert "synth" in synth_ids
    assert "real" in real_ids
    assert "real" not in synth_ids


def test_matched_control_check():
    from pdm.visualization.comparison import check_matched_control

    fly = _comparison_run()
    matched_random = _comparison_run(
        run_id="rand_ok",
        architecture="random_reservoir",
        graph_mode="random_rewire",
        graph_hash="randhash",
        parent_graph_hash="flyhash",
    )
    ok, warn = check_matched_control(fly, matched_random)
    assert ok is True
    assert warn == ""

    mismatched = dict(matched_random)
    mismatched["parent_graph_hash"] = "otherhash"
    ok2, warn2 = check_matched_control(fly, mismatched)
    assert ok2 is False
    assert "parent_graph_hash" in warn2


def test_comparison_table_labels_synthetic():
    from pdm.visualization.comparison import (
        SYNTHETIC_LABEL,
        SYNTHETIC_SECTION,
        build_comparison_table,
        comparison_sections,
    )

    runs = [
        _comparison_run(run_id="gru1", architecture="gru", graph_mode="", n_nodes=None),
        _comparison_run(run_id="fly_real"),
        _comparison_run(
            run_id="fly_synth",
            graph_mode="synthetic_fixture",
            is_synthetic=True,
            graph_hash="synthhash",
        ),
        _comparison_run(
            run_id="filters_gru",
            architecture="gru",
            dataset_id="filters",
            graph_mode="",
            n_nodes=None,
        ),
    ]
    table = build_comparison_table("bearings", runs=runs)
    assert "filters_gru" not in set(table["run_id"].astype(str))
    real, synth = comparison_sections(table)
    assert "fly_synth" not in set(real["run_id"].astype(str))
    assert "fly_synth" in set(synth["run_id"].astype(str))
    assert "gru1" in set(real["run_id"].astype(str))
    synth_labels = " ".join(synth["label"].astype(str))
    assert SYNTHETIC_LABEL in synth_labels
    assert all(row == SYNTHETIC_SECTION for row in synth["section"].tolist())
    assert SYNTHETIC_SECTION not in set(real["section"].tolist())


def test_alert_jump_uses_prefix():
    from pdm.visualization.explorer import (
        causal_prefix_rows,
        slice_trace_to_alert,
        stored_alert_prediction,
    )

    frame_map = [
        {"frame_index": 0, "timestamp_s": 10.0, "unit_id": "U1"},
        {"frame_index": 1, "timestamp_s": 20.0, "unit_id": "U1"},
        {"frame_index": 2, "timestamp_s": 30.0, "unit_id": "U1"},
    ]
    states = np.array([[0.0, 0.0], [1.0, 1.0], [9.0, 9.0]], dtype=float)
    trace = {
        "frame_map": frame_map,
        "states": states,
        "predicted_rul_s": 99.0,
        "node_order": ["0", "1"],
    }
    episode = {
        "unit_id": "U1",
        "timestamp_s": 20.0,
        "predicted_rul_s": 12.5,
        "type": "horizon_warning",
    }
    sliced = slice_trace_to_alert(trace, episode)
    assert all(float(frm["timestamp_s"]) <= 20.0 for frm in sliced["frame_map"])
    assert sliced["states"].shape[0] == 2
    assert 30.0 not in [float(frm["timestamp_s"]) for frm in sliced["frame_map"]]
    assert sliced["predicted_rul_s"] == 12.5
    assert stored_alert_prediction(episode) == 12.5
    meas = pd.DataFrame(
        {"unit_id": ["U1", "U1", "U1"], "timestamp_s": [10.0, 20.0, 30.0], "x": [1, 2, 3]}
    )
    prefix = causal_prefix_rows(meas, 20.0)
    assert float(prefix["timestamp_s"].max()) <= 20.0
    assert 30.0 not in prefix["timestamp_s"].tolist()


def test_list_runs_exposes_graph_identity(tmp_path, monkeypatch):
    from pdm.experiments import list_runs
    from pdm.io_util import atomic_write_json

    monkeypatch.setattr("pdm.experiments.dataset_runs", lambda ds: tmp_path / ds)
    rdir = tmp_path / "bearings" / "bearings_fly_demo"
    rdir.mkdir(parents=True)
    (rdir / "last.pt").write_bytes(b"ckpt")
    atomic_write_json(
        rdir / "status.json",
        {"status": "completed", "run_id": "bearings_fly_demo", "smoke": True},
    )
    atomic_write_json(
        rdir / "experiment_snapshot.json",
        {
            "model": {
                "architecture": "fly_connectome_reservoir",
                "head": "rul",
                "history_length": 5,
                "reservoir": {
                    "graph_mode": "synthetic_fixture",
                    "n_nodes": 8,
                    "graph_hash": "abc",
                    "leak": 0.2,
                    "spectral_radius": 0.9,
                    "state_mode": "window_reset",
                },
            }
        },
    )
    (rdir / "connectome").mkdir()
    atomic_write_json(
        rdir / "connectome" / "provenance.json",
        {
            "graph_mode": "synthetic_fixture",
            "is_synthetic": True,
            "graph_hash": "abc",
            "parent_graph_hash": None,
        },
    )
    atomic_write_json(rdir / "dataset_fingerprint.json", {"split_hash": "deadbeefdeadbeef"})
    rows = list_runs("bearings")
    assert len(rows) == 1
    rec = rows[0]
    assert rec["architecture"] == "fly_connectome_reservoir"
    assert rec["graph_mode"] == "synthetic_fixture"
    assert rec["is_synthetic"] is True
    assert rec["graph_hash"] == "abc"
    assert rec["split_hash"] == "deadbeefdeadbeef"
    assert rec["n_nodes"] == 8


def test_demo_instructions_use_n_nodes_8():
    from pdm.paths import project_root
    from pdm.visualization.demo import get_demo_instructions

    text = get_demo_instructions()
    assert "--arch fly_connectome_reservoir" in text
    assert "--smoke" in text
    assert "--n-nodes 8" in text
    readme = (project_root() / "README.md").read_text(encoding="utf-8")
    assert "--arch fly_connectome_reservoir --smoke --n-nodes 8" in readme
    assert "not a biological connectome" in readme
    demo_doc = (project_root() / "docs" / "fly_connectome_demo.md").read_text(encoding="utf-8")
    assert "--n-nodes 8" in demo_doc
    assert "quality benchmark" in demo_doc.lower()


def test_cli_train_help_has_source_path_and_n_nodes(capsys):
    from pdm.cli import main

    with pytest.raises(SystemExit) as exited:
        main(["train", "--help"])
    assert exited.value.code == 0
    text = capsys.readouterr().out
    assert "--n-nodes" in text
    assert "--source-path" in text
    assert "--graph-mode" in text


def test_acceptance_test_names_present():
    """Subtask 06 gate: numbered tests 1–14 and named extras stay grep-able."""
    from pdm.paths import project_root

    root = project_root() / "tests"
    blob = "\n".join(
        (root / name).read_text(encoding="utf-8")
        for name in (
            "test_reservoir.py",
            "test_neural_explorer.py",
            "test_spec_invariants.py",
            "test_worker_and_app.py",
        )
    )
    required = [
        "test_graph_orientation",
        "test_state_update_hand_calculation",
        "test_n_nodes_mismatch_vs_weights_npz_raises",
        "test_dataset_checkpoint_isolation",
        "test_seed_reproducibility",
        "test_split_preprocess_isolation",
        "test_no_future_frames",
        "test_predict_trace_parity",
        "test_window_reset",
        "test_edge_drive_previous_state",
        "test_contribution_sum",
        "test_raw_vs_display_postprocess",
        "test_filters_censoring_not_rul_zero",
        "test_trace_artifact_reload",
        "test_synthetic_fixture_label",
        "test_synthetic_n_nodes_clamps_not_raises",
        "test_stop_flag_sets_cancelled_not_completed",
        "test_gru_checkpoint_compat_ignores_reservoir_yaml_defaults",
        "test_filters_ridge_raises_before_targets",
        "test_ridge_bearings_no_double_softplus",
        "test_load_trained_model_missing_weights_npz_raises",
        "test_predict_and_trace_share_update_function",
        "test_random_reservoir_parent_graph_hash",
        "test_frozen_reservoir_weights_not_updated",
        "test_ridge_uses_forward_states_kernel",
        "test_ridge_no_backward",
        "test_gru_run_does_not_show_fake_biological_activity",
    ]
    missing = [name for name in required if f"def {name}(" not in blob]
    assert missing == []
