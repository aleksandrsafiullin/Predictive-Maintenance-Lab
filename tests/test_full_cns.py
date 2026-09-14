"""Scientific population, directed sparse computation and causal full-CNS replay."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import sparse

from pdm.connectome.full import anatomical_locations, classified_annotations, induced_connectome
from pdm.connectome.morphology import simplify_swc
from pdm.models.full_cns import FullCNSReservoir, load_full_cns
from pdm.models.reservoir import forward_states
from pdm.visualization.simulation import _full_cns_trace, neuron_details


@pytest.fixture
def full_model():
    # asymmetric graph: 11→22, 22→33, 33→11 and 33→33
    matrix = sparse.csr_matrix(np.array([[0, 0, .3], [.2, 0, 0], [0, .4, .1]], np.float32))
    model = FullCNSReservoir(np.array([[.3], [-.2], [.1]], np.float32), matrix,
                             np.zeros(3, np.float32), ["11", "22", "33"],
                             {"graph_hash": "fixture", "n_nodes": 3},
                             pool_index=np.array([0, 0, 1]), time_scale_s=600)
    model.load_pooled_readout(np.array([.2, -.3, .4, 1.0]))
    return model


def test_population_retains_classified_neurons_without_somas_and_all_pair_weights(tmp_path):
    path = tmp_path / "annotations.feather"
    pd.DataFrame({"bodyId": [33, 99, 11, 22], "superclass": ["motor", None, "sensory", "local"],
                  "somaLocation": [None, [1., 2., 3.], [2., 3., 4.], None],
                  "tosomaLocation": [None, None, None, [5., 6., 7.]]}).to_feather(path)
    frame = classified_annotations(path)
    assert frame.bodyId.tolist() == [11, 22, 33]
    positions, kinds = anatomical_locations(frame)
    assert set(positions) == {"11", "22"}
    assert kinds["22"] == "tosomaLocation"
    edges = tmp_path / "edges.feather"
    pd.DataFrame({"body_pre": [11, 22, 33, 33, 99], "body_post": [22, 33, 11, 33, 11],
                  "weight": [1, 2, 3, 4, 100]}).to_feather(edges)
    counts, stats = induced_connectome(edges, frame.bodyId, progress=lambda _: None)
    assert counts.nnz == 4 and counts.sum() == 10
    assert counts[1, 0] == 1 and counts[0, 1] == 0
    assert counts[2, 2] == 4  # Self connections are retained too.
    assert stats == {"source_segment_pairs": 5, "source_synapses": 110}


def test_sparse_dense_equation_pooling_and_expanded_readout_agree(full_model):
    model = full_model
    u = torch.linspace(-1, 2, 40).unsqueeze(1)
    with torch.no_grad():
        actual = model.forward_states(u)
        dense = forward_states(u, model.W_in, model.W_res.to_dense(), model.b_res, model.alpha)
        torch.testing.assert_close(actual, dense)
        pool = model.pooled_trajectory(u.numpy())
        np.testing.assert_allclose(pool[:, 0], actual[:, :2].mean(1), atol=1e-7)
        np.testing.assert_allclose(pool[:, 1], actual[:, 2], atol=1e-7)
        expected = pool @ np.array([.2, -.3]) + u.numpy().ravel() * .4 + 1
        np.testing.assert_allclose(model.forward_raw(actual, u).numpy().ravel(), expected, atol=2e-7)


def test_full_replay_is_bounded_causal_and_inspection_uses_real_sources(full_model):
    model = full_model
    u = np.linspace(.1, 2, 150, dtype=np.float32).reshape(-1, 1)
    ts = np.arange(len(u)) * 60.0
    prefix = _full_cns_trace(u[:70], ts[:70], model, 5, None)
    appended = _full_cns_trace(u, ts, model, 5, prefix)
    fresh = _full_cns_trace(u, ts, model, 5, None)
    np.testing.assert_allclose(appended["states"], fresh["states"], atol=1e-7)
    np.testing.assert_allclose(appended["raw_rul_s"], fresh["raw_rul_s"], atol=1e-4)
    assert appended["states"].shape == (2, 3)
    assert appended["activity_history"].shape == (120, 120)
    assert len(appended["raw_rul_s"]) == 150 and appended["n_history"] == 150
    rewind = _full_cns_trace(u[:70], ts[:70], model, 5, appended)
    np.testing.assert_allclose(rewind["states"], prefix["states"], atol=1e-7)
    with torch.no_grad():
        model.W_res.values().mul_(.5)
    changed = _full_cns_trace(u, ts, model, 5, prefix)
    changed_fresh = _full_cns_trace(u, ts, model, 5, None)
    np.testing.assert_allclose(changed["states"], changed_fresh["states"], atol=1e-7)
    prep = type("Prep", (), {"feature_names": ["sensor"]})()
    _, neighbors = neuron_details(changed, model, prep, 1)
    assert neighbors["Source body ID"].tolist() == ["11"]
    assert neighbors["Recurrent weight"].iloc[0] == pytest.approx(.1)


def test_full_artifact_roundtrip_and_tampering_fail_closed(full_model, tmp_path):
    model = full_model
    model.graph_payload = {"n_nodes": 3, "node_order": model.node_order, "edges": []}
    model.layout_payload = {"positions": {}}
    model.write_artifacts(tmp_path)
    meta = {"n_nodes": 3, "graph_hash": "fixture", "leak": model.alpha, "head": "rul", "time_scale_s": 600}
    restored = load_full_cns(tmp_path, meta)
    assert restored.node_order == model.node_order
    torch.testing.assert_close(restored.W_res.to_dense(), model.W_res.to_dense())
    (tmp_path / "graph.json").write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_full_cns(tmp_path, meta)


def test_swc_simplification_preserves_bifurcations_and_source_coordinates():
    text = "1 1 0 0 0 1 -1\n2 3 0 1 0 1 1\n3 3 0 2 0 1 2\n4 3 1 3 0 1 3\n5 3 -1 3 0 1 3\n"
    segments = simplify_swc(text, spacing=100)
    assert len(segments) == 3
    endpoints = segments.reshape(-1, 3).tolist()
    assert [0, 2, 0] in endpoints
    assert [1, 3, 0] in endpoints and [-1, 3, 0] in endpoints
    assert [0, 0, 0] in endpoints
