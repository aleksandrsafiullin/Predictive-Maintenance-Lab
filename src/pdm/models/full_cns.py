"""Full MaleCNS computational reservoir, trained from scratch on equipment data.

Biological topology and synapse counts are fixed. The tanh dynamics, sensor
projection and trained readout are engineering choices, not recorded fly activity.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from pdm.connectome.full import anatomical_locations, prepare_full_connectome, preview_edges
from pdm.io_util import atomic_write_json, read_json, sha256_file
from pdm.models.reservoir import LeakyESN


def torch_csr(matrix):
    matrix = matrix.astype(np.float32).tocsr()
    return torch.sparse_csr_tensor(
        torch.from_numpy(matrix.indptr), torch.from_numpy(matrix.indices),
        torch.from_numpy(matrix.data), size=matrix.shape, check_invariants=True,
    )


def scipy_csr(tensor):
    tensor = tensor.detach().cpu()
    return sparse.csr_matrix((tensor.values().numpy(), tensor.col_indices().numpy(),
                              tensor.crow_indices().numpy()), shape=tuple(tensor.shape))


def scaled_operator(counts, target=0.9):
    """Perron iteration on nonnegative log-counts; verified residual, no dense eigensolve."""
    matrix = counts.astype(np.float32)
    matrix.data = np.log1p(matrix.data)
    x = np.full(matrix.shape[0], 1 / np.sqrt(matrix.shape[0]), dtype=np.float32)
    for iteration in range(500):
        y = matrix @ x
        norm = np.linalg.norm(y)
        if norm <= 0 or not np.isfinite(norm):
            raise ValueError("Invalid full CNS recurrent operator")
        x = y / norm
        ax = matrix @ x
        rho = float(x @ ax)
        residual = float(np.linalg.norm(ax - rho * x) / max(rho, 1e-12))
        if residual < 1e-5:
            break
    else:
        raise ValueError(f"Spectral scaling did not converge (residual {residual})")
    matrix.data *= float(target) / rho
    return matrix, {"unscaled_spectral_radius": rho, "spectral_residual": residual,
                    "spectral_iterations": iteration + 1}


class FullCNSReservoir(LeakyESN):
    """Every classified neuron has a state, regardless of anatomical coverage."""

    def __init__(self, w_in, w_res, bias, node_order, provenance, *, leak=0.2,
                 time_scale_s=1.0, head="rul", pool_index=None):
        super().__init__(torch.as_tensor(w_in), torch_csr(w_res), torch.as_tensor(bias),
                         alpha=leak, state_mode="continuous", head=head, time_scale_s=time_scale_s)
        self.node_order = list(map(str, node_order))
        if len(self.node_order) != self.n_nodes or len(set(self.node_order)) != self.n_nodes:
            raise ValueError("Full CNS body order must exactly match the sparse state matrix")
        self.provenance = dict(provenance)
        self.graph_hash = provenance["graph_hash"]
        self.graph_mode, self.is_synthetic = "real_connectome", False
        self.architecture, self.graph = "fly_connectome_reservoir", None
        self.seed = int(provenance.get("seed", 42))
        self.spectral_radius = float(provenance.get("spectral_radius", 0.9))
        self.input_scale = float(provenance.get("input_scale", 0.1))
        self.pool_index = np.asarray(pool_index, np.int32) if pool_index is not None else None
        if self.pool_index is not None:
            if len(self.pool_index) != self.n_nodes or self.pool_index.min() < 0:
                raise ValueError("Invalid full CNS readout groups")
            self.pool_counts = np.bincount(self.pool_index)
            self.n_readout_features = len(self.pool_counts)
            self.pool_operator = sparse.csr_matrix(
                (1.0 / self.pool_counts[self.pool_index], (self.pool_index, np.arange(self.n_nodes))),
                shape=(self.n_readout_features, self.n_nodes), dtype=np.float32,
            )

    def pooled_trajectory(self, inputs, gap_before=None, *, should_stop=None):
        """Pool only the readout, after all 166k recurrent states have computed.

        At most 32 full-state frames are kept in training memory. No N×T history
        or N×N regression matrix. Every neuron belongs to one annotation group.
        """
        values = np.asarray(inputs, np.float32)
        gaps = np.zeros(len(values), bool) if gap_before is None else np.asarray(gap_before, bool)
        starts = np.unique(np.r_[0, np.flatnonzero(gaps)])
        output = []
        with torch.no_grad():
            for start, end in zip(starts, np.r_[starts[1:], len(values)], strict=True):
                x0 = None
                for offset in range(int(start), int(end), 32):
                    if should_stop and should_stop():
                        raise InterruptedError("Full CNS training cancelled")
                    states = self.forward_states(torch.from_numpy(values[offset:min(offset + 32, end)].copy()), x0=x0)
                    x0 = states[-1].clone()
                    output.append((self.pool_operator @ states.numpy().T).T)
        return np.concatenate(output)

    def load_pooled_readout(self, weights):
        n = self.n_readout_features
        # Expand the fitted group means into exactly equivalent neuron weights.
        expanded = np.r_[weights[:n][self.pool_index] / self.pool_counts[self.pool_index], weights[n:]]
        self.readout.load_ridge_vector(expanded)

    def write_artifacts(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(directory / "recurrent.npz", scipy_csr(self.W_res))
        np.savez(directory / "inputs.npz", W_in=self.W_in.numpy(), b_res=self.b_res.numpy(), pool_index=self.pool_index)
        atomic_write_json(directory / "graph.json", self.graph_payload)
        atomic_write_json(directory / "layout.json", self.layout_payload)
        provenance = dict(self.provenance)
        provenance["artifact_hashes"] = {name: sha256_file(directory / name) for name in
                                         ("recurrent.npz", "inputs.npz", "graph.json", "layout.json")}
        atomic_write_json(directory / "provenance.json", provenance)


def build_full_cns(input_size, time_scale_s, *, source_path=None, seed=42, leak=0.2,
                   spectral_radius=0.9, input_scale=0.1, head="rul"):
    frame, counts, provenance = prepare_full_connectome(source_path)
    operator, scaling = scaled_operator(counts, spectral_radius)
    node_order = frame.bodyId.astype(str).tolist()
    positions, kinds = anatomical_locations(frame)
    # Anatomical groups regularize the readout; they never replace neuron states.
    labels = frame.superclass.astype(str) + "/" + frame["class"].fillna("unspecified") + "/" + frame.somaSide.fillna("unknown")
    pool_index, pool_labels = pd.factorize(labels, sort=True)
    provenance = {**provenance, **scaling, "seed": seed, "leak": leak,
                  "spectral_radius": spectral_radius, "input_scale": input_scale,
                  "dynamics": "leaky_tanh_reservoir", "weight_policy": "log1p(synapse_count), global spectral scaling",
                  "readout_policy": "mean state per superclass/class/somaSide; fitted on training bearings",
                  "readout_groups": len(pool_labels), "n_positioned": len(positions),
                  "n_soma": sum(v == "somaLocation" for v in kinds.values()),
                  "n_tosoma": sum(v == "tosomaLocation" for v in kinds.values()),
                  "n_without_coordinates": len(node_order) - len(positions)}
    rng = np.random.default_rng(seed)
    model = FullCNSReservoir(rng.uniform(-input_scale, input_scale, (len(frame), input_size)).astype(np.float32),
                             operator, np.zeros(len(frame), np.float32), node_order, provenance,
                             leak=leak, time_scale_s=time_scale_s, head=head, pool_index=pool_index)
    model.graph_payload = {"format": "full_cns_csr_v1", "node_order": node_order,
                           "n_nodes": len(frame), "n_edges": int(counts.nnz),
                           "n_synapses": int(counts.sum()), "edges": preview_edges(counts, node_order),
                           "edges_role": "display_preview_only", "recurrent_file": "recurrent.npz",
                           "pool_labels": pool_labels.tolist()}
    model.layout_payload = {"method": "anatomical", "positions": positions, "position_kind": kinds,
                            "node_order": node_order, "source": "MaleCNS v1.0 curated soma/tosoma annotations",
                            "classes": frame.superclass.tolist()}
    return model


def load_full_cns(directory, meta):
    directory = Path(directory)
    provenance = read_json(directory / "provenance.json")
    for name, digest in provenance["artifact_hashes"].items():
        if sha256_file(directory / name) != digest:
            raise ValueError(f"Full CNS artifact hash mismatch: {name}")
    graph = read_json(directory / "graph.json")
    if graph["n_nodes"] != meta["n_nodes"] or provenance["graph_hash"] != meta["graph_hash"]:
        raise ValueError("Full CNS checkpoint/graph mismatch")
    with np.load(directory / "inputs.npz", allow_pickle=False) as data:
        model = FullCNSReservoir(data["W_in"], sparse.load_npz(directory / "recurrent.npz"),
                                 data["b_res"], graph["node_order"], provenance,
                                 leak=meta["leak"], time_scale_s=meta["time_scale_s"],
                                 head=meta["head"], pool_index=data["pool_index"])
    if model.n_nodes != meta["n_nodes"]:
        raise ValueError("Full CNS matrix/checkpoint width mismatch")
    model.graph_payload = graph
    model.layout_payload = read_json(directory / "layout.json")
    return model
