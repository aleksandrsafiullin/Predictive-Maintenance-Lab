"""Windowed numeric signal forecasts using every classified MaleCNS neuron.

The directed graph comes from the public connectome. Input projection, log-count
scaling, leaky tanh dynamics, anatomical pooling and ridge heads are engineering
choices. No synthetic fallback, neuron sampling or learned recurrent rewiring.
"""
from __future__ import annotations

import numpy as np

from pdm.connectome.anatomy import SOMA_ALLOWLIST
from pdm.connectome.sources import default_malemcns_path
from pdm.models.full_cns import build_full_cns, scipy_csr

ENGINE_ID = "full_cns"
ENGINE_LABEL = "Fly brain · Full MaleCNS"


def source_unavailable_reason() -> str | None:
    source = default_malemcns_path()
    required = (source, source.with_name(SOMA_ALLOWLIST[0]))
    if any(not path.is_file() for path in required):
        return ("Full MaleCNS needs the official v1.0 connection weights and body annotations "
                f"in {source.parent}. Download them from https://male-cns.janelia.org/download/. "
                "No synthetic substitute is used.")
    return None


class SignalFullCNS:
    def __init__(self, body):
        self.operator = scipy_csr(body.W_res).copy()
        self.input_weights = body.W_in.detach().numpy().copy()
        self.bias = body.b_res.detach().numpy().copy()
        self.pool = body.pool_operator.copy()
        self.node_order = body.node_order.copy()
        self.leak = float(body.alpha)
        self.provenance = {
            **body.provenance,
            "signal_adapter": "full_malecns_windowed_signal_v1",
            "state_policy": "zero reset per history window; no state across units or gaps",
            "readout_policy": "anatomical group means plus current signal; train-only ridge heads",
            "input_policy": "seeded uniform sensor projection; not biological sensory encoding",
            "neurotransmitter_policy": "not modeled; source synapse counts are unsigned",
        }
        self.design_mean = None
        self.design_std = None
        self.heads = []

    def transform(self, x, *, should_stop=None, status_cb=None):
        """Advance ALL neuron states; pool only after the last observed sample.

        Eight independent windows share a sparse matrix multiplication. Memory
        is O(neurons * 8), never a dense neuron-by-neuron or all-window matrix.
        Resetting each window gives training and prefix replay identical inputs.
        """
        values = np.asarray(x, np.float32)
        if values.ndim != 3 or values.shape[2] != 1 or not np.isfinite(values).all():
            raise ValueError("Full MaleCNS needs finite signal windows [batch, history, 1]")
        design = np.empty((len(values), self.pool.shape[0] + 1), np.float32)
        for start in range(0, len(values), 8):
            batch = values[start:start + 8]
            state = np.zeros((len(self.node_order), len(batch)), np.float32)
            for step in range(values.shape[1]):
                if should_stop and should_stop():
                    raise InterruptedError("Full MaleCNS signal training cancelled")
                drive = self.operator @ state + self.input_weights @ batch[:, step, :].T
                drive += self.bias[:, None]
                state = (1 - self.leak) * state + self.leak * np.tanh(drive)
            design[start:start + len(batch), :-1] = (self.pool @ state).T
            design[start:start + len(batch), -1] = batch[:, -1, 0]
            if status_cb:
                status_cb(start + len(batch), len(values))
        if not np.isfinite(design).all():
            raise FloatingPointError("Nonfinite Full MaleCNS states")
        return design

    def predict_design(self, design):
        if not self.heads:
            raise ValueError("Full MaleCNS signal readout has not been fitted")
        standardized = (design - self.design_mean) / self.design_std
        return np.column_stack([head.predict(standardized) for head in self.heads])

    def predict(self, x, *, should_stop=None):
        return self.predict_design(self.transform(x, should_stop=should_stop))


def build_signal_full_cns(seed: int) -> SignalFullCNS:
    reason = source_unavailable_reason()
    if reason:
        raise FileNotFoundError(reason)
    # Reuse the existing source-hash-verified full graph and spectral scaling.
    return SignalFullCNS(build_full_cns(input_size=1, time_scale_s=1.0, seed=seed))
