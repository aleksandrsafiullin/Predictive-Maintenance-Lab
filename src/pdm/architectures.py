from __future__ import annotations

ARCHITECTURES = {"gru", "lstm", "fly_connectome_reservoir", "random_reservoir"}
RESERVOIR_ARCHITECTURES = {"fly_connectome_reservoir", "random_reservoir"}
RNN_ARCHITECTURES = {"gru", "lstm"}
RESERVOIR_COMPAT_KEYS = (
    "n_nodes",
    "graph_mode",
    "graph_hash",
    "state_mode",
    "leak",
    "spectral_radius",
    "input_scale",
    "seed",
    "readout",
)


def is_reservoir(arch: str) -> bool:
    return str(arch or "").strip().lower() in RESERVOIR_ARCHITECTURES


def is_recurrent_nn(arch: str) -> bool:
    return str(arch or "").strip().lower() in RNN_ARCHITECTURES
