"""Full CNS display from saved body order and genuine anatomical coordinates."""
from pathlib import Path

import numpy as np

from pdm.connectome.morphology import morphology_payload
from pdm.io_util import read_json


def full_cns_scene(run_path):
    cdir = Path(run_path) / "connectome"
    graph = read_json(cdir / "graph.json")
    layout = read_json(cdir / "layout.json")
    provenance = read_json(cdir / "provenance.json")
    nodes = graph["node_order"]
    raw = layout["positions"]
    # Missing coordinates stay missing; computation and state order are unaffected.
    values = np.asarray(list(raw.values()), dtype=np.float64)
    low, high = values.min(axis=0), values.max(axis=0)
    center, scale = (low + high) / 2, float(np.max(high - low))
    positions = {body: np.round((np.asarray(xyz) - center) / scale, 6).tolist() for body, xyz in raw.items()}
    classes = sorted(set(layout["classes"]))
    class_index = {name: i for i, name in enumerate(classes)}
    morphology = morphology_payload(nodes, center, scale)
    brain = np.asarray([positions[body] for body, cls in zip(nodes, layout["classes"], strict=True)
                        if body in positions and (cls.startswith("cb_") or cls.startswith("ol_") or cls.startswith("visual_"))])
    return {"nodes": nodes, "positions": positions, "edges": graph["edges"],
            "states": [], "inputs": [], "frame_map": [], "context_positions": [],
            "hull_polyline": [], "morphology": morphology,
            "flags": {"full_cns": True, "hull_mode": "malecns_anatomy", "graph_mode": "real_connectome",
                      "is_synthetic": False, "anatomy_missing": False, "n_model": len(nodes),
                      "n_reservoir_visible": len(positions), "n_with_soma": provenance["n_soma"],
                      "n_without_coordinates": provenance["n_without_coordinates"],
                      "n_edges": provenance["n_edges"], "n_synapses": provenance["n_synapses"],
                      "graph_hash": provenance["graph_hash"], "coordinate_version": provenance["artifact_hashes"]["layout.json"],
                      "brain_bounds": [brain.min(axis=0).tolist(), brain.max(axis=0).tolist()],
                      "class_names": classes, "class_ids": [class_index[c] for c in layout["classes"]],
                      "show_morphology": True, "show_connections": False, "edge_cap": 12_000}}, None
