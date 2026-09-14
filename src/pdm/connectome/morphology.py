"""Official MaleCNS SWC arbors for a declared display subset, in source coordinates."""
from __future__ import annotations

import base64
import io
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen

import numpy as np

from pdm.connectome.full import classified_annotations
from pdm.io_util import atomic_write_bytes, atomic_write_json, read_json, sha256_file
from pdm.paths import data_raw, project_root

SWC_BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-swc/"


def morphology_directory():
    return project_root() / "data/cache/connectome/morphology-v1"


def simplify_swc(text, spacing=300.0):
    """Retain branch points and endpoints; simplify only unbranched SWC chains.

    Vertices are exact SWC locations in 8 nm voxels. Simplified chords describe
    one neuron's morphology, never an invented inter-neuron connection.
    """
    rows = np.loadtxt(io.StringIO(text), comments="#", ndmin=2)
    if rows.shape[1] != 7:
        raise ValueError("SWC must have seven columns")
    index = {int(row[0]): i for i, row in enumerate(rows)}
    parent = np.array([index.get(int(row[6]), -1) for row in rows])
    child_count = np.bincount(parent[parent >= 0], minlength=len(rows))
    keep = (child_count != 1) | (parent < 0)
    # Walk back from branch/endpoints, retaining actual vertices at path intervals.
    segments = []
    for start in np.flatnonzero(keep):
        child = int(start)
        cur = int(parent[child])
        steps = 0
        while cur >= 0:
            steps += 1
            if steps > len(rows):
                raise ValueError("Cycle in SWC morphology")
            if keep[cur] or np.linalg.norm(rows[child, 2:5] - rows[cur, 2:5]) >= spacing:
                segments.append(np.r_[rows[child, 2:5], rows[cur, 2:5]])
                child = cur
                if keep[cur]:
                    break
            cur = int(parent[cur])
    return np.asarray(segments, dtype=np.float32).reshape(-1, 6)


def prepare_morphology(limit=160):
    frame = classified_annotations(data_raw() / "connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather")
    directory = morphology_directory()
    directory.mkdir(parents=True, exist_ok=True)
    candidates = []
    # Balanced coverage across neuronal classes, deterministic and independent of activity.
    grouped = list(frame.groupby("superclass", sort=True))
    for _, group in grouped:
        group = group.loc[group.status.eq("Traced") & group.type.notna()]
        group = group.drop_duplicates("type")
        indices = np.linspace(0, max(0, len(group) - 1), min(7, len(group)), dtype=int)
        candidates.extend(group.iloc[indices].bodyId.astype(str).tolist())
    candidates = candidates[:limit]

    def fetch(body):
        path = directory / f"{body}.swc"
        try:
            if not path.is_file():
                with urlopen(SWC_BASE + body + ".swc", timeout=45) as response:
                    atomic_write_bytes(path, response.read())
            lines = simplify_swc(path.read_text())
            return body, lines, {"body_id": body, "url": SWC_BASE + body + ".swc",
                                 "sha256": sha256_file(path), "segments": len(lines)}
        except Exception as exc:
            return body, None, {"body_id": body, "error": str(exc)}

    arrays, body_ids, ranges, sources, failures = [], [], [], [], []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for body, lines, source in pool.map(fetch, candidates):
            if lines is None:
                failures.append(source)
                continue
            arrays.append(lines)
            ranges.append(len(lines))
            body_ids.append(body)
            sources.append(source)
    if not arrays:
        raise RuntimeError("No official MaleCNS skeletons could be loaded")
    np.savez_compressed(directory / "arbors.npz", segments=np.concatenate(arrays),
                        body_ids=np.asarray(body_ids), lengths=np.asarray(ranges, np.int32))
    manifest = {"dataset": "MaleCNS v1.0", "source": SWC_BASE, "coordinate_unit": "8 nm voxels",
                "role": "morphology_display_subset", "n_neurons": len(body_ids),
                "selection": "up to 7 traced types per superclass, deterministic evenly spaced type representatives",
                "simplification": "300 voxel spacing; preserve branch/endpoints and source vertices",
                "license": "CC-BY-4.0", "attribution": "FlyEM / Janelia, Cambridge / MRC LMB, Google Research",
                "sources": sources, "failures": failures,
                "artifact_sha256": sha256_file(directory / "arbors.npz")}
    atomic_write_json(directory / "manifest.json", manifest)
    return manifest


def morphology_payload(node_order, center, scale):
    directory = morphology_directory()
    if not (directory / "manifest.json").is_file():
        return {}
    manifest = read_json(directory / "manifest.json")
    if sha256_file(directory / "arbors.npz") != manifest["artifact_sha256"]:
        raise ValueError("Morphology artifact hash mismatch")
    index = {body: i for i, body in enumerate(node_order)}
    with np.load(directory / "arbors.npz", allow_pickle=False) as data:
        xyz = data["segments"].reshape(-1, 3)
        xyz = ((xyz - center) / scale).astype("<f4")
        # One owner index per segment, used to color real arbors with that cell's state.
        owners = np.repeat([index[str(body)] for body in data["body_ids"]], data["lengths"]).astype("<i4")
    return {"positions_b64": base64.b64encode(xyz.tobytes()).decode(),
            "owners_b64": base64.b64encode(owners.tobytes()).decode(),
            "n_neurons": manifest["n_neurons"], "n_segments": len(owners),
            "source": SWC_BASE, "version": manifest["artifact_sha256"]}
