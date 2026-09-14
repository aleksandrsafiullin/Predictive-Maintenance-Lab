"""Complete classified MaleCNS graph. No node sampling or dense N×N matrices.

The source includes unclassified segments, not just neurons. Our population is
every annotation with a non-null superclass; every connection between these
bodies is retained, including self connections, with its original synapse count.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
from scipy import sparse

from pdm.connectome.anatomy import SOMA_ALLOWLIST
from pdm.connectome.sources import MALEMCNS_HTTPS_URL, default_malemcns_path
from pdm.io_util import atomic_write_json, read_json, sha256_file
from pdm.paths import project_root

POPULATION = "all_classified_neurons"


def classified_annotations(path):
    frame = pd.read_feather(path)
    if not {"bodyId", "superclass", "somaLocation"}.issubset(frame.columns):
        raise ValueError("MaleCNS annotations require bodyId, superclass and somaLocation")
    frame = frame.loc[frame.superclass.notna()].sort_values("bodyId").reset_index(drop=True)
    if frame.empty or frame.bodyId.duplicated().any():
        raise ValueError("Classified body IDs must be unique and nonempty")
    return frame


def induced_connectome(path, body_ids, progress=print):
    """Stream source record batches; retain all directed edges in the population."""
    ids = pd.Index(np.asarray(body_ids, dtype=np.int64))
    if not ids.is_unique or len(ids) == 0:
        raise ValueError("Need unique nonempty body IDs")
    sources, targets, counts = [], [], []
    source_rows, source_synapses = 0, 0
    with pa.memory_map(str(path), "r") as f:
        reader = ipc.open_file(f)
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            src = ids.get_indexer(batch.column("body_pre").to_numpy())
            dst = ids.get_indexer(batch.column("body_post").to_numpy())
            weight = batch.column("weight").to_numpy()
            if np.any(weight <= 0):
                raise ValueError("Source synapse counts must be positive")
            source_rows += len(weight)
            source_synapses += int(weight.sum())
            keep = (src >= 0) & (dst >= 0)
            sources.append(src[keep].astype(np.int32))
            targets.append(dst[keep].astype(np.int32))
            counts.append(weight[keep].astype(np.int32))
            if i % 400 == 0:
                progress(f"Connectome batch {i + 1}/{reader.num_record_batches}")
    matrix = sparse.coo_matrix(
        (np.concatenate(counts), (np.concatenate(targets), np.concatenate(sources))),
        shape=(len(ids), len(ids)), dtype=np.int32,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix, {"source_segment_pairs": source_rows, "source_synapses": source_synapses}


def graph_digest(ids, counts):
    digest = hashlib.sha256()
    for values in (np.asarray(ids, dtype="<i8"), counts.indptr.astype("<i4"),
                   counts.indices.astype("<i4"), counts.data.astype("<i4")):
        digest.update(values.tobytes())
    return digest.hexdigest()


def prepare_full_connectome(source_path=None, cache_dir=None, progress=print):
    source = Path(source_path) if source_path else default_malemcns_path()
    if source.is_dir():
        source = source / default_malemcns_path().name
    annotations = source.with_name(SOMA_ALLOWLIST[0])
    # Full-brain mode has no synthetic fallback.
    if not source.is_file() or not annotations.is_file():
        raise FileNotFoundError("Full MaleCNS requires the weights and body-annotations feather files")
    frame = classified_annotations(annotations)
    ids = frame.bodyId.to_numpy(np.int64)
    hashes = {"file_hash": sha256_file(source), "annotations_sha256": sha256_file(annotations)}
    directory = Path(cache_dir) if cache_dir else project_root() / "data/cache/connectome/full-malecns-v1"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "provenance.json"
    counts_path = directory / "synapse_counts.npz"
    if manifest.is_file() and counts_path.is_file():
        provenance = read_json(manifest)
        if all(provenance.get(k) == v for k, v in hashes.items()):
            counts = sparse.load_npz(counts_path)
            if graph_digest(ids, counts) != provenance.get("graph_hash"):
                raise ValueError("Cached full connectome graph hash mismatch")
            return frame, counts, provenance
    counts, source_stats = induced_connectome(source, ids, progress)
    provenance = {
        "source": "local", "graph_mode": "real_connectome", "is_synthetic": False,
        "dataset": "MaleCNS v1.0", "graph_scope": "whole_classified_cns",
        "sampling_method": POPULATION, "population_rule": "superclass.notna()",
        "node_sampling": False, "edge_threshold": None, "minimum_confidence": 0.5,
        "n_nodes": len(ids), "n_edges": int(counts.nnz), "n_synapses": int(counts.sum()),
        "orientation": "W_res[i,j]=edge j→i", "node_id_dtype": "string",
        "graph_storage": "csr", "full_graph_materialized": False,
        "graph_hash": graph_digest(ids, counts), "url": MALEMCNS_HTTPS_URL,
        "local_path": str(source.resolve()), "license": "CC-BY-4.0",
        "attribution": "FlyEM / HHMI Janelia, Cambridge / MRC LMB, Google Research",
        **hashes, **source_stats,
    }
    sparse.save_npz(counts_path, counts)
    atomic_write_json(manifest, provenance)
    return frame, counts, provenance


def anatomical_locations(frame):
    """Curated soma first, then curated to-soma point; never invent a location."""
    positions, kinds = {}, {}
    for row in frame.itertuples():
        for key in ("somaLocation", "tosomaLocation"):
            value = getattr(row, key, None)
            if isinstance(value, (list, tuple, np.ndarray)) and len(value) == 3:
                if np.isfinite(value).all():
                    positions[str(row.bodyId)] = list(map(float, value))
                    kinds[str(row.bodyId)] = key
                    break
    return positions, kinds


def preview_edges(counts, node_order, cap=12_000):
    """Deterministic display-only subset. The recurrent operator uses every edge."""
    offsets = np.linspace(0, counts.nnz - 1, min(cap, counts.nnz), dtype=np.int64)
    targets = np.searchsorted(counts.indptr, offsets, side="right") - 1
    return [{"src": node_order[int(counts.indices[k])], "dst": node_order[int(i)],
             "weight": int(counts.data[k])} for k, i in zip(offsets, targets, strict=True)]
