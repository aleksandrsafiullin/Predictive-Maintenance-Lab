# Fly connectome reservoir

Leaky Echo State Network whose recurrent matrix comes from a fly CNS subgraph (or a matched random rewiring). GRU/LSTM remain the default architectures. This page documents import, graph orientation, CI fallback, and provenance.

## Preferred source (MaleCNS)

Filename: `connectome-weights-male-cns-v1.0-minconf-0.5.feather`

GCS URI (human download; not a CI step):

```text
gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather
```

HTTPS mirror of the same object:

```text
https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather
```

Place the feather file at `data/raw/connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather`, or pass `--source-path`.

**neuPrint** (`https://neuprint.janelia.org/`) is an alternate human import step. It is not a CI dependency and tests never call it.

The loader does not download GCS objects. Missing files fall back to the synthetic fixture with `source=unavailable` and `graph_mode=synthetic_fixture` — never silently labeled `real_connectome`.

Expected FlyEM-style columns: `body_pre`/`pre`, `body_post`/`post`, and `weight` (or `synapse_count`). Unknown columns are not invented.

## Local path import

```bash
.venv/bin/python -m pdm train --dataset bearings --arch fly_connectome_reservoir --graph-mode real_connectome --n-nodes 1000 --source-path /path/to/file.feather
```

`--source-path` may be a `.feather` file or a directory that contains the preferred filename. Default search path when `--source-path` is omitted: `data/raw/connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather`.

`real_connectome` `n_nodes` must be in **500–2000**. Out of range raises. Sampling never materialises the full ~152 M-edge NetworkX graph (see below).

## Memory and subsampling

The MaleCNS feather (~1 GB, ~152 M edges) is too large to load as a NetworkX graph. For `real_connectome`, the loader:

1. Reads the feather into a pandas DataFrame (columnar; does **not** materialise Python dicts per edge).
2. Builds two sorted numpy arrays (by source, by destination) for O(log n) neighbour lookups.
3. Runs a seeded BFS from a deterministically chosen start node, expanding the frontier using searchsorted. Stops at `n_nodes` (500–2000).
4. Filters the DataFrame to the sampled node set and builds a NetworkX DiGraph **only** for the sampled subgraph.

Provenance records `full_graph_materialized: false`, `sampling_method: seeded_bfs`, and the `seed` used.

The file is already pre-filtered at minimum confidence 0.5 by FlyEM (`minconf-0.5` in the filename). An optional `weight_threshold` parameter retains only edges with weight ≥ threshold; do not set this without a documented biological reason.

## CI / synthetic fixture

CI never requires MaleCNS. The shipped fixture lives in `pdm.connectome.fixtures` and has **≥50** nodes. It is labeled:

> Synthetic test graph — not a biological connectome

Behavior:

- Tests request 8–16 nodes via explicit `--n-nodes` / call kwargs.
- `synthetic_fixture` **clamps** `n_nodes = min(requested, fixture.N)` and logs the clamp. It never raises on oversize requests.
- Synthetic graphs are **never auto-promoted** to `real_connectome`.
- Smoke commands must pass `--n-nodes 8` (do not assume a 300-node fixture).

## Graph orientation and weights

Reservoir matrix convention:

```text
W_res[i, j] = edge j → i
```

A single directed edge `src → dst` therefore has `W_res[dst, src] != 0` and `W_res[src, dst] == 0`. Do not transpose to “fix” a plot.

Weight policy:

```text
A[i, j] = log1p(synapse_count of edge j → i)
```

Then scale `A` so its spectral radius equals the YAML default **0.9** (`model.reservoir.spectral_radius`).

State update (one kernel shared by train, predict, and traces):

```text
x[t] = (1 - alpha) * x[t-1] + alpha * tanh(W_res @ x[t-1] + W_in @ u[t] + b_res)
```

Default `state_mode=window_reset`: `x = 0` at the start of every window. Step `t` uses `x[t-1]` from the **same** window.

## Random reservoir (control, not biology)

`graph_mode=random_rewire` (`--arch random_reservoir`) is a degree-preserving directed rewiring of the parent graph. Provenance records `parent_graph_hash`. It is a matched control, not a biological connectome, and is excluded from fly-vs-biology averages when the parent is synthetic.

## Loading trained weights

`load_trained_model` reads `runs/<dataset>/<run_id>/connectome/weights.npz` (`W_in`, `W_res`, `b_res`) plus the checkpoint readout. Missing `weights.npz` raises (`will not rebuild from seed`). An `n_nodes` mismatch between the checkpoint / `graph.json` and `weights.npz` raises. Synthetic oversize requests never take this path — they clamp.

## Provenance fields

Written to `runs/<dataset>/<run_id>/connectome/provenance.json`:

| Field | Notes |
|-------|--------|
| `source` | `synthetic_fixture`, `local`, or `unavailable` |
| `url` | Documented HTTPS URI when applicable |
| `local_path` | Path or package resource |
| `file_hash` | SHA-256 of a local feather when present |
| `retrieved_at` | UTC timestamp |
| `column_names_read` | Columns actually read |
| `graph_mode` | `synthetic_fixture` \| `real_connectome` \| `random_rewire` |
| `n_nodes` / `n_edges` | After sampling |
| `seed` | Graph sample + `W_in` + rewiring |
| `node_id_dtype` | `string` |
| `orientation` | `W_res[i,j]=edge j→i` |
| `weight_policy` | `log1p(synapse_count)` |
| `disclaimer` | Synthetic sentence when not MaleCNS |
| `parent_graph_hash` | Present for `random_rewire` |
| `graph_hash` | Canonical SHA-256 of nodes/edges |
| `is_synthetic` | `true` for the fixture; never auto-promote |

Also stored: `connectome/graph.json`, `connectome/layout.json` (anatomical xyz when present, otherwise topological spring layout).

## Cancel status

Worker jobs that see `stop.flag` / `should_stop()` write status **`cancelled`**. They never write `completed` or `stopped` after a stop request. Legacy `stopped` remains readable for old `status.json` files only.

## Related

- Demo commands: [fly_connectome_demo.md](fly_connectome_demo.md)
- Explorer: [neural_activity_explorer.md](neural_activity_explorer.md)
- Test map: [fly_connectome_validation.md](fly_connectome_validation.md)
