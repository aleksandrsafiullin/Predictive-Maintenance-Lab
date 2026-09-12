# Subtask 1: Integration points, graph artifact, synthetic fixture, provenance

## Goal

Add the connectome package, a labeled synthetic test graph, provenance/manifest schema, YAML reservoir defaults, architecture constants, CLI/UI choice strings, run-directory helpers, and a `build_model()` passthrough that leaves GRU/LSTM behavior unchanged. Reservoir training math is **not** implemented here.

## Context

Phase A of the Fly Connectome Reservoir epic. Today `cli.py` `--arch` choices are `gru`/`lstm`, `app.py` Train selectbox is the same pair, and `run_training()` always constructs `PDMNet`. MaleCNS feather files are **not** assumed present in CI. This phase must still leave `pdm train --arch gru` and the existing test suite green.

Builds on current `src/pdm/config.py` `model_defaults()`, `src/pdm/cli.py`, `src/pdm/app.py` Train screen, `src/pdm/paths.py`, `src/pdm/data/download.py` (pattern for local-path import + provenance).

## Acceptance Criteria

- [ ] Package `src/pdm/connectome/` exists with `sources.py`, `graph.py`, `sampling.py`, `weights.py`, `layout.py`, and a shipped synthetic fixture.
- [ ] Synthetic graph always sets `graph_mode="synthetic_fixture"` and `is_synthetic=True`. Every user-visible string for that graph includes exactly: `Synthetic test graph — not a biological connectome`.
- [ ] MaleCNS loader accepts a **local path** to `connectome-weights-male-cns-v1.0-minconf-0.5.feather` (or equivalent documented filename). If the file is missing, loader does **not** invent rows; it records provenance `source=unavailable` and callers use the synthetic fixture. No silent promotion of synthetic → `real_connectome`.
- [ ] Provenance JSON is written next to every graph artifact (`source`, optional URL, local path, file hash if present, retrieval timestamp, column names actually read, `graph_mode`, `n_nodes`, `n_edges`, `seed`, `node_id` dtype=string, orientation convention `W_res[i,j]=edge j→i`, weight policy `log1p(synapse_count)`, disclaimer).
- [ ] Deterministic connected subgraph sampling: same `(source_graph, n_nodes, seed)` → same `node_id` list (stable **strings**). Sampling fails loudly if `n_nodes` exceeds the source graph.
- [ ] Adjacency convention is encoded now (even if spectral scaling lands in subtask 02): `A[i, j] = log1p(synapse_count of directed edge j→i)`. A one-edge unit test asserts `A[dst, src] != 0` and `A[src, dst] == 0` with **no transpose**.
- [ ] `configs/bearings.yaml` and `configs/filters.yaml` gain a `model.reservoir` block with the defaults in the master plan (`n_nodes`, `smoke_n_nodes`, `leak`, `spectral_radius`, `input_scale`, `ridge_alpha`, `seed`, `state_mode`, `graph_mode`). Default **architecture remains `gru`**.
- [ ] Architecture constants live in one module (e.g. `src/pdm/architectures.py`): `gru`, `lstm`, `fly_connectome_reservoir`, `random_reservoir`. CLI `--arch` and Train selectbox list all four. Default remains `gru`.
- [ ] `build_model(...)` for `gru`/`lstm` returns today’s `PDMNet` with the same constructor arguments. Existing `test_gru_lstm_both_heads_change_weights` still passes without edits to its assertions.
- [ ] For reservoir architecture strings, `build_model` / `run_training` **must not** call `RecurrentEncoder` (that class still only accepts gru/lstm). Until subtask 02, attempting to train a reservoir raises a dedicated, explicit error (not `architecture must be gru or lstm`).
- [ ] Helpers create `runs/<dataset_id>/<run_id>/connectome/` and document `traces/<unit_id>/` (directories may be empty until 02/03).
- [ ] `cancelled` is added to `pdm.STATUSES` as an interrupt status. `stopped` remains valid. No code path maps stop.flag to `completed`.
- [ ] No MaleCNS bulk file is committed. No FastAPI. No `filters_full_history`. GRU tests green; ruff clean.

## Key Files to Create/Modify

**Create**

- `src/pdm/connectome/__init__.py`
- `src/pdm/connectome/sources.py` — local-path feather/parquet import; GCS/neuPrint URLs as documented constants only; synthetic fallback
- `src/pdm/connectome/graph.py` — edge list → directed graph; `node_id` as `str`
- `src/pdm/connectome/sampling.py` — seeded connected subgraph
- `src/pdm/connectome/weights.py` — `log1p` adjacency in **j→i** orientation (spectral radius scaling may be a stub called in 02, but orientation must be fixed here)
- `src/pdm/connectome/layout.py` — networkx topological 2D/3D layout; anatomical coords only if present in source
- `src/pdm/connectome/fixtures/synthetic_connectome.json` (or `.csv`) — tiny directed graph, labeled synthetic
- `src/pdm/connectome/provenance.py` — manifest schema + write/read
- `src/pdm/architectures.py` — names, `is_reservoir()`, `is_recurrent_nn()`
- `tests/test_reservoir.py` — start file: synthetic label, sampling determinism, adjacency orientation, missing MaleCNS fallback, YAML defaults parse

**Modify**

- `configs/bearings.yaml`, `configs/filters.yaml` — `model.reservoir` block; do not change split/features/alerts
- `src/pdm/config.py` — `model_defaults()` copies reservoir keys with the master-plan defaults
- `src/pdm/cli.py` — `--arch` choices; do **not** add connectome to `download --dataset` choices unless a dedicated optional flag is used (do not break `bearings|filters|all`)
- `src/pdm/app.py` — Train architecture selectbox includes the two new names; default index still `gru`
- `src/pdm/models.py` — add `build_model()` that dispatches gru/lstm → `PDMNet`; reservoir → explicit not-yet-trained error **or** a thin import of a 02 module if already present. Do **not** convert to a package in this subtask (that is 02 step 0) unless it is required to add `build_model` without breaking imports
- `src/pdm/train.py` — `run_training` uses `build_model` for construction; gru/lstm path identical (same optimizer, datasets, losses)
- `src/pdm/paths.py` — `run_connectome_dir(dataset_id, run_id)`, `run_traces_dir(...)`
- `src/pdm/__init__.py` — add `"cancelled"` to `STATUSES`
- `src/pdm/worker.py` — if `stop.flag` is set when a job would otherwise complete, write `cancelled` (or keep `stopped` for current train emit); **never** `completed`
- `tests/test_worker_and_app.py` — AppTest still starts; architecture widget may have more options but Smoke/GRU path unchanged

**Do not modify (this subtask)**

- `src/pdm/losses.py`, `src/pdm/splits.py`, `src/pdm/windows.py`, `src/pdm/preprocessing.py` behavior
- Replay/alert math
- `filters.yaml` `mode` / `time_to_seconds`

## Implementation Notes

### MaleCNS fields

Inspect a real file **only if it exists** under `data/raw/connectome/` or a path argument. Do not invent columns. Implement a mapper for documented FlyEM-style names (`pre`, `post`, synapse count) **and** fail with the actual column list if none match. Tests use the synthetic fixture, not a fake MaleCNS table.

Document the intended remote as comments/constants in `sources.py`:

- `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather`
- HTTPS equivalent `https://storage.googleapis.com/flyem-male-cns/...` (may 404 in CI)
- neuPrint mentioned in provenance docs as an alternate human step, not a test dependency

### Sampling

Seeded RNG. Prefer BFS/DFS from a deterministic start node (e.g. min `node_id` after shuffle keyed by seed) until `n_nodes` is reached; require the result to be weakly connected. Store `node_id` as strings even if the source used ints.

### `build_model` / CLI

Keep `RecurrentEncoder` raising `architecture must be gru or lstm`. New names never enter that class. Train default architecture remains `gru` so `test_spec_invariants.py` jobs that pass `architecture="gru"` are untouched.

### Cancel vs completed

`run_training` already emits `stopped` on `should_stop`. Do not change successful GRU completion. Add `cancelled` to the public status set. If you change the stop emit string from `stopped` to `cancelled`, update any UI chip that lists statuses. Interrupted evaluate/trace jobs in later subtasks must use the same rule.

### Leakage / isolation

This phase does not train. Still: graph sampling must not read bearings/filters tables. Connectome artifacts are not features.

## Dependencies

None (first subtask).

## Verification

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python -m pytest tests/test_reservoir.py -q
.venv/bin/python -m ruff check src tests
.venv/bin/python -m pdm train --help   # --arch should list four names; default gru
```

Manual: Train screen still defaults to `gru`; selecting a reservoir architecture does not spawn a silent GRU run.

## Notes on constraints

- NEVER break GRU/LSTM tests.
- NEVER transpose adjacency “to make the plot look right”.
- NEVER treat `synthetic_fixture` as `real_connectome`.
- NEVER download/commit MaleCNS as a merge requirement.
- NEVER enable `filters_full_history`.
- Smoke training is not required for this subtask.
