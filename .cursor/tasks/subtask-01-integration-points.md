# Subtask 1: Integration points, graph artifact, synthetic fixture, provenance

## Goal

Add the connectome package, a labeled synthetic graph with **≥50 nodes**, provenance/manifest schema, YAML reservoir defaults (no `readout: ridge` in `filters.yaml`), architecture constants, CLI/UI choice strings including `--n-nodes`, run-directory helpers, package-data for the fixture, `cancelled` interrupt status, and a `build_model()` passthrough that leaves GRU/LSTM behavior unchanged. Reservoir training math is **not** implemented here.

## Context

Phase A of the Fly Connectome Reservoir epic. Today `cli.py` `--arch` choices are `gru`/`lstm`, `app.py` Train selectbox is the same pair, `run_training()` always constructs `PDMNet`, and worker `evaluate` writes `completed` without checking `stop.flag`. MaleCNS feather files are **not** assumed present in CI. `src/pdm/models.py` stays a **module** (package conversion is 02a). This phase must leave `pdm train --arch gru` and the existing test suite green.

Builds on current `src/pdm/config.py` `model_defaults()`, `src/pdm/cli.py`, `src/pdm/app.py` Train screen, `src/pdm/paths.py`, `src/pdm/worker.py`, `src/pdm/__init__.py` `STATUSES`, `src/pdm/data/download.py` (pattern for local-path import + provenance).

## Acceptance Criteria

- [ ] Package `src/pdm/connectome/` exists with `sources.py`, `graph.py`, `sampling.py`, `weights.py`, `layout.py`, `provenance.py`, and a shipped synthetic fixture.
- [ ] `pyproject.toml` includes `[tool.setuptools.package-data]` so `pdm.connectome.fixtures` data files install with the package (editable and non-editable). Fixture is loadable via `importlib.resources` (not a repo-relative path that breaks after `pip install`).
- [ ] Shipped synthetic fixture has **at least 50 nodes**. Tests never assume 300 nodes.
- [ ] Synthetic graph always sets `graph_mode="synthetic_fixture"` and `is_synthetic=True`. Every user-visible string for that graph includes exactly: `Synthetic test graph — not a biological connectome`.
- [ ] When `graph_mode=synthetic_fixture`, `n_nodes = min(requested, fixture.N)` and the clamp is **logged**; **never raise**; **never** auto-promote to `real_connectome`. Test: `test_synthetic_n_nodes_clamps_not_raises` (request 1000 against the fixture → gets `fixture.N`; request 8 → 8).
- [ ] Range **500–2000 is for `real_connectome` only**. Out-of-range or `n_nodes >` real graph size fails loudly for `real_connectome`. Synthetic clamp is the opposite policy.
- [ ] MaleCNS loader accepts a **local path** to `connectome-weights-male-cns-v1.0-minconf-0.5.feather` (or equivalent documented filename). If the file is missing, loader does **not** invent rows; it records provenance `source=unavailable` and callers use the synthetic fixture. No silent promotion of synthetic → `real_connectome`.
- [ ] Provenance JSON is written next to every graph artifact (`source`, optional URL, local path, file hash if present, retrieval timestamp, column names actually read, `graph_mode`, `n_nodes`, `n_edges`, `seed`, `node_id` dtype=string, orientation convention `W_res[i,j]=edge j→i`, weight policy `log1p(synapse_count)`, disclaimer).
- [ ] Deterministic connected subgraph sampling: same `(source_graph, n_nodes, seed)` → same `node_id` list (stable **strings**). For synthetic, `n_nodes` is the **clamped** value.
- [ ] Adjacency convention is encoded now (even if spectral scaling lands in 02b): `A[i, j] = log1p(synapse_count of directed edge j→i)`. A one-edge unit test asserts `A[dst, src] != 0` and `A[src, dst] == 0` with **no transpose**.
- [ ] `configs/bearings.yaml` gains a `model.reservoir` block with `n_nodes`, `leak`, `spectral_radius`, `input_scale`, `ridge_alpha`, `seed`, `state_mode`, `graph_mode`. Default **architecture remains `gru`**. Optional bearings `readout: ridge` is allowed **or** omitted so `model_defaults()` fills it.
- [ ] `configs/filters.yaml` gains the same reservoir keys **except it must NOT contain `readout: ridge`** (omit `readout` entirely). `model_defaults()` sets `readout` from `dataset_id` (`bearings` → `ridge`, `filters` → `gradient`). If a filters config explicitly sets `readout: ridge`, **raise** (implementation may live in 01 `model_defaults` or 02c; 01 must not ship the bad YAML).
- [ ] Do **not** document or implement synthetic smoke as `smoke_n_nodes=300`. CLI grows `--n-nodes` (and `--graph-mode` if needed). Tests and later README smoke commands pass **explicit** `--n-nodes 8` (or 8–16).
- [ ] Architecture constants live in one module (e.g. `src/pdm/architectures.py`): `gru`, `lstm`, `fly_connectome_reservoir`, `random_reservoir`. CLI `--arch` and Train selectbox list all four. Default remains `gru`.
- [ ] `build_model(...)` for `gru`/`lstm` returns today’s `PDMNet` with the same constructor arguments. Existing `test_gru_lstm_both_heads_change_weights` still passes without edits to its assertions.
- [ ] For reservoir architecture strings, `build_model` / `run_training` **must not** call `RecurrentEncoder` (that class still only accepts gru/lstm). Until 02c, attempting to train a reservoir raises a dedicated, explicit error (not `architecture must be gru or lstm`).
- [ ] Helpers create `runs/<dataset_id>/<run_id>/connectome/` and document `traces/<unit_id>/` (directories may be empty until 02c/03).
- [ ] `"cancelled"` is added to `pdm.STATUSES`. `"stopped"` remains in the tuple **only** so old `status.json` files still parse. **New** writes that see `stop.flag` / `should_stop()` use **`cancelled`**, never `completed`, never `stopped`.
- [ ] `run_training`: `should_stop()` sets final status `cancelled` (today it is `stopped` at `src/pdm/train.py` ~732/855). Successful GRU completion still `completed`.
- [ ] Worker **evaluate** (and any other `kind`) checks `stop.flag` **before** writing `completed`. If the flag is set, write `cancelled`.
- [ ] Test `test_stop_flag_sets_cancelled_not_completed` in `tests/test_worker_and_app.py`: with `stop.flag` present (or `should_stop` true), worker/train status is `cancelled` and is not `completed` or `stopped`.
- [ ] No MaleCNS bulk file is committed. No FastAPI. No `filters_full_history`. GRU tests green; ruff clean.

## Key Files to Create/Modify

**Create**

- `src/pdm/connectome/__init__.py`
- `src/pdm/connectome/sources.py` — local-path feather/parquet import; GCS/neuPrint URLs as documented constants only; synthetic fallback
- `src/pdm/connectome/graph.py` — edge list → directed graph; `node_id` as `str`
- `src/pdm/connectome/sampling.py` — seeded connected subgraph; **`resolve_n_nodes(graph_mode, requested, available_n)`** clamp vs raise
- `src/pdm/connectome/weights.py` — `log1p` adjacency in **j→i** orientation (spectral radius scaling may be a stub called in 02b, but orientation must be fixed here)
- `src/pdm/connectome/layout.py` — networkx topological 2D/3D layout; anatomical coords only if present in source
- `src/pdm/connectome/fixtures/synthetic_connectome.json` (or `.csv`) — directed graph, **N≥50**, labeled synthetic
- `src/pdm/connectome/provenance.py` — manifest schema + write/read
- `src/pdm/architectures.py` — names, `is_reservoir()`, `is_recurrent_nn()`
- `tests/test_reservoir.py` — start file: synthetic label, clamp-not-raise, sampling determinism, adjacency orientation, missing MaleCNS fallback, YAML defaults parse, filters.yaml has no `readout: ridge`

**Modify**

- `pyproject.toml` — `[tool.setuptools.package-data]` for `pdm.connectome.fixtures` (and `*` json/csv under that package)
- `configs/bearings.yaml` — `model.reservoir` block; do not change split/features/alerts
- `configs/filters.yaml` — `model.reservoir` block **without** `readout: ridge`
- `src/pdm/config.py` — `model_defaults()` copies reservoir keys; sets `readout` from `cfg["dataset_id"]`; does **not** inject reservoir keys into GRU training `compat` (compat wiring is 02c)
- `src/pdm/cli.py` — `--arch` choices include the two new names; `--n-nodes` int optional; optional `--graph-mode`; do **not** add connectome to `download --dataset` choices
- `src/pdm/app.py` — Train architecture selectbox includes the two new names; default still `gru`; do **not** add a fourth screen here
- `src/pdm/models.py` — add `build_model()` that dispatches gru/lstm → `PDMNet`; reservoir → explicit not-yet-trained error. Do **not** convert to a package (02a)
- `src/pdm/train.py` — `run_training` uses `build_model` for GRU/LSTM construction; `should_stop()` → emit **`cancelled`**; pass through `--n-nodes` into `mcfg["reservoir"]` without writing those keys onto GRU `compat`
- `src/pdm/paths.py` — `run_connectome_dir(dataset_id, run_id)`, `run_traces_dir(...)`
- `src/pdm/__init__.py` — add `"cancelled"` to `STATUSES`
- `src/pdm/worker.py` — `evaluate` / `replay_predict` / other kinds: if `stop.flag` exists when the job would complete, write `cancelled` not `completed`
- `tests/test_worker_and_app.py` — `test_stop_flag_sets_cancelled_not_completed`; AppTest still starts; architecture widget may have more options but Smoke/GRU path unchanged

**Do not modify (this subtask)**

- `src/pdm/losses.py`, `src/pdm/splits.py`, `src/pdm/windows.py`, `src/pdm/preprocessing.py` behavior
- Replay/alert math
- `filters.yaml` `mode` / `time_to_seconds`
- Converting `models.py` into a package

## Implementation Notes

### MaleCNS fields

Inspect a real file **only if it exists** under `data/raw/connectome/` or a path argument. Do not invent columns. Implement a mapper for documented FlyEM-style names (`pre`, `post`, synapse count) **and** fail with the actual column list if none match. Tests use the synthetic fixture, not a fake MaleCNS table.

Document the intended remote as comments/constants in `sources.py`:

- `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather`
- HTTPS equivalent `https://storage.googleapis.com/flyem-male-cns/...` (may 404 in CI)
- neuPrint mentioned in provenance docs as an alternate human step, not a test dependency

### Sampling and n_nodes (C1)

```text
if graph_mode == "synthetic_fixture":
    n_nodes = min(requested, fixture.N)   # log if clamped; never raise; never relabel
elif graph_mode == "real_connectome":
    if not (500 <= requested <= 2000): raise
    if requested > source.N: raise
else:
    # random_rewire uses the parent graph’s resolved n_nodes (02b)
```

Tests pass `n_nodes=8` or `16` explicitly. Do not special-case “smoke ⇒ 300”.

### `build_model` / CLI

Keep `RecurrentEncoder` raising `architecture must be gru or lstm`. New names never enter that class. Train default architecture remains `gru` so `test_spec_invariants.py` jobs that pass `architecture="gru"` are untouched. `--n-nodes` on a GRU train is ignored for the network and **must not** appear in GRU `compat`.

### Cancel vs completed (W1)

Today `run_training` sets `last_status = "stopped"` then emits it; worker `evaluate` always writes `completed` after `evaluate_run`. Change both:

- `should_stop()` / `stop.flag` → **`cancelled`**
- Do not write `stopped` for new interrupts
- Do not rewrite an interrupted job as `completed`
- UI chips that list statuses should treat `cancelled` (and legacy `stopped` if read from disk) as interrupt

### Leakage / isolation

This phase does not train reservoirs. Graph sampling must not read bearings/filters tables. Connectome artifacts are not features.

### Package data (W9)

```toml
[tool.setuptools.package-data]
"pdm.connectome.fixtures" = ["*.json", "*.csv"]
```

Keep `[tool.setuptools.packages.find] where = ["src"]`.

## Dependencies

None (first subtask).

## Verification Commands

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python -m pytest tests/test_reservoir.py tests/test_worker_and_app.py -q
.venv/bin/python -m ruff check src tests
.venv/bin/python -m pdm train --help   # --arch lists four names; default gru; --n-nodes present
```

Manual: Train screen still defaults to `gru`; selecting a reservoir architecture does not spawn a silent GRU run. Confirm `configs/filters.yaml` has no `readout: ridge`.

## Notes/Constraints

- NEVER break GRU/LSTM tests or grow GRU `compat` keys.
- NEVER transpose adjacency “to make the plot look right”.
- NEVER treat `synthetic_fixture` as `real_connectome`; NEVER raise when synthetic `n_nodes` exceeds fixture size — clamp and log.
- NEVER put `readout: ridge` in `configs/filters.yaml`.
- NEVER download/commit MaleCNS as a merge requirement.
- NEVER enable `filters_full_history`.
- NEVER map `stop.flag` to `completed` or (for new writes) `stopped`.
- Smoke training is not required for this subtask; any later smoke command must pass `--n-nodes 8`.
