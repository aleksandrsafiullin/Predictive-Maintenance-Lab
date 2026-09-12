# Master Plan: Fly Connectome Reservoir + Neural Activity Explorer

**Repo:** `/workspace` · package `src/pdm/` · **Stack:** Python ≥3.11 · PyTorch · Streamlit · Plotly · numpy/pandas/scipy · scikit-learn · pyarrow · networkx · pytest · ruff  
**Date:** 2026-09-12  
**Replaces:** previous review-patch plan (sampler / causal gaps / alert scoring). That patch is already on this branch. This file is a **new epic**, not a continuation of those subtask numbers.

**Source of truth for this epic:** this plan + `Cursor_Predictive_Maintenance_MVP_Spec.md` (existing PDM leakage/split/loss rules). Reservoir architectures and the Neural Activity Explorer are an **explicit scoped exception** to the MVP “GRU/LSTM only” line: add **exactly two** new architecture strings and keep GRU/LSTM as defaults. Do not add Transformers, CNNs, extra RNN cells, FastAPI, React SPAs, Docker, MLflow, or live SCADA.

`--smoke` is not a quality claim. Do not run Full 30-epoch training as an acceptance gate. Real data only; do not invent MAT/CSV/connectome fields. UI English.

## Overview

Add a leaky Echo State Network whose recurrent matrix comes from a fly connectome subgraph (or a matched random rewiring), train **only** a linear readout on existing bearings/filters windows, and visualize **real** reservoir states in a local WebGL explorer.

Existing product stays intact:

- `PDMNet` + `RecurrentEncoder` still implement **GRU and LSTM only**
- Bearings head remains `rul` + Smooth L1; filters head remains `weibull` + right-censored NLL
- Splits stay bearings **9/3/3** and filters **40/10/50** (author test 50 held out)
- Worker still owns heavy jobs; Streamlit still binds `127.0.0.1:8501`
- `filters_full_history` stays disabled
- GRU/LSTM checkpoint `compat` blobs stay **bit-compatible** with today’s `tests/test_spec_invariants.py` fixtures (no extra reservoir keys on GRU/LSTM)

| ID | Phase | Theme |
|----|-------|--------|
| A | 01 | Integration points, synthetic graph (≥50 nodes), provenance, config/CLI hooks, cancel status |
| B1 | 02a | `models.py` → package; GRU/LSTM imports and tests stay green |
| B2 | 02b | Leaky ESN + weights + fly/random graphs; `forward()` contracts |
| B3 | 02c | Train/eval wiring (bearings ridge **or** gradient; filters gradient only); reservoir-only compat keys |
| C | 03 | `predict_with_trace`, contributions, lazy artifacts, shared kernel |
| D | 04 | Neural Activity Explorer (vendored WebGL + 4-way `app.py` branch) |
| E | 05 | Comparison table, alerts inspection, demo hooks |
| F | 06 | Full suite, docs, validation notes, ruff |

### Product constraints (non-negotiable)

1. **Never break GRU/LSTM** code paths or `tests/test_spec_invariants.py` / `tests/test_worker_and_app.py`.
2. **Orientation:** `W_res[i,j]` = edge **j→i**. Unit-test it. Never “fix” orientation with a silent transpose.
3. **Filters censoring:** gradient Weibull NLL (or the same censored loss already used). Never ridge/MSE with censored end as `RUL=0`. Never `nan_to_num` censored `target_rul_s` to 0.
4. **`synthetic_fixture`** always displays: `Synthetic test graph — not a biological connectome`. Never use it as the real connectome in fly-vs-random comparisons. **Never auto-promote** synthetic → `real_connectome`.
5. **MaleCNS in CI:** if the feather file is not downloadable, ship loader + local-path import + synthetic fallback + provenance. Document real-data steps. Do not fake MaleCNS weights.
6. **`predict` and `predict_with_trace` share one state-update implementation** (same function object). Ridge feature collection calls that same `forward_states` / `LeakyESN.forward` kernel.
7. **Contribution identity:** intercept + input contributions + neuron contributions = **raw** linear readout **before** Softplus / Weibull median / `time_scale_s` (tolerance `1e-5`).
8. **Cancel:** any job that sees `stop.flag` / `should_stop()` writes status **`cancelled`** (not `completed`, not `stopped`). Worker `evaluate` checks `stop.flag` before writing `completed`. Legacy `stopped` remains in `STATUSES` for old `status.json` files only.
9. Ruff clean; all tests green.
10. **No CDN** anywhere in `frontend/**/*.html` or non-vendor `*.js`. Vendor Three.js **and** the Streamlit component bridge. No `unpkg` / `cdn.jsdelivr` / `cdnjs` / `googleapis` URLs.
11. **`synthetic_fixture` n_nodes clamps**, never raises: `n_nodes = min(requested, fixture.N)` and log it. Range 500–2000 applies to **`real_connectome` only**. Smoke commands pass explicit `--n-nodes 8` (or 8–16). Shipped fixture has **≥50** nodes; tests request 8–16 explicitly.
12. **Reservoir compat keys** (`n_nodes`, `graph_mode`, `graph_hash`, `state_mode`, `leak`, `spectral_radius`, `input_scale`, `seed`) are written and checked **only** when `architecture` is `fly_connectome_reservoir` or `random_reservoir`. `checkpoints_compatible()` compares those keys only if **both** saved and current are reservoir architectures (same optional pattern as `dataset_version`).
13. **`configs/filters.yaml` must not contain `readout: ridge`.** `model_defaults()` sets readout from `dataset_id`; explicit ridge on filters raises **before any target is used**.
14. **`load_trained_model` for a reservoir** reads `connectome/weights.npz` (and checkpoint readout) only. Missing `weights.npz` raises clearly — never rebuild from seed.

### Layout decision (`models.py` vs `models/`)

`src/pdm/models.py` is a **module** today. Reservoir files require a **package**.

**Do this in Phase B1 (subtask 02a), before ESN code:** move the current module to `src/pdm/models/recurrent.py` and add `src/pdm/models/__init__.py` that re-exports `PDMNet`, `RecurrentEncoder`, `RULHead`, `WeibullHead` unchanged so every existing `from pdm.models import PDMNet` keeps working. Prove GRU tests green, **then** add reservoir modules in 02b.

Do **not** leave both `src/pdm/models.py` and `src/pdm/models/` on disk.

Connectome + visualization packages are new and do not collide:

```
src/pdm/connectome/          # graph IO, sampling, weights, layout, fixtures
src/pdm/models/              # package: recurrent GRU/LSTM + reservoirs + readout
src/pdm/visualization/       # traces, contributions, export, Streamlit component
```

---

## Phases

### Phase A — Integration points + graph artifact + synthetic fixture + provenance

Create the connectome package, a **labeled synthetic graph with ≥50 nodes** (tests pass `n_nodes=8–16` and clamp), provenance/manifest schema, YAML defaults, architecture constants, CLI/UI choice strings (`--n-nodes`, `--graph-mode`), run-directory layout (`runs/<run_id>/connectome/`), `[tool.setuptools.package-data]` for `pdm.connectome.fixtures`, `cancelled` status + worker evaluate stop check.

Reservoir **training math is not in this phase.** `pdm train --arch fly_connectome_reservoir` may refuse with a clear error until Phase B3 implements train wiring. CLI/UI must still list the new names.

GRU/LSTM `build_model()` is a pass-through. Do **not** convert `models.py` to a package here (02a).

**Deliverable:** `subtask-01-integration-points.md`  
**Risk:** inventing MaleCNS column names; silent synthetic→real relabel; raising on synthetic `n_nodes`; putting reservoir keys on GRU `compat`; breaking `--arch` default `gru`; `filters.yaml` containing `readout: ridge`.

### Phase B1 — models.py → package (GRU stays green)

Atomic conversion of `src/pdm/models.py` → `src/pdm/models/` with re-exports. No ESN math. Existing GRU/LSTM tests and checkpoint reload stay green.

**Deliverable:** `subtask-02a-models-package.md`  
**Depends on:** A (only for `build_model` hook placement; conversion is valid even if 01’s `build_model` still lives in the moved module)

### Phase B2 — ESN + weights + graph

Leaky ESN update, frozen `W_in` / `W_res` / `b_res`, linear readout module, `FlyConnectomeReservoir` and `RandomReservoir` (degree-preserving directed rewiring; `graph_mode=random_rewire` + `parent_graph_hash`). `forward()` return types match `_run_epoch`: bearings 1D `[B]` non-negative normalized RUL; filters tuple `(lam, k)` after Softplus. Unit tests for orientation, hand-calculated update, clamp vs raise, seeds. **Not** wired through `run_training` yet (still the 01 explicit error until 02c).

**Deliverable:** `subtask-02b-esn-weights.md`  
**Depends on:** A + B1  
**Risk:** silent transpose; a second tanh loop; `forward()` types that `_run_epoch` cannot unpack.

### Phase B3 — Train/eval wiring

Wire `run_training`, `load_trained_model`, `evaluate_run`, `Predictor`. Bearings may use ridge **or** gradient linear readout. Filters **must** use gradient censored loss; `readout=ridge` raises before any target is used. Ridge collects last-step states via the shared `forward_states` kernel on **train windows only**; no `loss.backward()` on the ridge path; still writes `best.pt`/`last.pt`, `status.json`, snapshot, artifacts; `should_stop()` → `cancelled`. Reservoir compat keys only on reservoir architectures. Missing `weights.npz` → raise, never rebuild from seed.

**Deliverable:** `subtask-02c-train-eval-wiring.md`  
**Depends on:** B2  
**Risk:** GRU `compat` growth; ridge on censored filters; `nan_to_num` of `target_rul_s`; optimizer updating frozen weights; mixing dataset checkpoints.

### Phase C — predict_with_trace + contributions + lazy artifacts

One state-update kernel used by both `predict` and `predict_with_trace` (same function object; test `test_predict_and_trace_share_update_function`). Per-frame states, per-frame contributions on **raw** pre-display output, window/frame maps for replay. Traces computed only when requested.

**Deliverable:** `subtask-03-predict-trace.md`  
**Depends on:** B3  
**Risk:** duplicated update loops; identity test on display RUL; computing traces on every eval.

### Phase D — Neural Activity Explorer

New Streamlit screen. **Explicit 4-way branch** in `app.py` (never `else: screen_replay()`). Local custom component with **vendored** Three.js **and** component bridge (no `https://` script/module src). AppTest selects screens **by label**, not radio index. “Build trace” while worker busy → error caption, no inline run.

**Deliverable:** `subtask-04-neural-explorer.md`  
**Depends on:** C  
**Risk:** CDN URLs; `else` replay fallback; fake flashes; AppTest index breakage.

### Phase E — Comparison table + alerts + demo hooks

Same-split comparison of GRU/LSTM vs fly vs random (matched N/E; random rows carry `parent_graph_hash`) vs baselines. Synthetic fixture **excluded** from biological comparisons. Alert-inspection jumps to the triggering window’s real trace.

**Deliverable:** `subtask-05-comparison-alerts.md`  
**Depends on:** D (shared `app.py`)

### Phase F — Suite, docs, validation, ruff

Close tests 1–14 plus review extras. Docs under `docs/`. README reservoir smoke commands pass **explicit** `--n-nodes 8` (synthetic fixture). Full `pytest` + `ruff`.

**Deliverable:** `subtask-06-tests-docs.md`  
**Depends on:** A–E.

---

## Dependencies (DAG)

```text
01 Phase A (graph, provenance, config, CLI/UI, package-data, cancel)
        │
        ▼
02a Phase B1 (models.py → package; GRU green)
        │
        ▼
02b Phase B2 (ESN + weights + fly/random; forward() contracts)
        │
        ▼
02c Phase B3 (train/eval wiring, ridge, reservoir-only compat)
        │
        ▼
03 Phase C (predict_with_trace, contributions, artifacts)
        │
        ▼
04 Phase D (WebGL explorer page, 4-way branch, no CDN)
        │
        ▼
05 Phase E (comparison + alerts + demo)
        │
        ▼
06 Phase F (full suite + docs)
```

Do **not** parallelize. Shared files: `src/pdm/models/` (02a–03), `src/pdm/train.py` (01, 02c), `src/pdm/predict.py` (02c, 03), `src/pdm/app.py` (01, 04, 05), `src/pdm/worker.py` (01, 02c, 03, 04), `tests/test_worker_and_app.py` (01, 04, 05), `tests/test_spec_invariants.py` (02a, 02c).

---

## Execution order

| # | Subtask file | Scope |
|---|--------------|--------|
| 1 | `subtask-01-integration-points.md` | Connectome IO, ≥50-node synthetic fixture, clamp, YAML/CLI/UI, package-data, `cancelled`, `build_model` passthrough |
| 2 | `subtask-02a-models-package.md` | Convert `models.py` → package; GRU/LSTM stay green |
| 3 | `subtask-02b-esn-weights.md` | Leaky ESN, weights, fly/random, `forward()` contracts |
| 4 | `subtask-02c-train-eval-wiring.md` | Train/eval/Predictor; ridge vs gradient; reservoir-only compat |
| 5 | `subtask-03-predict-trace.md` | Shared update kernel, traces, raw contributions, lazy export |
| 6 | `subtask-04-neural-explorer.md` | Vendored WebGL + 4-way Streamlit screen |
| 7 | `subtask-05-comparison-alerts.md` | Comparison table, alert jump, demo hooks |
| 8 | `subtask-06-tests-docs.md` | Close remaining tests, four docs, README `--n-nodes 8`, ruff |

Phase B is split so the package conversion can go red→green before ESN math, and train wiring can go red→green after unit-tested `forward()` contracts.

---

## Scope / invasiveness

| Subtask | Invasiveness |
|---------|----------------|
| 01 | New `connectome/` package + YAML/CLI/UI strings + worker cancel; no train math |
| 02a | Mechanical package split of `models.py`; high import risk, small diff |
| 02b | New ESN math modules; does not yet change `run_training` loop |
| 02c | Invasive `train.py` / `load_trained_model` / ridge branch / compat |
| 03 | `predict.py` + new `visualization/` |
| 04 | `app.py` screen list + vendored frontend |
| 05 | Comparison + explorer alert mode (shared `app.py`) |
| 06 | Tests + docs only |

---

## Verification (every subtask)

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check src tests
```

| Area | Gate |
|------|------|
| Existing leakage / GRU/LSTM / censoring / checkpoints | `tests/test_spec_invariants.py` |
| Existing UI / worker | `tests/test_worker_and_app.py` |
| Reservoir math, orientation, traces, contributions | `tests/test_reservoir.py` (created A–C, completed F) |
| Explorer page, caption, synthetic banner, no CDN | `tests/test_neural_explorer.py` (created D, completed F) |
| Env | `.venv/bin/python -m pdm doctor` if CLI/env touched |
| Browser | Exercise explorer on `http://127.0.0.1:8501` in D/E if tools exist; AppTest is the merge gate |
| Smoke / Full train | **Not** a quality gate. Synthetic smoke **must** pass `--n-nodes 8` (or similar). |

After **each** subtask the full existing suite must still pass. Do not merge a phase that reds GRU tests.

---

## Leakage, isolation, and math (carry through all subtasks)

- Fit encoder / imputer / scaler / `time_scale_s` on **train units only**; persist with the run. Reservoir `W_in` is fixed from `seed`, not fit on test units.
- `unit_id`, `event_time_s`, `RUL`, split labels, official filter RUL are never model inputs.
- Predictor receives **raw** rows ≤ t; evaluator joins GT afterward. Traces use that same prefix.
- Never load a bearings checkpoint into filters.
- `compatibility_dict` / `checkpoints_compatible`: always compare today’s GRU keys. Add and compare `n_nodes`, `graph_mode`, `graph_hash`, `state_mode`, `leak`, `spectral_radius`, `input_scale`, `seed` **only when both sides are reservoir architectures** (mirror optional `dataset_version` / `features_hash`). GRU/LSTM blobs must remain bit-compatible with current invariant fixtures.
- `W_res @ x` uses **row i, column j = edge j→i**. Test with a one-edge graph; do not transpose if a plot looks wrong.
- Default `state_mode=window_reset`: `x=0` at the start of every window. Step t uses `x[t-1]` from **this** window.
- Filters: `event=0` stays censored in `weibull_nll`. Ridge readout is bearings-only. No `duration_s=0` hack. No `nan_to_num(..., nan=0)` on censored `target_rul_s`.
- `synthetic_fixture` sets `graph_mode=synthetic_fixture` and `is_synthetic=true` in every manifest, checkpoint `compat` (reservoir only), and UI string. Clamp `n_nodes`; never relabel as `real_connectome`.
- Ridge (bearings) fits **normalized RUL** (`y = target_rul_s / time_scale_s`), the same space `_run_epoch` uses for Smooth L1. Contribution identity uses the **pre-display linear raw** (`W_x @ x + W_u @ u + b`) before Softplus / Weibull median / `time_scale_s`. Document both spaces; do not mix them in tests.

---

## Defaults (YAML `model.reservoir`)

| Key | Default | Notes |
|-----|---------|--------|
| `n_nodes` | 1000 | **500–2000 validated only for `real_connectome`.** `synthetic_fixture`: `n_nodes = min(requested, fixture.N)`, log, never raise, never auto-promote. Shipped fixture **≥50** nodes. Tests pass `--n-nodes` / call kwargs **8–16**. |
| `smoke_n_nodes` | omit or real-connectome-only | **Do not** assume a 300-node fixture. Synthetic smoke commands **must** pass explicit `--n-nodes 8`. |
| `leak` / `alpha` | 0.2 | Same symbol; store as `leak` in YAML, `alpha` in code alias |
| `spectral_radius` | 0.9 | Scale `W_res` after `log1p` |
| `input_scale` | 0.1 | `W_in` |
| `ridge_alpha` | 0.001 | Bearings ridge only; ignored for filters |
| `seed` | 42 | Graph sample + `W_in` + rewiring |
| `state_mode` | `window_reset` | |
| `graph_mode` | `synthetic_fixture` until a local MaleCNS file exists; `real_connectome` when provenance says so | Never auto-promote synthetic |
| `readout` | **not in `filters.yaml`**. Set in `model_defaults()` from `dataset_id`: bearings `ridge`, filters `gradient` | Explicit `readout: ridge` on filters **raises** |

Weight policy: `A[i,j] = log1p(synapse_count of edge j→i)`, then scale so the spectral radius of `A` equals `spectral_radius`.

State update (single implementation):

```text
x[t] = (1 - alpha) * x[t-1] + alpha * tanh(W_res @ x[t-1] + W_in @ u[t] + b_res)
```

Readout (raw, contribution identity):

```text
raw = W_x @ x[T] + W_u @ u[T] + b
```

`forward()` contract (must match existing `_run_epoch`):

- bearings: 1D `[B]` non-negative **normalized** RUL (same as `RULHead` = Softplus of raw)
- filters: **tuple** `(lam, k)` after Softplus (same as `WeibullHead`)

Display: bearings `forward() * time_scale_s`; filters `weibull_median_rul` on λ, k.

---

## Artifact layout

```
runs/<dataset_id>/<run_id>/
  connectome/
    provenance.json          # source, hashes, graph_mode, disclaimer, parent_graph_hash if rewired
    graph.json               # node_id (str), edges, optional xyz
    weights.npz              # W_in, W_res, b_res (frozen); required at load — never rebuilt from seed
    layout.json              # 2D/3D coordinates for explorer
  traces/<unit_id>/          # created only when a trace is requested
    meta.json
    states.npz               # [frames, n_nodes]
    contributions.npz
    frame_map.json           # window ↔ replay step ↔ frame
  best.pt / last.pt          # includes readout; reservoir weights frozen
```

`random_reservoir` provenance records `parent_graph_hash` and `graph_mode=random_rewire`.

---

## Acceptance tests (map)

Implemented in `tests/test_reservoir.py` unless noted. Extra review tests are listed in the owning subtask AC (not only in 06).

| # | Test | Primary subtask |
|---|------|-----------------|
| 1 | Graph orientation `W_res[i,j] = j→i` | 02b (adjacency convention starts in 01) |
| 2 | State update vs hand calculation | 02b |
| 3 | Wrong `n_nodes` vs **artifact** raises; synthetic clamp does **not** raise; `real_connectome` 500–2000 | 01 (clamp) + 02b/02c (artifact mismatch) |
| 4 | Seed reproducibility | 02b |
| 5 | Split/preprocess isolation (no leakage) | 02c |
| 6 | Causality (no future frames) | 03 |
| 7 | predict / trace parity | 03 |
| 8 | Window reset at window boundary | 03 |
| 9 | Edge drive uses previous state in the same window | 03 |
| 10 | Contribution sum within `1e-5` on **raw** pre-Softplus | 03 |
| 11 | Postprocessing raw vs display | 03 |
| 12 | Censoring not RUL=0; no `nan_to_num` to 0 | 02c |
| 13 | Artifact reload recovers predictions | 03 |
| 14 | Synthetic labeling | 01 (graph) + 04 (UI) + 05 (comparisons) |
| — | `test_synthetic_n_nodes_clamps_not_raises` | 01 |
| — | `test_stop_flag_sets_cancelled_not_completed` | 01 (`tests/test_worker_and_app.py`) |
| — | `test_gru_checkpoint_compat_ignores_reservoir_yaml_defaults` | 02c (`tests/test_spec_invariants.py`) |
| — | `test_filters_ridge_raises_before_targets` | 02c |
| — | `test_ridge_uses_forward_states_kernel` / no `loss.backward` | 02c |
| — | `test_load_trained_model_missing_weights_npz_raises` | 02c |
| — | `test_predict_and_trace_share_update_function` | 03 |
| — | `test_random_reservoir_parent_graph_hash` | 02b |
| — | No CDN in frontend HTML/JS (incl. component bridge) | 04 |
| — | AppTest Screen radio by **label** | 04 |

---

## Out of scope

- FastAPI / React SPA / Docker / MLflow / TensorFlow / Transformers / OpenAI SDK
- Additional NN architectures beyond GRU, LSTM, `fly_connectome_reservoir`, `random_reservoir`
- Enabling `filters_full_history` or inventing MATLAB / MaleCNS fields
- Changing split protocol or `time_to_seconds`
- Bundling the full MaleCNS feather in git
- Treating `--smoke` metrics or explorer pretty pictures as model quality
- Downloading GCS objects in CI as a merge requirement
- Auto-promoting `synthetic_fixture` to `real_connectome`
- Rebuilding reservoir weights from seed when `weights.npz` is missing
- Putting reservoir keys on GRU/LSTM `compat` blobs

---

## Next step

Run **plan-reviewer** on `.cursor/tasks/`, then `/orchestration` (planner → reviewer → per-subtask developer → code-reviewer).
