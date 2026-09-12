# Master Plan: Fly Connectome Reservoir + Neural Activity Explorer

**Repo:** `/workspace` · package `src/pdm/` · **Stack:** Python ≥3.11 · PyTorch · Streamlit · Plotly · numpy/pandas/scipy · scikit-learn · pyarrow · networkx · pytest · ruff  
**Branch:** `cursor/fly-connectome-reservoir-a7e4`  
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

| ID | Phase | Theme |
|----|-------|--------|
| A | 01 | Integration points, synthetic graph, provenance, config/CLI hooks |
| B | 02 | Leaky ESN core, fly + random, readout, train/eval for both datasets |
| C | 03 | `predict_with_trace`, contributions, lazy artifacts, parity |
| D | 04 | Neural Activity Explorer (WebGL component + Streamlit page) |
| E | 05 | Comparison table, alerts inspection, demo hooks |
| F | 06 | Full suite, docs, validation notes, ruff |

### Product constraints (non-negotiable)

1. **Never break GRU/LSTM** code paths or `tests/test_spec_invariants.py` / `tests/test_worker_and_app.py`.
2. **Orientation:** `W_res[i,j]` = edge **j→i**. Unit-test it. Never “fix” orientation with a silent transpose.
3. **Filters censoring:** gradient Weibull NLL (or the same censored loss already used). Never ridge/MSE with censored end as `RUL=0`.
4. **`synthetic_fixture`** always displays: `Synthetic test graph — not a biological connectome`. Never use it as the real connectome in fly-vs-random comparisons.
5. **MaleCNS in CI:** if the feather file is not downloadable, ship loader + local-path import + synthetic fallback + provenance. Document real-data steps. Do not fake MaleCNS weights.
6. **`predict` and `predict_with_trace` share one state-update implementation.**
7. **Contribution identity:** intercept + input contributions + neuron contributions = raw prediction (tolerance `1e-5`).
8. **Cancel:** stop.flag → status `cancelled` (existing `stopped` remains an interrupt synonym). Never rewrite an interrupted job as `completed`.
9. Ruff clean; all tests green.
10. **No CDN** in the WebGL component; vendor Three.js locally.

### Layout decision (`models.py` vs `models/`)

`src/pdm/models.py` is a **module** today. The requested files `src/pdm/models/reservoir.py` etc. require a **package**.

**Do this in Phase B, step 0:** move the current module to `src/pdm/models/recurrent.py` and add `src/pdm/models/__init__.py` that re-exports `PDMNet`, `RecurrentEncoder`, `RULHead`, `WeibullHead` unchanged so every existing `from pdm.models import PDMNet` keeps working. Then add reservoir modules beside it.

Do **not** leave both `src/pdm/models.py` and `src/pdm/models/` on disk.

Connectome + visualization packages are new and do not collide:

```
src/pdm/connectome/          # graph IO, sampling, weights, layout
src/pdm/models/              # package: recurrent GRU/LSTM + reservoirs + readout
src/pdm/visualization/       # traces, contributions, export, Streamlit component
```

---

## Phases

### Phase A — Integration points + graph artifact + synthetic fixture + provenance

Create the connectome package, a tiny **labeled** synthetic graph used by tests/UI, provenance/manifest schema, YAML defaults, architecture constants, CLI/UI choice strings, and run-directory layout (`runs/<run_id>/connectome/`). GRU/LSTM training still constructs `PDMNet` via a new `build_model()` that is a pass-through for `gru`/`lstm`.

Reservoir **training math is not in this phase.** `pdm train --arch fly_connectome_reservoir` may refuse with a clear error until Phase B implements the ESN. CLI/UI must still list the new names.

**Deliverable:** `subtask-01-integration-points.md`  
**Risk:** inventing MaleCNS column names; silent synthetic→real relabel; breaking `--arch` default `gru`.

### Phase B — Reservoir core + train/eval

Leaky ESN update, frozen `W_in` / `W_res` / `b_res`, trainable linear readout, `fly_connectome_reservoir` and `random_reservoir` (degree-preserving directed rewiring, same N/E). Wire `run_training`, `load_trained_model`, `evaluate_run`, `Predictor`. Bearings may use ridge **or** gradient linear readout. Filters **must** use gradient censored loss. CPU unit tests for orientation, hand-calculated update, seeds, isolation, censoring.

**Deliverable:** `subtask-02-reservoir-core.md`  
**Depends on:** A (graph + config + `build_model` hook)  
**Risk:** silent transpose; ridge on censored filters; optimizer updating frozen reservoir weights; mixing dataset checkpoints.

### Phase C — predict_with_trace + contributions + lazy artifacts

One state-update kernel used by both `predict` and `predict_with_trace`. Per-frame states, per-frame contributions, window/frame maps for replay. Traces computed only when requested. Save/load under `runs/<run_id>/traces/<unit_id>/`. Parity, window-reset, edge-drive, contribution-sum, postprocess, artifact-reload tests.

**Deliverable:** `subtask-03-predict-trace.md`  
**Depends on:** B  
**Risk:** duplicated update loops; traces that do not match displayed RUL; computing traces on every eval.

### Phase D — Neural Activity Explorer

New Streamlit screen in `app.py`. Local Streamlit custom component under `src/pdm/visualization/component/` with **vendored** Three.js (no CDN). Browser-side animation (not Streamlit rerun-per-frame). States from the same inference as prediction. Required caption + synthetic-graph banner. Modes: Overview, Equipment replay, Inside prediction window, Alert inspection. Network topological layout required; anatomical 3D only if coords exist.

**Deliverable:** `subtask-04-neural-explorer.md`  
**Depends on:** C (trace artifacts + mappings). Heavy trace jobs still go through `pdm.worker`.  
**Risk:** CDN URLs; fake flashes; AppTest regressions on the three existing screens.

### Phase E — Comparison table + alerts + demo hooks

Same-split comparison of GRU/LSTM vs fly vs random (matched N/E) vs baselines. Synthetic fixture **excluded** from biological connectome comparisons. Alert-inspection jumps to the triggering window’s real trace. Demo scenario hooks documented and callable from UI/CLI without pretending MaleCNS was used.

**Deliverable:** `subtask-05-comparison-alerts.md`  
**Depends on:** D for explorer modes; B for metrics; C for traces. Sequential after D because `app.py` is shared.

### Phase F — Suite, docs, validation, ruff

`tests/test_reservoir.py` and `tests/test_neural_explorer.py` complete (acceptance tests 1–14). Docs under `docs/`. README only gains **working** commands. Full `pytest` + `ruff`. No smoke metrics presented as quality.

**Deliverable:** `subtask-06-tests-docs.md`  
**Depends on:** A–E.

---

## Dependencies (DAG)

```text
01 Phase A (graph, provenance, config, CLI/UI choices, build_model passthrough)
        │
        ▼
02 Phase B (ESN + readout + train/eval)
        │
        ▼
03 Phase C (predict_with_trace, contributions, artifacts)
        │
        ▼
04 Phase D (WebGL explorer page)
        │
        ▼
05 Phase E (comparison + alerts + demo)
        │
        ▼
06 Phase F (full suite + docs)
```

Do **not** parallelize. Shared files: `src/pdm/models/` (B+C), `src/pdm/train.py` (A+B), `src/pdm/predict.py` (B+C), `src/pdm/app.py` (A, D, E), `src/pdm/worker.py` (B, C, D), `tests/test_worker_and_app.py` (A, D, E).

---

## Execution order

| # | Subtask file | Scope |
|---|--------------|--------|
| 1 | `subtask-01-integration-points.md` | Connectome IO, synthetic fixture, YAML/CLI/UI strings, run layout, `build_model` passthrough |
| 2 | `subtask-02-reservoir-core.md` | Convert `models.py` → package; ESN; fly/random; train/eval both datasets |
| 3 | `subtask-03-predict-trace.md` | Shared update kernel, traces, contributions, lazy export |
| 4 | `subtask-04-neural-explorer.md` | Custom WebGL component + new Streamlit screen |
| 5 | `subtask-05-comparison-alerts.md` | Comparison table, alert jump, demo hooks |
| 6 | `subtask-06-tests-docs.md` | Close 1–14 tests, four docs, README commands, ruff |

Phases B and D are the invasive subsystems (train factory + ESN math; vendored WebGL). Orchestration still runs them as one subtask each, as named above.

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
| Smoke / Full train | **Not** a quality gate |

After **each** subtask the full existing suite must still pass. Do not merge a phase that reds GRU tests.

---

## Leakage, isolation, and math (carry through all subtasks)

- Fit encoder / imputer / scaler / `time_scale_s` on **train units only**; persist with the run. Reservoir `W_in` is fixed from `seed`, not fit on test units.
- `unit_id`, `event_time_s`, `RUL`, split labels, official filter RUL are never model inputs.
- Predictor receives **raw** rows ≤ t; evaluator joins GT afterward. Traces use that same prefix.
- Never load a bearings checkpoint into filters. `checkpoints_compatible` must include `architecture` **and**, for reservoirs, `n_nodes`, `graph_mode`, `graph_hash`, `state_mode`, `leak`, `spectral_radius`, `input_scale`, `seed`.
- `W_res @ x` uses **row i, column j = edge j→i**. Test with a one-edge graph; do not transpose if a plot looks wrong.
- Default `state_mode=window_reset`: `x=0` at the start of every window. Step t uses `x[t-1]` from **this** window.
- Filters: `event=0` stays censored in `weibull_nll`. Ridge readout is bearings-only (or any fully observed RUL head). No `duration_s=0` hack.
- `synthetic_fixture` sets `graph_mode=synthetic_fixture` and `is_synthetic=true` in every manifest, checkpoint `compat`, and UI string.
- Linear readout raw output is the contribution identity. Display RUL (`softplus` / Weibull median / `time_scale_s`) is postprocessing (test 11).

---

## Defaults (YAML `model.reservoir`)

| Key | Default | Notes |
|-----|---------|--------|
| `n_nodes` | 1000 | Range 500–2000 for real graphs; tests use tiny graphs |
| `smoke_n_nodes` | 300 | Smoke only |
| `leak` / `alpha` | 0.2 | Same symbol; store as `leak` in YAML, `alpha` in code alias |
| `spectral_radius` | 0.9 | Scale `W_res` after `log1p` |
| `input_scale` | 0.1 | `W_in` |
| `ridge_alpha` | 0.001 | Bearings ridge only; ignored for filters |
| `seed` | 42 | Graph sample + `W_in` + rewiring |
| `state_mode` | `window_reset` | |
| `graph_mode` | `synthetic_fixture` until a local MaleCNS file exists; `real_connectome` when provenance says so | Never auto-promote synthetic |
| `readout` | `ridge` bearings / `gradient` filters | Filters cannot select ridge |

Weight policy: `A[i,j] = log1p(synapse_count of edge j→i)`, then scale so the spectral radius of `A` equals `spectral_radius`.

State update (single implementation):

```text
x[t] = (1 - alpha) * x[t-1] + alpha * tanh(W_res @ x[t-1] + W_in @ u[t] + b_res)
```

Readout (raw):

```text
raw = W_x @ x[T] + W_u @ u[T] + b
```

Display: bearings `softplus(raw) * time_scale_s` (or equivalent non-negative map already used); filters `weibull_median_rul` on `softplus` λ, k.

---

## Artifact layout

```
runs/<dataset_id>/<run_id>/
  connectome/
    provenance.json          # source, hashes, graph_mode, disclaimer
    graph.json               # node_id (str), edges, optional xyz
    weights.npz              # W_in, W_res, b_res (frozen)
    layout.json              # 2D/3D coordinates for explorer
  traces/<unit_id>/          # created only when a trace is requested
    meta.json
    states.npz               # [frames, n_nodes]
    contributions.npz
    frame_map.json           # window ↔ replay step ↔ frame
  best.pt / last.pt          # includes readout; reservoir weights frozen
```

---

## Acceptance tests (map)

Implemented in `tests/test_reservoir.py` unless noted. Explorer-facing checks also in `tests/test_neural_explorer.py`.

| # | Test | Primary subtask |
|---|------|-----------------|
| 1 | Graph orientation `W_res[i,j] = j→i` | 02 (adjacency convention starts in 01) |
| 2 | State update vs hand calculation | 02 |
| 3 | Validation errors (wrong `n_nodes`, dataset isolation) | 02 |
| 4 | Seed reproducibility | 02 |
| 5 | Split/preprocess isolation (no leakage) | 02 |
| 6 | Causality (no future frames) | 03 |
| 7 | predict / trace parity | 03 |
| 8 | Window reset at window boundary | 03 |
| 9 | Edge drive uses previous state in the same window | 03 |
| 10 | Contribution sum within `1e-5` | 03 |
| 11 | Postprocessing raw vs display | 03 |
| 12 | Censoring not RUL=0 | 02 |
| 13 | Artifact reload recovers predictions | 03 |
| 14 | Synthetic labeling | 01 (graph) + 04 (UI) + 05 (comparisons) |

---

## Out of scope

- FastAPI / React SPA / Docker / MLflow / TensorFlow / Transformers / OpenAI SDK
- Additional NN architectures beyond GRU, LSTM, `fly_connectome_reservoir`, `random_reservoir`
- Enabling `filters_full_history` or inventing MATLAB / MaleCNS fields
- Changing split protocol or `time_to_seconds`
- Bundling the full MaleCNS feather in git
- Treating `--smoke` metrics or explorer pretty pictures as model quality
- Downloading GCS objects in CI as a merge requirement

---

## Next step

Run **plan-reviewer** on `.cursor/tasks/`, then `/orchestration` (planner → reviewer → per-subtask developer → code-reviewer).
