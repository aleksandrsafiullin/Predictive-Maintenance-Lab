# Master Plan: MaleCNS Neural Activity Explorer (anatomy viz)

**Repo:** Predictive Maintenance Lab workspace · package `src/pdm/` · **Stack:** Python ≥3.11 · PyTorch · Streamlit · Plotly · numpy/pandas/scipy · scikit-learn · pyarrow · networkx · pytest · ruff  
**Date:** 2026-09-13  
**Revised:** 2026-09-13 (plan-reviewer REVISE — all Criticals + Warnings + cheap Suggestions)  
**Replaces:** fly-connectome reservoir + explorer epic (subtasks 01–06). That epic is **already implemented**. This file is a **new visualization epic**. **Do not overwrite** `.cursor/tasks/subtask-01-*.md` … `subtask-06-*.md`. New work is numbered **07 onward**.

**Source of truth:** this plan + `Cursor_Predictive_Maintenance_MVP_Spec.md` (leakage / splits / loss) + existing reservoir docs (`docs/fly_connectome.md`, `docs/neural_activity_explorer.md`).

GRU/LSTM remain defaults. Reservoir architectures stay exactly `fly_connectome_reservoir` and `random_reservoir`. **Do not** raise ESN `n_nodes` to ~166k. Split **N_model** (reservoir width from run/provenance, `real_connectome` 500–2000) from **N_viz** (soma context after cap). Do not add Transformers, CNNs, extra RNN cells, FastAPI, React SPAs, Docker, MLflow, Neuroglancer runtime, or a CAVE client as a required dep.

`--smoke` is not a quality claim. Real data only; do not invent MAT/CSV/feather columns. UI English. Bind `127.0.0.1:8501`.

## Overview

Ship the **MaleCNS-style** Neural Activity Explorer we already agreed: local Streamlit WebGL (existing custom component, **no Neuroglancer iframe, no CDN**). Visual contract of the official MaleCNS scene:

- Reference (do not embed): https://neuroglancer-demo.appspot.com/#!gs://flyem-male-cns/v1.0/male-cns-v1.0.json
- Hub: https://male-cns.janelia.org/explore/
- Layers: neuropil/CNS hull + **full soma context cloud** (~1e5 points) + **reservoir subset glow** (existing ESN 500–2000) on **true anatomical xyz**
- Overlay chart already shipped (`build_work_overlay_figure`) stays
- MaleCNS data is **CC-BY**. Vendor or local-download soma xyz; never invent MAT/feather columns

### Current bug

Explorer treats `n_nodes=1000` BFS subgraph as “the brain”: parametric optic-lobe cartoon (`polarToCns` / `buildHullPositions`) + additive bloom. Weights feather has only `body_pre` / `body_post` / synapse counts — **no xyz**. `layout.json` is seeded spring / ring (including possible 3D spring). The 1000 reservoir nodes are not a MaleCNS soma field.

| ID | Phase | Theme | Subtask |
|----|-------|--------|---------|
| G | 07 | Soma xyz artifact: allowlist search, provenance `viz_soma_xyz`, synthetic fixture, loud explicit-path errors | `subtask-07-soma-anatomy.md` |
| H | 08 | Scene builder: raw-layout vs spring, soma wins, `n_model` from provenance, N_viz cap | `subtask-08-scene-builder.md` |
| I | 09 | WebGL: dim context cloud, reservoir glow from `states`; cartoon only if schematic | `subtask-09-webgl-anatomy.md` |
| J | 10 | App + live: UI scene helper wins; catch unknown schema; overlay unchanged | `subtask-10-explorer-wiring.md` |
| K | 11 | Tests + docs (CDN grep, N_model vs N_viz, CC-BY, Neuroglancer is reference) | `subtask-11-anatomy-tests-docs.md` |

### Product constraints (non-negotiable)

1. **Never break GRU/LSTM** or `tests/test_spec_invariants.py`. This epic must not touch splits, windows, losses, or checkpoint `compat` key sets.
2. **Do not rewrite reservoir math.** `W_res` orientation, leaky ESN update, ridge-vs-gradient, `predict` / `predict_with_trace` shared kernel stay as shipped in 01–06.
3. **N_model vs N_viz.** `flags.n_model` is ESN width from the **run / provenance / checkpoint** (`real_connectome` 500–2000). It is **not** `len(nodes)` after the live 512 display cap. `flags.n_viz` / `context_n` = context count **after** the 80k cap. `flags.downsampled` and `n_nodes_display` stay the existing reservoir-display downsample. Never expand `states` to N_viz. Never set ESN width to 166k.
4. **Synthetic banner exact:** `Synthetic test graph — not a biological connectome`. Never auto-promote synthetic → `real_connectome`. Synthetic runs keep **schematic** hull + banner even if a local soma file exists (synthetic node ids are not FlyEM body ids).
5. **Explorer disclaimer exact:** `Computational activity in a connectome-based reservoir. This is not a biophysical simulation or recorded activity of a living fly brain.`
6. **Node flashes from real `states` only.** No `Math.random` in `main.js` (tests grep it). No `sin(time)` fake spikes. Deterministic downsample only: **numeric body-id sort** then even stride / `np.linspace` — not `numpy.random`, not `Math.random`.
7. **No CDN.** Vendored Three.js + Streamlit component bridge. No `https://` script/module `src`. No unpkg / cdnjs / jsdelivr / googleapis.
8. **No Neuroglancer iframe / runtime / CAVE client.** Official scene is a **visual reference**, not a dependency.
9. **CI never requires** the ~1 GB weights feather or a ~166k soma file. Missing anatomy / allowlist miss → **labeled schematic fallback**, never silent `real_connectome`.
10. **Worker owns heavy jobs.** UI does not train. Build trace while busy → error caption `A heavy job is already running. Please wait.`
11. Do **not** change `EXPLORER_MODES` values or required button labels (`Build trace`, `Load demo scenario`).
12. Overlay leakage rules already in place: do not rescore with future rows; `show_gt` does not move the predicted zone. Do not “fix” overlay by changing `predicted_rul_s`.
13. `filters_full_history` stays disabled. Do not mix bearings/filters checkpoints.
14. **Soma search is an ordered filename allowlist** (see Defaults). No `*.feather` glob. Skip `MALEMCNS_FILENAME` and `syn-points-*` **by name before schema**. Allowlist miss → empty + `source=unavailable`, **never raise**. An **explicit path** to the weights feather or a syn-points file still **raises** `ValueError`.
15. **Soma provenance is viz-only.** `graph_mode="viz_soma_xyz"` when a soma table loaded; missing/allowlist-miss `"unavailable"`. Count **`n_points`**, not `n_nodes`. **Never overwrite** `runs/<dataset>/<run_id>/connectome/provenance.json`.
16. **UI must not crash on unknown schema.** Catch soma `ValueError` in Streamlit: show the columns message, fall back to `hull_mode=schematic_cns` + `anatomy_missing=True`. No uncaught Streamlit exception.
17. **Live snapshot must not stamp `hull_mode=schematic_cns`.** That extra must not overwrite anatomy. The **UI scene helper always wins**. No 80k xyz in `live_activity.json` / `.npz`. Cache soma load.
18. **Tests inject a tmp soma dir** (monkeypatch `default_soma_dir` / search path). Never assume workspace `data/raw/connectome/` is empty or unused.
19. **AppTest does not certify look.** Browser on `http://127.0.0.1:8501` when tools exist; pytest greps + payload tests are the merge gate for WebGL appearance.
20. Ruff clean; full pytest green after each subtask.

### Layout (extend, do not replace)

```
src/pdm/connectome/          # existing graph/weights + NEW soma anatomy loader
src/pdm/visualization/       # explorer / overlay / live / component — extend
src/pdm/visualization/component/frontend/  # vendored Three.js; no CDN
```

Reservoir training still writes `runs/<dataset>/<run_id>/connectome/{graph.json,weights.npz,layout.json,provenance.json}` with **N_model** nodes. Soma context is loaded at **viz time** from the allowlist under `data/raw/connectome/` (or a tiny synthetic fixture in tests). Old runs do not need retraining to gain anatomy. Train `provenance.json` stays the ESN graph record.

---

## Phases

### Phase G — Anatomy artifact (soma xyz)

Loader for soma xyz keyed by **string body id** matching `graph.json` node ids. Provenance + file hash + `is_synthetic`. **Default search = ordered allowlist** in `data/raw/connectome/` (same dir as the weights feather). First allowlisted filename that exists is inspected. **No glob.** Skip weights / `syn-points-*` **by name** before opening schema. Allowlist miss → empty table, `source=unavailable`, `graph_mode="unavailable"` — **do not raise**. Optional tiny **labeled synthetic** xyz fixture for tests (explicit `load_synthetic_somas()` only).

Document GCS / neuPrint **human** download. **Do not invent column names:** if an allowlisted or **explicit** file exists, inspect columns and map documented FlyEM-style fields (`bodyId` / `bodyid` / `body_id` + `somaLocation` or `x,y,z`). Unknown schema on an **explicit path** → `ValueError` listing actual columns. Explicit path to `MALEMCNS_FILENAME` or `syn-points-*` (or a table with `x_pre`/`y_pre`/`z_pre`) still raises.

**Deliverable:** `subtask-07-soma-anatomy.md`  
**Risk:** `*.feather` glob picking the 1 GB weights file; using synapse-point xyz as somas; auto-promoting synthetic xyz; downloading in CI; writing soma `n_nodes` over the run provenance; stuffing 166k points into `W_res`.

### Phase H — Scene builder

Python merge (viz only):

1. Reservoir `nodes` / `positions` / `states` from the run. `flags.n_model` from **run/provenance ESN width**, not post-cap `len(nodes)`.
2. Anatomical vs spring decided on **raw `layout.json`** as written. Any coordinate set produced by `layout_positions()` spring (2D **or 3D**) is **non-anatomical**. Real / rewired-real **soma table wins over spring**.
3. `context_positions`: all / downsampled somas (N_viz), **not** the BFS subgraph. Normalize context+reservoir into a shared bbox (center/scale) so JS can skip `polarToCns`.
4. Flags: `hull_mode=malecns_anatomy|schematic_cns`, `context_n`, `context_downsampled`, `anatomy_missing`, `n_model`, `n_viz`. `downsampled` / `n_nodes_display` remain the live reservoir-display cap.
5. Deterministic downsample if N > cap (**80_000**): **numeric body-id sort** then even stride. Caption when downsampled.
6. `hull_polyline` from soma XY convex hull (scipy) so JS does not invent a second cartoon when anatomy exists.

**Deliverable:** `subtask-08-scene-builder.md`  
**Depends on:** G  
**Risk:** treating 3D spring as anatomy (z-span heuristic); setting `n_model = len(nodes)` after 512 cap; remapping synthetic ids onto FlyEM somas; changing train `n_nodes`.

### Phase I — WebGL

`main.js` / `index.html`: dim **context** point cloud (normal blending, not additive bloom); brighter **reservoir** points from `states`; anatomy hull from payload polyline; schematic cartoon **only** when `hull_mode === "schematic_cns"`. `applyArgs` defaults missing `hull_mode` to schematic. Idle almost black; additive **only** for active reservoir nodes. Scale point size with N. Play/pause stays in JS. No `Math.random`. Both renderers honor `hull_mode`.

**Tests:** JS source grep that `fitNodesIntoHull` / `polarToCns` run only under `hull_mode === "schematic_cns"`. Python payload test: `hull_mode=malecns_anatomy` + non-empty `context_positions`. Browser on `:8501` when tools exist; **AppTest does not certify look**.

**Deliverable:** `subtask-09-webgl-anatomy.md`  
**Depends on:** H (payload contract)

### Phase J — App wiring

Explorer + train-live widget consume the new payload. **UI scene helper always wins** over any live extra. **Do not** stamp `hull_mode=schematic_cns` in `write_training_live_snapshot` extras. Cache soma load. Catch unknown-schema `ValueError` → columns caption + schematic + `anatomy_missing`. Synthetic → schematic + exact banner. `real_connectome` without soma → anatomy-missing caption. Overlay unchanged. `EXPLORER_MODES` and button labels unchanged.

**Deliverable:** `subtask-10-explorer-wiring.md`  
**Depends on:** H (and I for visual contract)

### Phase K — Tests / docs

Extend `tests/test_neural_explorer.py`. Include syn-points filename + `x_pre/y_pre/z_pre` cases; tmp-dir soma injection; 3D spring vs soma; junk/weights explicit path in UI. CDN / `Math.random` greps stay green. Docs: `docs/neural_activity_explorer.md` + `docs/malecns_visualization.md` (N_model vs N_viz, soma allowlist, CC-BY, Neuroglancer is reference, **`layout.json` spring is schematic**).

**Deliverable:** `subtask-11-anatomy-tests-docs.md`  
**Depends on:** G–J

---

## Dependencies (DAG)

```text
07  Soma xyz loader + allowlist + synthetic fixture + viz provenance
        │
        ▼
08  Scene builder (raw layout vs spring; n_model from provenance; N_viz cap)
        │
        ├──────────────┐
        ▼              ▼
09  WebGL layers   10  App + live (UI helper wins; catch schema errors)
        │              │
        └──────┬───────┘
               ▼
11  Tests + docs closeout
```

**Do not parallelize 09 and 10 in the same developer pass** unless 08’s payload keys are frozen and tests already cover them. Shared files: `explorer.py` (08, 10), `app.py` (10), `live.py` (10 — remove hull_mode stamp), `main.js` (09), `tests/test_neural_explorer.py` (all).

Do **not** edit `src/pdm/models/`, `losses.py`, `splits.py`, `windows.py`, `connectome/weights.py`, or ESN update code in this epic.

---

## Execution order

| # | Subtask file | Scope | Effort |
|---|--------------|--------|--------|
| 7 | `subtask-07-soma-anatomy.md` | Allowlist loader, viz provenance, tiny synthetic xyz, syn-points tests | 1–1.5 h |
| 8 | `subtask-08-scene-builder.md` | Raw layout vs spring; `n_model` from provenance; N_viz cap | 1–1.5 h |
| 9 | `subtask-09-webgl-anatomy.md` | Context cloud + reservoir glow; schematic-only cartoon grep | 1.5–2 h |
| 10 | `subtask-10-explorer-wiring.md` | UI helper wins; schema catch; live no hull stamp | 1 h |
| 11 | `subtask-11-anatomy-tests-docs.md` | Remaining tests, docs, ruff | 1 h |

---

## Scope / invasiveness

| Subtask | Invasiveness |
|---------|----------------|
| 07 | New `connectome/anatomy.py` + fixture JSON; optional doctor field; **no train loop**; **no run provenance writes** |
| 08 | New scene helper + `build_explorer_payload` keys; layout helper for raw vs spring (not z-only) |
| 09 | `main.js` / `index.html` only (both renderers) |
| 10 | `app.py` + `live.py` (remove `hull_mode` stamp); soma cache; schema catch |
| 11 | Tests + docs |

---

## Verification (every subtask)

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check src tests
```

| Area | Gate |
|------|------|
| Existing leakage / GRU/LSTM / censoring / checkpoints | `tests/test_spec_invariants.py` (must stay green; **do not add anatomy cases here**) |
| Existing UI / worker | `tests/test_worker_and_app.py` |
| Explorer / soma / payload / CDN | `tests/test_neural_explorer.py` |
| Reservoir math | `tests/test_reservoir.py` (must stay green; do not change ESN tests) |
| Env | `.venv/bin/python -m pdm doctor` if CLI/doctor reports soma presence |
| Browser | Exercise explorer on `http://127.0.0.1:8501` in 09/10 **when tools exist**. AppTest is the **behavior** merge gate (captions, buttons, flags). **AppTest does not certify WebGL look.** |
| Smoke / Full train | **Not** a quality gate. Do not retrain to “prove” anatomy. |

After **each** subtask the full existing suite must still pass.

---

## Payload contract (freeze in 08)

`build_explorer_payload` / component args (extend, do not remove existing keys):

| Key | Meaning |
|-----|---------|
| `nodes` | Reservoir node ids (str); may be display-downsampled (live 512) |
| `positions` | `{node_id: [x,y,z]}` for those display nodes |
| `states` | `[frames, n_nodes_display]` real ESN states for **display nodes only** — never padded to N_viz |
| `edges` | Reservoir edges (optional; keep dim if crowded) |
| `context_positions` | `[[x,y,z], …]` soma context after cap, **no 166k ids** |
| `hull_polyline` | `[[x,y,z], …]` silhouette when anatomy exists; empty on schematic |
| `flags.hull_mode` | `malecns_anatomy` \| `schematic_cns` (JS `applyArgs` defaults missing → `schematic_cns`) |
| `flags.n_model` | **ESN width from run/provenance** (500–2000 real). Not `len(nodes)` after live cap |
| `flags.n_nodes_display` | `len(nodes)` / `states.shape[1]` after reservoir display downsample |
| `flags.n_viz` / `flags.context_n` | Context count **after** 80k (or live 20k) cap |
| `flags.context_downsampled` | bool (context cap) |
| `flags.downsampled` | existing **reservoir-activity** downsample (live 512), distinct from context |
| `flags.anatomy_missing` | real / rewired-real without usable soma (file miss, allowlist miss, or caught schema error) |
| `flags.is_synthetic` | existing; drives exact synthetic banner |

Default `hull_mode` remains `schematic_cns` so old callers / GRU-skip paths do not crash.

---

## Leakage, isolation, and math (carry through)

Unchanged from 01–06:

- Train-only scalers; whole-unit splits; predictor never sees future rows or reference RUL
- Overlay: prefix `timestamp_s ≤ Now`; `show_gt` does not move predicted zone
- Traces / live snapshots still use the shared `forward_states` kernel
- Live probe remains **train-split windows only**
- Never load a bearings checkpoint into filters
- `synthetic_fixture` clamps `n_nodes`; never relabel as `real_connectome`

New:

- Soma join is **display**. It must not change `W_res`, run `n_nodes`, or traces.
- Soma provenance lives on the `SomaTable` object only. Never rewrite run `connectome/provenance.json`.
- Context downsample: sort body ids **numerically** when the string is an int (`int(id)`), else lexicographic fallback; then even stride. Not RNG.

---

## Defaults

| Key | Default | Notes |
|-----|---------|--------|
| Reservoir `n_nodes` | 1000 (YAML) | **500–2000 for `real_connectome` only.** Unchanged. This is `flags.n_model`. |
| `CONTEXT_CAP` / N_viz cap | **80_000** | Numeric-id sort + stride; caption when hit |
| Compact live context cap | ≤ 20_000 | Caption if used; states still `LIVE_NODE_CAP=512` |
| Soma search dir | `data/raw/connectome/` | Same directory as weights feather |
| Soma **allowlist** (ordered) | `body-annotations-male-cns-v1.0-minconf-0.5.feather` first; extend only with additional **named** annotation files if documented | **No glob.** Skip `MALEMCNS_FILENAME` and `syn-points-*` by name |
| `hull_mode` without soma | `schematic_cns` | Labeled schematic, never silent FlyEM |
| Soma `graph_mode` | `viz_soma_xyz` if loaded; `unavailable` if miss | Not `real_connectome` |

Weight feather filename (already shipped): `connectome-weights-male-cns-v1.0-minconf-0.5.feather` (`MALEMCNS_FILENAME`) — **not** a soma source.

---

## Artifact layout

```
data/raw/connectome/
  connectome-weights-male-cns-v1.0-minconf-0.5.feather   # existing; SKIP by name
  syn-points-*.feather                                   # if present; SKIP by name
  body-annotations-male-cns-v1.0-minconf-0.5.feather     # allowlist #1; optional

src/pdm/connectome/fixtures/
  synthetic_connectome.json     # existing ≥50-node graph
  synthetic_somas.json          # NEW tiny labeled xyz for tests

runs/<dataset>/<run_id>/connectome/
  graph.json / weights.npz / layout.json / provenance.json
  # provenance.json = ESN graph (01–06). Soma loader MUST NOT overwrite it.
```

Explorer **joins at read time**. Do not rewrite `weights.npz` with 166k rows.

---

## Out of scope

- FastAPI / React SPA / Docker / MLflow / TensorFlow / Transformers / OpenAI SDK
- Neuroglancer iframe, CAVE/neuprint Python client as a required import
- Raising ESN `n_nodes` to full MaleCNS (~166k)
- Extra NN architectures beyond GRU, LSTM, `fly_connectome_reservoir`, `random_reservoir`
- Enabling `filters_full_history` or inventing MATLAB / feather fields
- Bundling the 1 GB feather or full 166k soma table in git / CI
- Treating `--smoke` or pretty WebGL as model quality
- Auto-promoting `synthetic_fixture` to `real_connectome`
- Changing overlay leakage / rescoring alerts
- Rewriting leaky-ESN math, orientation tests, or GRU `compat` blobs
- Using `*.feather` glob as the soma search
- Treating `layout.json` spring (2D or 3D) as FlyEM anatomy

---

## Next step

Re-run **plan-reviewer** on `.cursor/tasks/` (this master plan + `subtask-07`…`subtask-11` only; 01–06 are historical), then `/orchestration` for implement → review per subtask.
