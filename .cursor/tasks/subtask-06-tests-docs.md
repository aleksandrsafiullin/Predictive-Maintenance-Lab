# Subtask 6: Full test suite, docs, validation notes, ruff

## Goal

Close acceptance tests 1–14 **and** the review extras already named in 01–05, write the four required docs, add only **working** README commands that pass **explicit `--n-nodes 8`** for synthetic_fixture, and leave `pytest tests -q` plus `ruff check src tests` green without treating smoke metrics as quality.

## Context

Phase F. Phases A–E implemented graph, package split, ESN, train wiring, traces, explorer, and comparison. Numbered tests and named extras should already live in their owning subtasks (W7); this subtask fills gaps, writes docs, and is the regression gate. `docs/` does not exist yet. README currently documents GRU/LSTM only.

## Acceptance Criteria

- [ ] `tests/test_reservoir.py` implements tests **1–14** listed in the master plan (14 may be split with explorer tests). Names should be grep-able (`test_graph_orientation`, `test_state_update_hand_calculation`, `test_wrong_n_nodes_raises`, `test_dataset_checkpoint_isolation`, `test_seed_reproducibility`, `test_split_preprocess_isolation`, `test_no_future_frames`, `test_predict_trace_parity`, `test_window_reset`, `test_edge_drive_previous_state`, `test_contribution_sum`, `test_raw_vs_display_postprocess`, `test_filters_censoring_not_rul_zero`, `test_trace_artifact_reload`, `test_synthetic_fixture_label`).

- [ ] Review extras are present (implement here only if a prior subtask skipped them):
  - `test_synthetic_n_nodes_clamps_not_raises` (01)
  - `test_stop_flag_sets_cancelled_not_completed` in `tests/test_worker_and_app.py` (01, W1)
  - `test_gru_checkpoint_compat_ignores_reservoir_yaml_defaults` in `tests/test_spec_invariants.py` (02c, C2)
  - `test_filters_ridge_raises_before_targets` (02c)
  - `test_ridge_uses_forward_states_kernel` / `test_ridge_no_backward` (02c, W2)
  - `test_load_trained_model_missing_weights_npz_raises` (02c, W8)
  - `test_predict_and_trace_share_update_function` (03, W2)
  - `test_random_reservoir_parent_graph_hash` (02b, W8)
  - frontend CDN grep including component bridge (04, W5)
  - AppTest Screen selection **by label** (04, W6)
  - explorer Build trace while worker busy → error caption (04, W8)

- [ ] `tests/test_neural_explorer.py` covers: new page loads via **4-way** branch; required caption; synthetic banner; no `https://` script/module src and no `unpkg`/`cdn.jsdelivr`/`cdnjs`/`googleapis`; GRU run does not show fake biological activity; cancel/stop writes `cancelled` not `completed`/`stopped`.

- [ ] Existing `tests/test_spec_invariants.py` and `tests/test_worker_and_app.py` pass unchanged in intent (GRU/LSTM, leakage, Weibull censoring, AppTest). GRU `compat` fixtures still load (C2).

- [ ] Docs created:
  - `docs/fly_connectome.md` — loader, local path, GCS/neuPrint human steps, orientation, weight policy, synthetic fallback, **clamp vs 500–2000**, provenance fields, `parent_graph_hash`
  - `docs/neural_activity_explorer.md` — modes, captions, layout fallback, how traces are real inference, no CDN, busy-worker caption
  - `docs/fly_connectome_demo.md` — working commands only (venv python), **`--n-nodes 8`**, synthetic demo path, what not to claim
  - `docs/fly_connectome_validation.md` — which tests lock which constraint; MaleCNS not required in CI; contribution identity on **raw**; ridge in normalized RUL; orientation; censoring; GRU compat isolation

- [ ] README: keep current GRU/LSTM instructions; **append** working commands for reservoir smoke **only if they actually run** with the synthetic fixture (no MaleCNS download as a required step). Commands **must** pass `--n-nodes 8` (or similar tiny size). Do **not** assume 300 fixture nodes. Do not paste fake MAE tables.

- [ ] `cancelled` documented as interrupt status; worker never writes `completed` or `stopped` after `stop.flag`.

- [ ] `configs/filters.yaml` still has no `readout: ridge`. `pyproject.toml` package-data includes `pdm.connectome.fixtures` and visualization component assets.

- [ ] Ruff clean on `src` and `tests`.
- [ ] No `filters_full_history`. No CDN. No extra NN families.

## Key Files to Create/Modify

**Create**

- `docs/fly_connectome.md`
- `docs/neural_activity_explorer.md`
- `docs/fly_connectome_demo.md`
- `docs/fly_connectome_validation.md`

**Modify**

- `tests/test_reservoir.py` — fill any gaps vs tests 1–14 and named extras
- `tests/test_neural_explorer.py` — fill explorer gaps
- `tests/test_spec_invariants.py` — only if C2 test is missing
- `tests/test_worker_and_app.py` — only if W1/W6 tests are missing
- `README.md` — working commands with `--n-nodes 8`
- `src/pdm/cli.py` `doctor` — optional: still GRU forward/backward; may mention networkx; do not fail doctor if MaleCNS absent

**Do not**

- Rewrite `reports/implementation_report.md` with new quality numbers
- Commit `data/` or MaleCNS feather
- Change default architecture away from `gru`
- Add `readout: ridge` to `configs/filters.yaml`
- Add reservoir keys to GRU `compat` fixtures

## Implementation Notes

### Docs content (must be true)

`fly_connectome.md`:

- Preferred source filename and gs:// path
- `pdm` local-path import
- CI behavior: synthetic fixture **≥50** nodes; tests use 8–16 via `n_nodes`; **clamp** `min(requested, fixture.N)` for `synthetic_fixture`; **never auto-promote**
- `real_connectome` n_nodes range 500–2000
- `W_res[i,j] = edge j→i`
- `A[i,j] = log1p(synapse_count j→i)` then spectral radius
- Random reservoir is a control (`graph_mode=random_rewire`, `parent_graph_hash`), not biology
- Synthetic sentence verbatim
- `load_trained_model` reads `weights.npz` only (no seed rebuild)

`neural_activity_explorer.md`:

- Caption verbatim
- Four modes
- Animation in browser
- Trace artifact paths
- Topological vs anatomical layout
- No CDN; vendored Three.js + component bridge
- Build trace while worker busy → error caption

`fly_connectome_demo.md` (commands must match CLI; **explicit tiny n_nodes**):

```bash
.venv/bin/python -m pdm doctor
.venv/bin/python -m pdm train --dataset bearings --arch fly_connectome_reservoir --smoke --n-nodes 8
.venv/bin/python -m pdm app
```

Do **not** document `--smoke` without `--n-nodes` for synthetic_fixture. Note smoke is not quality. Tell the operator to expect the synthetic banner unless they imported MaleCNS.

`fly_connectome_validation.md`:

- Table mapping tests 1–14 **and** named extras → file::test name
- Ridge target space = normalized RUL; identity = pre-display `raw`
- GRU `compat` ignores reservoir YAML defaults
- What was **not** validated (full MaleCNS download in CI, Full 30-epoch accuracy)

### README

Do not replace the GRU quickstart. Add a short “Connectome reservoir (optional)” section with synthetic-graph warning and `--n-nodes 8`.

### Test gaps

If a prior subtask skipped a numbered or named test, implement it here with tiny synthetic graphs and in-memory windows (reuse `tiny_bearing_tables` / `tiny_filter_tables` from `tests/conftest.py`). Do not require XJTU-SY or HSE files. Tests pass `n_nodes=8` or `16` explicitly.

Filters test 12: train or call loss with `event=0` vs a counterfactual `event=1` and `duration=0` / `RUL=0`; assert the implementation does not use the counterfactual. Ridge must be unreachable for filters (`test_filters_ridge_raises_before_targets`). No `nan_to_num` to 0.

## Dependencies

Subtasks 01, 02a, 02b, 02c, 03, 04, 05.

## Verification Commands

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check src tests
.venv/bin/python -m pdm doctor
```

Optional help check:

```bash
.venv/bin/python -m pdm train --help
```

`--smoke` runs are optional and not a quality gate. If shown, they **must** include `--n-nodes 8` (or similar) for `synthetic_fixture`.

## Notes/Constraints

- All product constraints in the master plan still apply, including C1–C3 and W1–W9.
- Do not claim MaleCNS was used in CI.
- Do not treat explorer visuals or smoke MAE as model quality.
- GRU/LSTM remain the default path documented first in README.
- Never raise on synthetic `n_nodes`; clamp. Never put reservoir keys on GRU `compat`. Never CDN. Never `else: screen_replay()`. Never `readout: ridge` in `filters.yaml`.
