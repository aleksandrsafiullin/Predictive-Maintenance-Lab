# Subtask 6: Full test suite, docs, validation notes, ruff

## Goal

Close acceptance tests 1–14, write the four required docs, add only **working** README commands, and leave `pytest tests -q` plus `ruff check src tests` green without treating smoke metrics as quality.

## Context

Phase F. Phases A–E implemented graph, ESN, traces, explorer, and comparison. `docs/` does not exist yet. README currently documents GRU/LSTM only. This subtask is documentation + test completion + regression, not new architectures.

## Acceptance Criteria

- [ ] `tests/test_reservoir.py` implements tests **1–14** listed in the master plan (14 may be split with explorer tests). Names should be grep-able (`test_graph_orientation`, `test_state_update_hand_calculation`, `test_wrong_n_nodes_raises`, `test_dataset_checkpoint_isolation`, `test_seed_reproducibility`, `test_split_preprocess_isolation`, `test_no_future_frames`, `test_predict_trace_parity`, `test_window_reset`, `test_edge_drive_previous_state`, `test_contribution_sum`, `test_raw_vs_display_postprocess`, `test_filters_censoring_not_rul_zero`, `test_trace_artifact_reload`, `test_synthetic_fixture_label`).
- [ ] `tests/test_neural_explorer.py` covers: new page loads; required caption; synthetic banner; no CDN in component sources; GRU run does not show fake biological activity; cancel/stop does not mark `completed`.
- [ ] Existing `tests/test_spec_invariants.py` and `tests/test_worker_and_app.py` pass unchanged in intent (GRU/LSTM, leakage, Weibull censoring, AppTest).
- [ ] Docs created:
  - `docs/fly_connectome.md` — loader, local path, GCS/neuPrint human steps, orientation, weight policy, synthetic fallback, provenance fields
  - `docs/neural_activity_explorer.md` — modes, captions, layout fallback, how traces are real inference
  - `docs/fly_connectome_demo.md` — working commands only (venv python), synthetic demo path, what not to claim
  - `docs/fly_connectome_validation.md` — which tests lock which constraint; MaleCNS not required in CI; contribution tolerance; orientation; censoring
- [ ] README: keep current GRU/LSTM instructions; **append** working commands for reservoir smoke **only if they actually run** with the synthetic fixture (no MaleCNS download as a required step). Do not paste fake MAE tables.
- [ ] `cancelled` documented as interrupt status; worker never writes `completed` after stop.flag.
- [ ] Ruff clean on `src` and `tests`.
- [ ] No `filters_full_history`. No CDN. No extra NN families.

## Key Files to Create/Modify

**Create**

- `docs/fly_connectome.md`
- `docs/neural_activity_explorer.md`
- `docs/fly_connectome_demo.md`
- `docs/fly_connectome_validation.md`

**Modify**

- `tests/test_reservoir.py` — fill any gaps vs tests 1–14
- `tests/test_neural_explorer.py` — fill explorer gaps
- `README.md` — working commands only
- `src/pdm/cli.py` `doctor` — optional: still GRU forward/backward; may mention networkx; do not fail doctor if MaleCNS absent

**Do not**

- Rewrite `reports/implementation_report.md` with new quality numbers
- Commit `data/` or MaleCNS feather
- Change default architecture away from `gru`

## Implementation Notes

### Docs content (must be true)

`fly_connectome.md`:

- Preferred source filename and gs:// path
- `pdm` local-path import
- CI behavior: synthetic fixture
- `W_res[i,j] = edge j→i`
- `A[i,j] = log1p(synapse_count j→i)` then spectral radius
- Random reservoir is a control, not biology
- Synthetic sentence verbatim

`neural_activity_explorer.md`:

- Caption verbatim
- Four modes
- Animation in browser
- Trace artifact paths
- Topological vs anatomical layout

`fly_connectome_demo.md`:

```bash
.venv/bin/python -m pdm doctor
.venv/bin/python -m pdm train --dataset bearings --arch fly_connectome_reservoir --smoke
.venv/bin/python -m pdm app
```

Only include evaluate/train commands that match actual CLI flags. Note smoke is not quality. Tell the operator to expect the synthetic banner unless they imported MaleCNS.

`fly_connectome_validation.md`:

- Table mapping tests 1–14 → file::test name
- What was **not** validated (full MaleCNS download in CI, Full 30-epoch accuracy)

### README

Do not replace the GRU quickstart. Add a short “Connectome reservoir (optional)” section with synthetic-graph warning.

### Test gaps

If a prior subtask skipped a numbered test, implement it here with tiny synthetic graphs and in-memory windows (reuse `tiny_bearing_tables` / `tiny_filter_tables` from `tests/conftest.py`). Do not require XJTU-SY or HSE files.

Filters test 12: train or call loss with `event=0` vs a counterfactual `event=1` and `duration=0` / `RUL=0`; assert the implementation does not use the counterfactual. Ridge must be unreachable for filters.

## Dependencies

Subtasks 01–05.

## Verification

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check src tests
.venv/bin/python -m pdm doctor
```

Optional help check:

```bash
.venv/bin/python -m pdm train --help
```

`--smoke` runs are optional and not a quality gate.

## Notes on constraints

- All ten hard constraints in the master plan still apply.
- Do not claim MaleCNS was used in CI.
- Do not treat explorer visuals or smoke MAE as model quality.
- GRU/LSTM remain the default path documented first in README.
