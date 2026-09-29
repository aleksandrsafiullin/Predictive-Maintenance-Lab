# Subtask 5: Data Quality control to move whole units between sets

## Goal
In each Data Quality tab, let the user pick one or more physical units and move them to another set, persisted via `move_units` as a new snapshot, with clear refusals and honest messaging about earlier model runs and Test optimism.

## Context
Subtask 01 provides `move_units`, `preview_move`, and `pdm.worker.heavy_job_active`. Runs are snapshot-bound; Training/Results already filter by `snapshot_id`. `project_ui.main` renders Data Quality via `render_quality(snapshot, theme)`.

## Acceptance Criteria
- [ ] In each tab (`render_quality`), an `st.expander("Move units", expanded=False)` containing:
  - `st.multiselect("Units to move", movable_ids, key=f"quality_move_units:{part}", help=QUALITY_MOVE_UNITS_HELP)`;
  - `st.selectbox("Move to", <the other two set names>, key=f"quality_move_to:{part}", help=QUALITY_MOVE_TO_HELP)`;
  - preview caption from `preview_move`: "After the move: Training {a} · Validation {b} · Testing {c} units.";
  - `st.button("Move selected units", type="primary", disabled=<no selection or problem or heavy_job_active()>, key=f"quality_move:{part}", help=QUALITY_MOVE_SUBMIT_HELP, on_click=_do_move, args=(...))`;
  - `preview_move` problem → `st.warning(problem)`.
- [ ] Official HSE test units are not silently movable (W12): in the Testing tab they are excluded from `movable_ids`, with caption "{n} official HSE test unit(s) are fixed in Testing Data." (`preview_move(...)["fixed_units"]`, empty when `units` is missing — W5).
- [ ] Test-optimism caption (W9): whenever the current tab is Testing or "Move to" is Testing Data: "Changing Testing units after reviewing results makes later Test scores optimistic."
- [ ] Linked legacy projects: expander shows `st.info("This project uses the published split of its source dataset. Create a new project to change the split.")` and no controls.
- [ ] Job lock: controls disabled with caption "Wait for the current job to finish before changing sets." when `pdm.worker.heavy_job_active()` is True (W1 — the same predicate `move_units` re-checks; no second predicate). `render_quality` calls it once per run.
- [ ] `_do_move` (`on_click` callback): calls `move_units(pid, units, dest, expected_snapshot_id=snapshot["snapshot_id"])`; on success pops `quality_move_units:*`, `quality_move_to:*`, `quality_unit_*` keys **inside the callback** (before widgets re-render) and stores a flash message; on `ValueError`/`RuntimeError`/`OSError` stores the message as a warning flash. Main body shows `st.success("Moved {n} unit(s) to {Set}. A new data snapshot is active; earlier model runs stay with the previous data snapshot.")` or `st.warning(...)`; no traceback. Callback-triggered rerun reloads the new snapshot automatically.
- [ ] `render_quality(snapshot, theme="dark", *, storage_mode="owned") -> bool`; positional compatibility kept for existing callers/tests. `project_ui.main` passes `storage_mode=selected["storage_mode"]`. `_reset_project_session` also clears `quality_move` prefixes.
- [ ] `project_training_ui.render_training`: when `list_project_runs` has runs with other `snapshot_id`s, caption "{n} earlier model run(s) were trained on a previous data snapshot and are not shown. Train again on this data." Results nav stays disabled until a completed run on the new snapshot exists (existing `runs_ready` check — verify, do not change).
- [ ] Training admission after the move is recomputed from the new snapshot (automatic via `load_snapshot` on rerun).
- [ ] Every new widget has `help=`; copy in `src/pdm/ui_copy.py`: `QUALITY_MOVE_UNITS_HELP`, `QUALITY_MOVE_TO_HELP`, `QUALITY_MOVE_SUBMIT_HELP`, `QUALITY_MOVE_LEGACY`, `QUALITY_MOVE_DONE`, `QUALITY_MOVE_TEST_OPTIMISM`, `QUALITY_MOVE_FIXED_HSE`, `TRAIN_STALE_RUNS_CAPTION`.

## Implementation Notes
- Files: `src/pdm/project_quality_ui.py`, `src/pdm/project_ui.py` (pass `storage_mode`, session reset), `src/pdm/project_training_ui.py` (stale-run caption), `src/pdm/ui_copy.py`.
- Import `heavy_job_active` from `pdm.worker` at module level in `project_quality_ui.py` so tests can monkeypatch `pdm.project_quality_ui.heavy_job_active`.
- Moves are whole units only — no row/window selection anywhere.
- Do not call `move_units` inside a fragment.
- The zone cache (subtask 04) invalidates itself via the new `snapshot_id`.
- Undo/restore is out of scope.

## Dependencies
Subtasks 01 and 04 (same file; run after 04).

## Verification
`tests/test_project_ui.py` (real store via `make_contract_snapshot`, `PDM_PROJECTS_ROOT` = tmp; monkeypatch `pdm.project_quality_ui.heavy_job_active` → False and `pdm.worker.heavy_job_active` → False):
- `test_quality_move_unit_publishes_new_snapshot` — select one Training unit, "Move to" Validation, click; `project_store().get(pid)["active_snapshot_id"]` changed, success message shown, Validation tab lists the unit, old snapshot still loads.
- `test_quality_move_refuses_emptying_a_set` — select all Validation units → warning text present, button disabled, snapshot id unchanged.
- `test_quality_move_disabled_for_linked_legacy_and_active_job` — `storage_mode="linked_legacy"` → info text, no multiselect; `heavy_job_active` → True → button disabled + wait caption.
- `test_quality_move_testing_shows_optimism_caption` — Testing tab (`quality_tab="Testing Data"`) or "Move to" Testing Data → caption "Changing Testing units after reviewing results makes later Test scores optimistic." present.
- `test_quality_move_hides_official_hse_test_units` — monkeypatched snapshot with `units.source_group == "author_test"` → those ids absent from the multiselect options, fixed-unit caption present.
- `test_training_page_mentions_runs_from_previous_snapshot` — fake `list_project_runs` returning a run with an old `snapshot_id`.
- `tests/test_worker_and_app.py::test_import_and_quality_controls_expose_help` extended to assert help on the move widgets.
- Manual at `http://127.0.0.1:8501`: move a unit; Training shows stale-run caption; Results disabled until retraining.
```bash
.venv/bin/python -m pytest tests/test_project_ui.py tests/test_worker_and_app.py -q -k "quality or project or import"
.venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check src tests
```
