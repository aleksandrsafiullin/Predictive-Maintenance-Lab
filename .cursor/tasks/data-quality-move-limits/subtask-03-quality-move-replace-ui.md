# Subtask 3: Move and replace controls on Data Quality

## Goal
On the open Data Quality tab, let the user move the inspected unit to another set, or swap it with one legal unit from another set, by calling `move_units` or `swap_units` once.

## Context
Subtask 01 provides `preview_swap` and `swap_units`. `move_units` / `preview_move` already exist. `render_quality` in `src/pdm/project_quality_ui.py` has the inspect selectbox `quality_unit_{part}` and no membership controls. The 2026-09-29 expander plan (`subtask-05-quality-move-control.md`) is stale: do not add `st.expander("Move units")` or a multiselect. `test_product_screens_single_primary_button` allows one primary button on this page, `Continue to Training`. `test_import_and_quality_controls_expose_help` currently asserts `Move to` is absent; that assertion is updated here. Suggest-limits is subtask 04. Do not add it in this subtask.

## Acceptance Criteria
- [ ] `render_quality(snapshot, theme="dark", *, storage_mode="owned") -> bool`. `project_ui` passes `storage_mode=str(selected.get("storage_mode") or "owned")`. Return value remains training admission. Existing `render_quality(snapshot, theme)` callers keep working.
- [ ] Owned projects, open tab only (after the inspect selectbox, behind the same `if not is_open: continue` as the chart): `Move to` selectbox of the other two set labels (values are split keys), button `Move unit`; `Replace with` selectbox of legal partner ids labeled `"{unit_id} · {set label}"`, button `Replace unit`. Keys: `quality_move_to:{part}`, `quality_move:{part}`, `quality_replace_with:{part}`, `quality_replace:{part}`. Buttons are not `type="primary"`. No expander, no multiselect.
- [ ] Move `on_click` calls `move_units` with the inspected id only. Replace `on_click` calls `swap_units` with the inspected id and the partner. Both callbacks receive `expected_snapshot_id` for the snapshot this page rendered and pass that argument through. They read the unit ids from `st.session_state` (same pattern as `_on_limits_save`). They do not call `move_units` twice.
- [ ] Preview caption when there is no problem: `After the move: Training {a} · Validation {b} · Testing {c} units.` Replace caption: `{unit_a} joins {set_a}; {unit_b} joins {set_b}. Training {a} · Validation {b} · Testing {c} units.` `{set_a}` is B's set label and `{set_b}` is A's. Problem → `st.warning(problem)` and the commit button disabled. Snapshot id unchanged.
- [ ] Partners are ids in the other two sets for which `preview_swap` has no problem. `author_test` units in Testing stay inspectable and are not partners. Inspecting one disables Move and Replace and shows only `Official HSE test units stay in Testing Data.` Do not also show `QUALITY_REPLACE_NONE` (`No unit in another set can take this place.`). If the partner list is empty for any other reason, do not create an empty selectbox; show that sentence.
- [ ] Testing involved (current tab or move destination; either replace side): caption `Changing Testing units after reviewing results makes later Test scores optimistic.`
- [ ] `storage_mode != "owned"`: one `st.info` above the tabs with `LINKED_LEGACY_MOVE_ERROR` and no move/replace widgets. Limits column still renders.
- [ ] `heavy_job_active()` is read once per run. When true, membership controls are disabled and the caption is `Wait for the current job to finish before changing sets.` Monkeypatch target remains `pdm.project_quality_ui.heavy_job_active` plus `pdm.worker.heavy_job_active` (`_no_job`).
- [ ] Success flashes once above the tabs. `QUALITY_MOVE_DONE` is `Moved {unit} to {destination}. A new data snapshot is active; earlier model runs stay with the previous data snapshot.` `{destination}` is the set label. Replace success stays `Swapped {unit_a} with {unit_b}. A new data snapshot is active; earlier model runs stay with the previous data snapshot.` On success the callback pops `quality_move_to:*`, `quality_replace_with:*`, and `quality_unit_*` before rerun. On `KeyError` / `ValueError` / `RuntimeError` / `OSError` it stores `st.warning` text and does not pop those keys. No traceback. Parent snapshot still loads. Active snapshot id changes. Replace leaves `realized_counts` unchanged.
- [ ] `_reset_project_session` clears prefixes `quality_move_to:`, `quality_move:`, `quality_replace_with:`, `quality_replace:`, `quality_membership_notice`. It does not start clearing every `quality_limit_` key.
- [ ] Help constants from the master plan are on every new widget. `assert not at.multiselect` remains true. Data Quality still has a single primary button, `Continue to Training`.

## Implementation Notes
- Files: `src/pdm/project_quality_ui.py`, `src/pdm/project_ui.py` (keyword `storage_mode`, session-reset prefixes), `src/pdm/ui_copy.py`, `tests/test_project_ui.py`, `tests/test_worker_and_app.py`.
- Import `move_units`, `preview_move`, `preview_swap`, `swap_units`, `SPLIT_LABELS`, and `LINKED_LEGACY_MOVE_ERROR` / `JOB_ACTIVE_MOVE_ERROR` from `pdm.data.project_prepare`. `heavy_job_active` stays a module-level import from `pdm.worker`.
- Draw membership only on the open tab so AppTest sees one `Move to` widget. Keys still include `{part}`.
- Empty replace options must not be passed to `st.selectbox`.
- Assert membership from `load_snapshot` split lists, not from duplicated tab metrics.
- Contract fixture (`make_contract_snapshot`): 9 units. Move a unit out of a set that `preview_move` says can spare it (Training can). Use the Testing tab, whose realized test count is 1 under seed 19, as the empty-set refusal. Read the counts from the snapshot; do not paste magic counts into production code.
- HSE UI case can be a monkeypatched snapshot dict with `units.source_group == "author_test"` plus a real `split`. The API refusal is already covered in subtask 01. This subtask checks the controls hide that id as a partner and disable when it is inspected.
- Do not delete snapshots. Do not train.

## Dependencies
Subtask 01.

## Verification
```bash
.venv/bin/python -m pytest tests/test_project_ui.py tests/test_worker_and_app.py -q -k "quality or product_screens or product_titles or import_and_quality"
.venv/bin/python -m ruff check src/pdm/project_quality_ui.py src/pdm/project_ui.py src/pdm/ui_copy.py tests/test_project_ui.py tests/test_worker_and_app.py
```

New AppTest names in `tests/test_project_ui.py` (real store, `_no_job`, `_quality_app`):

- `test_quality_move_unit_publishes_new_snapshot`
- `test_quality_move_refuses_emptying_a_set`
- `test_quality_replace_keeps_counts_in_one_snapshot`
- `test_quality_membership_disabled_for_linked_legacy_and_active_job`
- `test_quality_testing_tab_shows_optimism_caption`
- `test_quality_hides_official_hse_test_units_as_move_sources`
- `test_quality_membership_failure_warns_without_traceback`

Update `test_import_and_quality_controls_expose_help`: require help on `Move to`, `Move unit`, `Replace with`, and `Replace unit`; delete the assertions that `Move to` and `Move selected units` are absent. Keep `assert not at.multiselect`.

`test_product_screens_single_primary_button` and `test_product_titles_unchanged` must still pass. The tiny Data Quality snapshot in `_product_screens` has one unit per set and no `units` frame; move controls render, the empty-set warning shows, and `at.exception` stays empty.
