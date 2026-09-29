# Subtask 3: Create ML-style three-card Import data page

## Goal
Replace the two-box source layout in `src/pdm/project_ui.py` with three equal bordered cards (Training Data, Validation Data, Testing Data), each with its own source row, auto-split state, and Units / Admitted rows / Gaps from the active snapshot.

## Context
Current `_source_widgets` renders "Training source" + "Validation and Testing" radios; `_render_import` puts weights/seed in a collapsed expander. The submitted `source` dict (`primary`, `validation_mode`, `validation`, `test_mode`, `test`, `seed`, `weights`, signal, thresholds) and `project_import` job stay unchanged. Reference: Create ML screenshot — three cards in a row, counts + "View", Validation "Automatically split from training data" with a Split selector, source row at the bottom.

## Acceptance Criteria
- [ ] `_source_widgets` replaced by `_import_cards(store, project, summaries: dict | None) -> tuple[primary, validation, test, val_mode, test_mode]`; cards via `st.columns(3, gap="medium")`, each `st.container(border=True, key="pdm-import-card-{train|validation|test}")`, title `st.subheader` with the `PARTS` names from `project_quality_ui.py`.
- [ ] Card counts (top): if an active snapshot exists → three `st.metric`s "Units", "Admitted rows", "Gaps" with `QUALITY_UNITS_HELP` / `QUALITY_ADMITTED_ROWS_HELP` / `QUALITY_GAPS_HELP`, and a caption "Saved data" so the counts are clearly the saved snapshot, not the folder just picked. No snapshot → "No data yet" + "Import to see units, rows, and gaps." (no zeros, no fake counts).
- [ ] Counts are not recomputed with a full `load_snapshot` hash check on every rerun (W11): `_snapshot_summaries(project_id, snapshot_id)` decorated with `st.cache_data` (key = those two strings; snapshot is immutable) loads the snapshot once and returns `{part: {"units", "rows", "gaps"}}` from `part_summary` (drop the `frame`). Loaded only when `project["state"] == "ready" and project["active_snapshot_id"]`; any load error → empty state + `st.caption` with the error, never a crash. Acceptance: displayed counts equal `part_summary` for each part.
- [ ] "View" (tertiary button, `key="import_view:{part}"`, `help=IMPORT_VIEW_HELP`) uses `on_click` to set `st.session_state["quality_tab"] = "<Tab label>"` and `st.session_state["project_step"] = "Data Quality"` (W2).
- [ ] Data Quality tabs (only edit to `project_quality_ui.render_quality` in this subtask): `st.tabs([name for _, name in PARTS], key="quality_tab", on_change="rerun")` with **no** `default=`. Selection is driven by the `quality_tab` session key.
- [ ] Validation and Testing cards: `st.selectbox("Validation data from" / "Testing data from", ["Split from training", "Separate folder"], key="validation_mode" / "test_mode", help=…)`. Auto → caption "Automatically split from training data" + effective share, e.g. "15% of the training pool" (renormalized over auto groups; if Testing is a separate folder, Validation shows `round(100 * val / (train + val))`%). For `source_kind == "hse_filters"` with Testing auto, Testing shows "Official HSE test units (Test_Data_CSV.csv) in the Training folder stay in Testing." instead of a share. Separate folder → `_source_input(...)` inside the card.
- [ ] Training card source row: `_source_input(store, pid, "Training", f"{pid}:primary")` — keeps "Choose folder" / "Server folder path" and `accept_multiple_files="directory"`; uploader label stays "Training folder".
- [ ] Split weights/seed (W3): widgets have stable keys `import_weight_train`, `import_weight_validation`, `import_weight_test`, `import_seed` (defaults 70/15/15/42). Cards read `st.session_state.get(key, default)` **before** the popover widgets render, so shares shown on cards match the inputs. The popover `st.popover("Split settings", key="import_split_settings", help=IMPORT_SPLIT_SETTINGS_HELP)` has a fixed label and key (current values shown in a caption next to it, e.g. "70 / 15 / 15 · seed 42") so value changes don't close it. Shown when at least one holdout is auto; otherwise caption "Both holdouts use separate folders; split settings do not apply." Same `number_input` labels/help constants and the "Split weights must add to 100%." check.
- [ ] Caption under the cards when the active snapshot's `split["protocol"] == "whole_unit_project_v1_manual"` (available via the cached summary): "Current sets include manual moves from Data Quality. Importing again creates a fresh split."
- [ ] "Signal and limits" container stays below, unchanged. Submit flow, validation messages, upload staging/cleanup unchanged.
- [ ] `_reset_project_session` also pops `quality_tab`, `import_weight_*`, `import_seed`, `import_split_settings`, and `import_view:` prefixed keys.

## Implementation Notes
- New copy in `src/pdm/ui_copy.py`: `IMPORT_VALIDATION_FROM_HELP`, `IMPORT_TEST_FROM_HELP` (reuse text of `IMPORT_VALIDATION_MODE_HELP`/`IMPORT_TEST_MODE_HELP`), `IMPORT_SPLIT_SETTINGS_HELP`, `IMPORT_VIEW_HELP`, `IMPORT_CARD_EMPTY`.
- Selector values → modes: "Split from training" → `auto`, "Separate folder" → `folder`.
- Pure helper `_auto_shares(weights, val_mode, test_mode) -> dict[str, float]` mirroring `allocate_project_split` renormalization; unit-tested.
- The popover's number inputs must still be rendered on every run (Streamlit keeps widget state only for rendered widgets); `st.popover` content is always executed, so this holds.
- W4: `tests/test_project_ui.py::test_failed_worker_launch_cleans_this_attempt_uploads` monkeypatches `pdm.project_ui._source_widgets` with `lambda *_args: (...)`. Change the patch target to `pdm.project_ui._import_cards` (same return tuple; `*_args` already absorbs the extra `summaries` argument). Do not keep a dead `_source_widgets` wrapper.
- Other tests to update: `test_create_two_persistent_projects_and_simple_navigation` asserts radios "Validation", "Testing" → selectboxes "Validation data from", "Testing data from". `tests/test_worker_and_app.py::test_import_and_quality_controls_expose_help` requires `help=` on every radio/number/text/select on Import → keep help on all new widgets (popover inputs included). Check `test_browser_import_persists_both_required_filter_files` (~line 2388) for old labels.
- Don't touch legacy `app.py` `_render_import`.

## Dependencies
Subtask 01 (manual-move protocol name for the caption). No code dependency on 02.

## Verification
`tests/test_project_ui.py`:
- `test_import_page_three_cards_empty_state` — fresh project: subheaders "Training Data", "Validation Data", "Testing Data"; no "Units" metric; "No data yet"; uploader "Training folder"; selectboxes default "Split from training"; caption "Automatically split from training data".
- `test_import_cards_show_snapshot_counts_and_view` — `make_contract_snapshot`: per-card Units/Admitted rows/Gaps equal `part_summary`; click "View" on Testing → `at.session_state["project_step"] == "Data Quality"` and `at.session_state["quality_tab"] == "Testing Data"` (AppTest `Tab` does not expose selection).
- `test_import_separate_folder_card_and_renormalized_share` — Testing "Separate folder" → "Testing folder" uploader appears; Validation caption shows 18%.
- `test_import_split_settings_keys_drive_card_shares` — set `import_weight_validation` to 20 and `import_weight_train` to 65 via the number inputs → Validation card caption shows 20%.
- Unit test for `_auto_shares`.
- Updated `test_failed_worker_launch_cleans_this_attempt_uploads` passes.
- `tests/test_worker_and_app.py::test_import_and_quality_controls_expose_help` passes.
- Manual: `.venv/bin/python -m pdm app` → `http://127.0.0.1:8501`, Import data in light and dark; compare with the Create ML reference.
```bash
.venv/bin/python -m pytest tests/test_project_ui.py tests/test_worker_and_app.py -q -k "import or project or quality"
.venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check src tests
```
