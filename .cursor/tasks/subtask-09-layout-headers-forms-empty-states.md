# Subtask 09: Layout — page headers, grouped forms, empty states, one primary action

## Goal
Make the small Streamlit layout changes the design spec needs on the five product screens: consistent page header, grouped Import and Training forms, §5 empty-state cards, and exactly one primary button per screen — without changing labels, keys, defaults, or behavior.

## Context
Files: `src/pdm/ui_theme.py` (helpers), `src/pdm/project_ui.py`, `src/pdm/project_quality_ui.py`, `src/pdm/project_training_ui.py`, `src/pdm/project_results_ui.py`; `src/pdm/app.py::_workflow_nav` (legacy nav labels only). Screen notes: `design-spec.md` + subtask 02 per-screen notes.

## Acceptance Criteria
- [ ] `ui_theme.page_header(title: str, description: str)` renders `st.title(title)` + a description line with class `pdm-page-desc` (styled in CSS). Used on Projects, Import data, Data Quality, Training, Results. Title strings unchanged (`Projects`, `Import data`, `Data Quality`, `Training`, and the Results title as it is today); descriptions reuse the existing first `st.write` text.
- [ ] `ui_theme.empty_state(title: str, body: str)` renders the §5 card (`st.container(border=True, key="pdm-empty-…")`). Applied to: Projects "No projects yet…", Data Quality "Import data to inspect Train, Validation, and Test.", Training "No signal engine is eligible…", fallback "Finish importing and checking…". The body keeps the existing sentence verbatim so AppTest string assertions still match (`rg -n "No projects yet|Import data to inspect|No signal engine|Finish importing" tests`).
- [ ] Import form grouping (non-form widgets stay non-form): bordered containers "Training source", "Validation and Testing", "Signal and limits"; limits in 2 columns, split weights in 3 columns.
- [ ] Training form grouping inside the existing `st.form("signal_train_form")`: "Data window" (History samples, Forecast horizons), "Model size" (Hidden units + Batch size for GRU/LSTM; Boosting iterations for Quantile boosting), "Repeatability" (Random seed; Training epochs for GRU/LSTM only). For Quantile boosting the Repeatability group shows only Random seed plus one caption: "Quantile boosting has no epoch count; it tries two tree counts and keeps the better one on Validation." Widget labels, defaults, ranges and the submitted `params` dict unchanged.
- [ ] Exactly one primary button per product screen: Projects `Create project`; Import `Import and check data`; Data Quality `Continue to Training` (when ready); Training `Train model`; Results none or its single main action. Sidebar nav keeps `type="primary"` for the selected step (excluded from the count by key prefix `project_nav:`).
- [ ] Legacy `_workflow_nav` drops `✓`/`01` markers from labels only if no test depends on them (`rg -n "✓|\"01" tests`); otherwise untouched.
- [ ] No emoji or decorative glyph in any product label or chrome.
- [ ] Do not move non-form widgets into `st.form` (changes submit semantics).

## Implementation Notes
- Use `st.container(key="pdm-…")` → `.st-key-pdm-…` for targeted CSS (existing pattern `.st-key-lab_controls`); add the matching rules to `explorer.css` with vars only.
- `st.container(border=True)` is allowed inside `st.form`.
- Re-run `tests/test_project_ui.py` after each screen; AppTest finds widgets by label regardless of container.

## Dependencies
Subtasks 04, 05 (help strings present), 08 (CSS rules to hang classes on).

## Verification
Add to `tests/test_worker_and_app.py`:
- `test_product_screens_single_primary_button`: Projects (empty root), Import data (generic project), Data Quality (snapshot fixture from `tests/test_project_ui.py::test_quality_three_sets_and_training_gate`, engines available), Training (`make_contract_snapshot`). For each, `primaries = [b for b in at.button if b.proto.type == "primary" and not str(b.key or "").startswith("project_nav:")]`; assert `len(primaries) == 1`. Form submit buttons: include `at.button` entries from forms (AppTest exposes `form_submit_button` as buttons; verify and adjust accessor if needed).
- `test_product_titles_unchanged`: `[t.value for t in at.title]` equals `["Projects"]`, `["Import data"]`, `["Data Quality"]`, `["Training"]` on the respective screens.
- `test_training_form_boosting_has_no_epochs`: Training screen, set `Model` to `"quantile_boosting"`, `at.run()`; assert no number input labelled `Training epochs` and the boosting caption present; set back to `"gru"`; `Training epochs` present.
- Existing: `tests/test_project_ui.py`, `tests/test_project_replay.py`, `tests/test_worker_and_app.py` green.
- `.venv/bin/python -m pytest tests -q`, `.venv/bin/python -m ruff check src tests`.
- Manual (required): `http://127.0.0.1:8501`, compare against design-spec per-screen notes in Light and Dark; screenshots to `runs/_ui_review/after/<screen>-<theme>.png`; ui-designer returns PASS or a punch list, fixed before 10.
