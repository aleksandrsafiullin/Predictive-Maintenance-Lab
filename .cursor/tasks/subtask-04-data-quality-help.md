# Subtask 04: Plain-language help on Projects, Import and Data Quality

## Goal
Give every control a user sets on Projects, Import data and Data Quality (product flow) a plain-English `help=` (what it is + how it changes the result), and add one-line captions where Streamlit has no help slot.

## Context
Data Quality reflects decisions made on Import (split weights, seed, validation/test mode, signal, limits). Today only `"{label} server folder path"` and legacy `"Local file or folder path"` carry `help=`. Files:
- `src/pdm/project_ui.py`: `_render_projects` (L240–283), `_source_input` (L99–106), `_source_widgets` (L116–127), `_render_import` (L157–237).
- `src/pdm/project_quality_ui.py`: `render_quality` (L54–122).
- Legacy (research harness only): `src/pdm/app.py::screen_data` (L559–694), `_render_import` (L226–266).
Facts: yellow/red limits are stored in the schema and used only by `signal_inference._resolved_thresholds` / Results; training (`signal_training.py`) never reads them. Signal column is validated with `r"[A-Za-z_][A-Za-z0-9_]*"` (project_ui L209). Copy rules: `design-spec.md` §7.

## Acceptance Criteria
- [ ] `src/pdm/ui_copy.py` (created in subtask 03) gets these constants under `# Projects and Import` / `# Data Quality`; widgets reference constants, no inline long strings.
- [ ] `help=` added (polish allowed, meaning must stay):

| Widget (label) | Help |
|---|---|
| `Project name` | "A name to find this project later. It does not affect the data or the model." |
| `Source format` (projects) | "The kind of files you will import. It decides which columns are read and which signal is used. It is fixed when the project is created." |
| `Create project` button | "Creates an empty project with this name and format. You import data on the next step." |
| `{label} source` radio | "Upload a folder from this browser, or type a folder path on the computer running the app. Use a path for very large data." |
| `{label} folder` uploader | "Pick the folder that holds this set's files. Each file or subfolder should belong to one physical unit (one machine, bearing, or filter)." |
| `Validation` radio | "Validation data picks the best saved model during training. Automatic holdout sets aside whole units from the Training folder; Separate folder uses units you choose." |
| `Testing` radio | "Test data is kept away from training and model selection, so its score shows how the model may do on new units. Automatic or your own folder." |
| `Train weight (%)` | "Share of automatically split units used to train. More training units usually help the model learn; fewer validation or test units make their scores less reliable." |
| `Validation weight (%)` | "Share of automatically split units used to choose the best saved model. Weights must add up to 100%." |
| `Test weight (%)` | "Share of automatically split units kept for the final, untouched score. Weights must add up to 100%." |
| `Split seed` | "A number that fixes which units land in each set. Same files and seed give the same split; a different seed gives a different random split." |
| `Signal column` | "Exact CSV column holding the measurement to forecast, e.g. vibration or pressure. Must start with a letter or underscore; then letters, digits or underscores only." |
| `Signal name` | "Friendly name shown on charts and results. It does not affect the model." |
| `Signal unit` | "Unit of the signal (e.g. g, Pa, °C). Shown on charts and used for your limits. It does not convert values." |
| `Red condition` | "Whether trouble means the signal going up (for example vibration or filter pressure) or going down. It sets how yellow and red limits are compared." |
| `Yellow limit ({unit})` | "Early-warning level in the signal's unit. Results mark a forecast yellow when it reaches this level. It does not change training." |
| `Red limit ({unit})` | "Action level in the signal's unit. Results mark a forecast red when it reaches this level. For a rising signal it must be above yellow." |
| `Import and check data` button | "Copies the files into this project, checks every row, and splits units into Train, Validation and Test. Runs in the background." |
| `Inspect {name} unit` selectbox | "Choose one physical unit to see its measurements over time. Viewing does not change the data." |
| `Continue to Training` button | "Go to Training. Available when all three sets have usable measurements." |
| `st.metric("Units")` | "Physical units (machines, bearings, filters) in this set. A unit's rows never appear in two sets." |
| `st.metric("Admitted rows")` | "Rows kept after import checks. Rows the import rejected are not counted; any remaining missing values are listed below." |
| `st.metric("Gaps")` | "Breaks in a unit's timeline. The model never learns or forecasts across a gap, so many gaps mean fewer usable windows." |

- [ ] One caption (1 sentence) where there is no help slot:
  - Under the Data Quality `st.tabs`: "Each tab shows one set. Training learns from Training Data, Validation Data picks the best model, Testing Data is scored once at the end."
  - First line inside the `Automatic split weights · 70 / 15 / 15` expander (replace or tighten the existing caption, same meaning).
- [ ] Readiness messages from `training_admission` / eligibility fallback stay factually identical.
- [ ] Legacy `app.py::screen_data`: `help=` on `Inspect & prepare` ("Reads the imported files, checks them, and builds the prepared snapshot used for training. Runs in the background.") and `Unit for sensor plot` ("Choose one unit to plot its raw sensor signal. Viewing does not change the data."). Legacy `_render_import`: `help=` on `Source format`, `Source files`, `Open project`.
- [ ] No change to validation logic, defaults, ranges, keys, labels, or the payload sent to `spawn_worker`.

## Implementation Notes
- `st.metric`, `st.button`, `st.form_submit_button`, `st.file_uploader`, `st.radio`, `st.number_input`, `st.text_input`, `st.selectbox` all accept `help=` in 1.63.
- Keep constants unit-free so they stay testable.
- Coordinate with subtask 03: it edits only the sidebar block of `project_ui.main()`.

## Dependencies
Subtask 02 (copy rules), subtask 03 (creates `ui_copy.py`; this subtask only appends).

## Verification
Add to `tests/test_worker_and_app.py` (helper `_help_of(w) = (getattr(w, "help", None) or w.proto.help or "").strip()`):
- `test_import_and_quality_controls_expose_help`:
  1. Projects screen (empty `PDM_PROJECTS_ROOT`): assert non-empty help on text input `Project name`, selectbox `Source format`, button `Create project`.
  2. Import screen: `project_store().create("Motor", "generic_sensor_csv")`, `at.session_state["project_id"]=pid; at.session_state["project_step"]="Import data"`; `at.run()`. For every widget in `at.radio + at.number_input + at.text_input + at.selectbox` except labels `"Appearance"` and `"Project"`, assert `_help_of(w)`; assert every `at.file_uploader` has help; assert button `Import and check data` has help.
  3. Data Quality: snapshot + monkeypatches from `tests/test_project_ui.py::test_quality_three_sets_and_training_gate`; `at.run()`; assert every selectbox whose label starts with `"Inspect "` has help, every `at.metric` has help, button `Continue to Training` has help, and the tab caption is in `[c.value for c in at.caption]`.
  4. Assert `_help_of(admitted_rows_metric) == ui_copy.QUALITY_ADMITTED_ROWS_HELP` and that constant equals the exact sentence in the table above.
- `test_ui_copy_help_strings_follow_rules`: every `str` constant in `pdm.ui_copy` has `len <= 220`, ends with `.`, contains no `!`, and does not contain the word `likely` (covers `APPEARANCE_HELP` too).
- Existing `tests/test_project_ui.py` stays green (labels unchanged).
- `.venv/bin/python -m pytest tests -q`, `.venv/bin/python -m ruff check src tests`.
- Manual: hover each `?` on Projects, Import and Data Quality at `http://127.0.0.1:8501`.
