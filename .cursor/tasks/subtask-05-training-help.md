# Subtask 05: Plain-language help on every Training parameter

## Goal
Every Training control a user sets gets a plain-English `help=` that says what it is and how it changes the model or result, grounded in what `src/pdm/signal_training.py` actually does; add short captions for model choice and saved-run metrics.

## Context
`src/pdm/project_training_ui.py::render_training` (L69–127): `Model` selectbox; form with `History samples`, `Forecast horizons (seconds)` (only one with help today), `Random seed`, then GRU/LSTM-only `Training epochs`, `Hidden units`, `Batch size` or boosting-only `Boosting iterations`; `Train model`. `Stop job` renders only inside the `_job_status` fragment (L40–66) while a job is running. Metrics: `Validation mean absolute error`, `Held-out Test mean absolute error`.
Legacy research path: `src/pdm/future_red_ui.py::render_training` (L174–288) `Model families` multiselect, `Start training`; `render_comparison` `Models to compare`.

Facts from `src/pdm/signal_training.py` the copy must match:
- `history_length` = consecutive past measurements per window; windows never cross gaps (`_segments`); too long → "Train and validation each need eligible, unbroken signal windows".
- `horizons_s` = seconds ahead, one output each. A target counts only if a real measurement lies within `min(1% of median step, 0.01 s)` of `as_of + horizon` (L61, L113). Default = 1×, 2×, 3× median sampling interval (`_default_horizons`). Every horizon needs train and validation targets (L330).
- GRU/LSTM: AdamW over `epochs`; validation MAE after each epoch; **best epoch kept** (L237–267).
- `hidden_size` = recurrent memory size; `batch_size` = windows per update; `seed` fixes init and shuffling.
- Quantile boosting: 5% / 50% / 95% quantile models per horizon (L286) → point forecast plus a rough, uncalibrated range (`interval_status="unvalidated_pointwise_quantiles"`, L376); tries `max_iter // 2` and `max_iter`, keeps lower validation MAE (L275–309).
- MAE is `unit_equal_mae` (each unit weighted equally, `_weights` L131), in the signal's unit.
- **Stop:** on a stop request the run's `manifest.json` is deleted (L387) — nothing from a stopped run is saved; earlier completed runs are untouched.

## Acceptance Criteria
- [ ] Constants in `src/pdm/ui_copy.py` under `# Training`, referenced from widgets:

| Widget | Help |
|---|---|
| `Model` | "The forecasting method. GRU and LSTM are neural networks that learn patterns over time; Quantile boosting is a fast tree method that also gives a rough 5–95% range (not calibrated). Try GRU first." |
| `History samples` | "How many past measurements the model sees for each forecast. More history can capture slower trends but needs longer unbroken records; too long and training finds no usable windows." |
| `Forecast horizons (seconds)` | "How far ahead to forecast, in seconds, separated by commas. Use multiples of your sampling interval (default: 1×, 2×, 3×); a horizon with no matching measurement cannot be trained." |
| `Random seed` | "Fixes the random start and shuffling. Same data and seed give repeatable results; change it to check that a result is not luck." |
| `Training epochs` | "Epoch = one full pass over the training data. More epochs give the model more chances to improve but take longer; the app keeps the epoch that scored best on Validation." |
| `Hidden units` | "Size of the model's memory. Larger can learn more complex patterns but trains slower and may memorize small datasets. 32 is a good start." |
| `Batch size` | "How many examples the model looks at before each update. Smaller is noisier but updates more often; larger is smoother and faster on big data." |
| `Boosting iterations` | "Number of trees added one after another. More can fit finer detail but may overfit; the app also tries half this number and keeps whichever scores better on Validation." |
| `Train model` button | "Starts training in the background with these settings. Training uses Train units, picks the best model on Validation, then scores Test once." |
| `Stop job` button (`TRAIN_STOP_HELP`) | "Asks the job to stop after its current safe step. No model is saved from a stopped run; earlier saved runs stay available." |

- [ ] `st.metric` help: `Validation mean absolute error` → "Average size of the forecast error on Validation units, in the signal's unit; each unit counts equally. Lower is better. This set was used to pick the model."; `Held-out Test mean absolute error` → "Average forecast error on Test units, which the model never saw during training or selection; each unit counts equally. The fairest estimate for new units."
- [ ] Caption under `Model` replaced with one sentence stating the model forecasts the signal itself (not remaining life) in its native unit (same meaning as L83).
- [ ] Legacy `future_red_ui`: `Model families` help "Which model types to train on the same data and target. Each one is trained and scored separately so you can compare them."; `Start training` help "Starts one background run that trains each selected model, picks checkpoints on Validation and scores Test after freezing."; `Models to compare` gets a one-line help.
- [ ] No change to defaults (8, 12, 32, 32, 40, 42, horizon default), ranges, steps, labels, form structure, or the `params` dict sent to `spawn_worker`.

## Implementation Notes
- `help=ui_copy.TRAIN_HISTORY_HELP` etc.; `Stop job` uses `help=ui_copy.TRAIN_STOP_HELP`.
- `st.form_submit_button` supports `help=`.
- If code and copy disagree, code wins and copy is fixed.

## Dependencies
Subtask 02 (copy rules), subtask 03 (`ui_copy.py`).

## Verification
Add to `tests/test_worker_and_app.py`:
- `test_training_controls_expose_help`: `root = tmp_path / "projects"`, `monkeypatch.setenv("PDM_PROJECTS_ROOT", str(root))`, `_, project, _ = make_contract_snapshot(root)`; set `project_id`/`project_step="Training"`; `at.run()`. Assert `_help_of` non-empty for `Model`, `History samples`, `Forecast horizons (seconds)`, `Random seed`, `Training epochs`, `Hidden units`, `Batch size`, and the `Train model` button. Then `next(w for w in at.selectbox if w.label == "Model").set_value("quantile_boosting"); at.run()`; assert `Boosting iterations` has help and `Model` help contains `"5–95% range (not calibrated)"`.
- `test_training_stop_help_copy`: `assert ui_copy.TRAIN_STOP_HELP == "Asks the job to stop after its current safe step. No model is saved from a stopped run; earlier saved runs stay available."` and `assert "ui_copy.TRAIN_STOP_HELP" in Path(project_root() / "src/pdm/project_training_ui.py").read_text()` (the button can only render during a live job, so the constant and its use are asserted statically).
- `test_training_mae_help_mentions_equal_units`: both MAE help constants contain `"each unit counts equally"`.
- `test_future_red_training_controls_expose_help`: legacy harness with `tiny_bearing_tables` as in `test_training_button_dispatches_selected_models`; `Model families` multiselect and `Start training` button have non-empty help.
- Existing: `tests/test_project_ui.py::test_saved_snapshot_offers_one_signal_engine_and_horizon_controls`, `tests/test_worker_and_app.py::test_training_button_dispatches_selected_models`, `tests/test_future_red_ui.py` green (same labels, same dispatched job).
- `.venv/bin/python -m pytest tests -q`, `.venv/bin/python -m ruff check src tests`.
