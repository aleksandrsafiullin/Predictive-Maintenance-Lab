# Subtask 3: Window-reset fly and random parity

## Goal

Lock numeric seek, rewind, and export parity of `window_forecast_history` for window-reset fly and random reservoirs. Do not change the scorer.

## Context

`tests/test_model_comparison.py::test_seek_and_replay_export_use_identical_forecasts` is parametrized with `architecture in ("gru", "lstm")` only. Those models, and window-reset fly/random, already share `window_forecast_history`. Subtask 02 locks the chart choice: window-reset fly and random call that helper and do not call `continuous_forecast_history`. This subtask locks the numbers against `replay_unit`.

`tests/test_training_v2.py::test_feature_recipe_replay_matches_ordinary_prediction` stays a GRU + `degradation_v1` filters case.

Changing `window_forecast_history` or `Predictor` is out of scope. If the new cases fail, stop and report. Do not patch the scorer inside this epic.

## Acceptance Criteria

- [ ] `test_seek_and_replay_export_use_identical_forecasts` runs for `fly_connectome_reservoir` and `random_reservoir` with `state_mode="window_reset"`, plus the existing gru and lstm cases. One assertion body.
- [ ] Cold `window_forecast_history` on the prefix, a rewind to an earlier row with `cached=` from the cold frame, and a continue that passes the rewind cache: `pd.testing.assert_frame_equal` on the overlapping timestamps, same as the GRU/LSTM test today.
- [ ] `points["predicted_rul_s"]` matches `replay_unit(...)["predictions"]["predicted_rul_s"]` with `np.testing.assert_allclose(..., equal_nan=True, rtol=0, atol=0)`.
- [ ] The rewind call passes a partial `cached=` dict and still matches the cold call on every timestamp the cache already held.
- [ ] Models are tiny and in memory. Fly: `FlyConnectomeReservoir(load_synthetic_fixture().graph, input_size=len(prep.feature_names), state_mode="window_reset", time_scale_s=prep.time_scale_s)`. Random: `RandomReservoir` on that same synthetic parent graph. No saved checkpoint and no full MaleCNS.
- [ ] No production edit in `src/pdm/visualization/simulation.py`, `src/pdm/predict.py`, or `src/pdm/replay.py` for this subtask. GRU, LSTM, and `test_feature_recipe_replay_matches_ordinary_prediction` stay as they are.
- [ ] No Streamlit change. Routing of these models through `equipment_forecast_frame` is subtask 02, not this file.

## Implementation Notes

- Reuse `tiny_bearing_tables`, `fit_preprocessor("bearings", ...)`, `history_length=5`, and the `Bearing1_1` prefix of 16 rows from the current test. Default recipe is `base_v1`, so the helper takes `predict_from_history`.
- Branch only the model factory. `PDMNet` cannot construct fly or random.
- Default reservoir head is `rul`. Do not require `lower_rul_s` / `upper_rul_s`.
- Do not add `show_gt`. Do not construct `state_mode="continuous"` here.

## Dependencies

None. Independent of subtasks 01 and 02.

## Verification

```bash
.venv/bin/python -m pytest tests/test_model_comparison.py::test_seek_and_replay_export_use_identical_forecasts tests/test_training_v2.py::test_feature_recipe_replay_matches_ordinary_prediction -q
.venv/bin/python -m ruff check tests/test_model_comparison.py
```
