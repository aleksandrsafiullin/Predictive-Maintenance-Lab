# Subtask 2: Equipment chart choice

## Goal

Move the three-way equipment forecast choice into a pure function, and make `_render_equipment_simulation` plot and metric that function’s frame.

## Context

`_render_equipment_simulation` in `src/pdm/visualization/simulation_ui.py` chooses the line inline:

1. `profile is not None` — `predict_failure_interval(trace["timestamps_s"], trace["raw_rul_s"], profile)`.
2. `elif not continuous` — `window_forecast_history` and `session["forecast_rows"]`.
3. `else` — a frame of `session["predictions"]` (visited playhead times only).

`build_work_overlay_figure` already draws `predicted_rul_by_time` as the third-row “predicted RUL” line. This subtask replaces the inline choice. The check is a test that imports the choice function, not a reading of the else branch.

The filters official-RUL block (about lines 243–247) recomputes `event` / `actual` / `actual_now` from `official_rul_at_prefix_end_s` after `points` exist. Leave that block as an evaluator overlay. `show_gt` stays the checkbox passed to `build_work_overlay_figure`.

## Acceptance Criteria

- [ ] `equipment_forecast_frame(prefix, model, prep, history_length, *, profile, trace, window_cache=None, segment_cache=None)` lives in `src/pdm/visualization/simulation.py`. `_render_equipment_simulation` is the only caller and passes `feat.iloc[:index + 1]`. The function does not accept a visited-timestamp dict, `show_gt`, or `official_rul_at_prefix_end_s`.
- [ ] `profile is not None` returns exactly `predict_failure_interval(trace["timestamps_s"], trace["raw_rul_s"], profile)`. It does not call `continuous_forecast_history` or `window_forecast_history`. It does not pass the stitched prefix or `gap_before`. Pre-gap rows stay off a profiled chart.
- [ ] `state_mode == "window_reset"` returns `window_forecast_history(prefix, model, prep, history_length, cached=window_cache)` and does not call `continuous_forecast_history`. This includes fly and random.
- [ ] `state_mode == "continuous"` and `profile is None` returns `continuous_forecast_history(...)` and does not call `window_forecast_history`.
- [ ] Test `test_equipment_forecast_frame_choice` in `tests/test_model_comparison.py` imports `equipment_forecast_frame` and locks all of the following without Streamlit:
  - Continuous model, `profile is None`. Monkeypatch `window_forecast_history` to raise, and assert the frame equals `continuous_forecast_history` on the same prefix with `trace` and `segment_cache` forwarded. A 16-row count alone is not enough, because `window_forecast_history` also returns one row per measurement. A forward skip (call at row 1, then at row 15 with no intermediate calls) is that same equality on the 16-row prefix. A one-entry visited dict is not an argument and cannot shrink the frame.
  - Rewind: `segment_cache` filled by the 16-row call, then the same function on the shorter prefix. `len(result)` equals the shorter prefix, and every `timestamp_s <=` that prefix’s last timestamp.
  - `inspect.signature` on `equipment_forecast_frame` and `continuous_forecast_history` rejects `show_gt`, `official_rul_at_prefix_end_s`, and any visited-timestamp parameter.
  - `state_mode="window_reset"` for `FlyConnectomeReservoir` and `RandomReservoir` on `load_synthetic_fixture().graph`: monkeypatch `continuous_forecast_history` to raise, and assert the result equals `window_forecast_history` on that prefix.
  - Profile present: monkeypatch both helpers to raise, and assert the result equals `predict_failure_interval(trace["timestamps_s"], trace["raw_rul_s"], profile)` and no other columns were rebuilt from the full prefix.
  - Playhead is the last row. On a prefix whose trace `status == "predicted"`, `result.iloc[-1]["predicted_rul_s"]` equals `trace["predicted_rul_s"]`. On a prefix whose current row is still warmup after `gap_before`, `result.iloc[-1]["predicted_rul_s"]` is NaN, it equals the trace scalar (None / NaN), and it is not the previous segment’s finite RUL.
- [ ] UI playhead: after the call, if `points.iloc[-1]["predicted_rul_s"]` is finite, that value is `pred`. If it is missing, `pred` is None. The existing metric branch then shows “Forecast” / “Collecting history” and “Remaining life” / “—”. It must not keep a finite RUL from the previous segment or from `session["predictions"]`.
- [ ] Continuous segment cache uses its own session key (`continuous_segments`), cleared in every place that already clears `forecast_rows` (dataset version, cursor clamp, reset, checkpoint change). Window-reset `forecast_rows` stays the `window_forecast_history` cache only. `session["predictions"]` is not written into the chart frame.
- [ ] No new UI copy. No AppTest.

## Implementation Notes

- Dispatch on `profile is not None` first, then `getattr(model, "state_mode", None) != "continuous"`, then the continuous helper. Same order as today’s branches.
- Pass `trace` from the `simulate_step` that just ran. Do not call `continuous_trace` again for the active segment inside the UI.
- Interval for the continuous no-profile branch stays `trace["lower_rul_s"]` / `trace["upper_rul_s"]` when those keys exist. Do not build an empirical band from the stitched series.
- `actual = np.maximum(event - points["timestamp_s"].to_numpy(), 0)` still uses the frame’s timestamps, including the official-RUL overwrite of `event` that already sits below the forecast call. That overwrite must remain after the frame is built.
- Monkeypatch the two helpers in the choice test so a future edit cannot satisfy the assertion by inlining a third implementation that happens to return the same numbers.

## Dependencies

Subtask 01 (`continuous_forecast_history`).

## Verification

`test_equipment_forecast_frame_choice` passes the continuous no-profile case only when `window_forecast_history` is patched to raise and the frame equals `continuous_forecast_history` on that prefix with `trace` and `segment_cache` forwarded. A 16-row length check by itself does not pass this test.

```bash
.venv/bin/python -m pytest tests/test_model_comparison.py::test_equipment_forecast_frame_choice tests/test_continuous_replay.py::test_continuous_forecast_reuses_traced_segment -q
.venv/bin/python -m ruff check src/pdm/visualization/simulation.py src/pdm/visualization/simulation_ui.py tests/test_model_comparison.py
```
