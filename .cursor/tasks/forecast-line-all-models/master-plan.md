# Forecast line on every saved architecture

## Overview

Equipment replay already draws a complete causal predicted-RUL line for every `state_mode != "continuous"` model. GRU, LSTM, and window-reset fly/random all go through `window_forecast_history` in `src/pdm/visualization/simulation.py`, called from `_render_equipment_simulation`. The hole is the continuous branch with no `interval_profile.json`: full CNS and continuous fly/random plot `session["predictions"]`, which stores only timestamps the playhead has visited. A seek does not backfill that dict.

`continuous_trace` / `_full_cns_trace` already return `raw_rul_s` for the active acquisition segment. The chart series for that no-profile case must be one row per measurement in the observed prefix, warmup-masked, with gaps reset the same way `continuous_trace` resets them. With a profile loaded, the chart stays `predict_failure_interval(trace["timestamps_s"], trace["raw_rul_s"], profile)` on those trace arrays only. Pre-gap rows are not added to a profiled chart.

The three-way choice (profile, window-reset, continuous raw) moves into one pure function. `_render_equipment_simulation` calls it. A Streamlit-free test imports that function. Project Results (“Forecast horizon”) is a different chart and stays out of scope.

## Phases

1. **Continuous series** — `continuous_forecast_history` in `src/pdm/visualization/simulation.py`. Segment only after the same filter gap recompute `continuous_trace` uses. Reuse the active-segment trace and a separate segment cache. Bearings parity is `replay._continuous_bearing_predictions(..., forecast_profile=None)`.
2. **Chart choice** — `equipment_forecast_frame` in the same module. Profile, window-reset, and continuous no-profile. UI session key for the continuous cache is not `forecast_rows`. Playhead scalar is `points.iloc[-1]`.
3. **Window-reset numeric lock** — extend the existing GRU/LSTM seek-and-export test to window-reset fly and random. Do not change that scorer.

## Dependencies

- Phase 2 calls `continuous_forecast_history` from phase 1 and `window_forecast_history` for window-reset models.
- Phase 3 does not depend on the continuous helper. It locks `window_forecast_history` itself.
- `show_gt`, `actual_rul_s`, and `official_rul_at_prefix_end_s` stay in `_render_equipment_simulation` after the frame exists. Neither helper accepts them.

## Execution order

1. `subtask-01-continuous-forecast-series.md`
2. `subtask-02-equipment-chart-wiring.md`
3. `subtask-03-window-reset-parity.md`

Subtask 03 can be implemented beside subtask 01. Subtask 02 waits on subtask 01.

## Estimated effort

About 4 hours. Subtask 01 ~2 hours, subtask 02 ~90 min, subtask 03 ~45 min. No multi-hour training.

## Verification

Baseline after the epic:

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check src tests
```

Each subtask names the test node ids it owns. Smoke training is not a gate. No Streamlit `AppTest`: both new functions are pure, and the equipment fragment needs a checkpoint plus the playback timer.

## Out of scope

- A numeric signal head for fly or random. `available_signal_engines` in `src/pdm/signal_training.py` keeps those rows `available=False`.
- Passing `rollout_steps` from `src/pdm/project_results_ui.py`.
- New architectures, FastAPI/React, or enabling `filters_full_history`.
- Changing `window_forecast_history`, `Predictor`, or any other window-reset scorer. Subtask 03 only adds a test.
- Changing `predict_failure_interval` math, or feeding the stitched full-prefix series into it. A profiled chart keeps pre-gap rows off the line.
- Passing `show_gt` or `official_rul_at_prefix_end_s` into either helper. The filters official-RUL block in `_render_equipment_simulation` stays an evaluator overlay.
- Scoring a full-CNS prefix once per measurement, or once more for a segment already on `trace`.
