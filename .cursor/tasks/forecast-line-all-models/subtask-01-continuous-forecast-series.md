# Subtask 1: Continuous forecast series

## Goal

Add `continuous_forecast_history` so a continuous model with no interval profile returns one causal predicted-RUL row per measurement in the observed prefix, reusing the active-segment trace.

## Context

`window_forecast_history` walks every prefix with `Predictor`. That matches window-reset models and is the wrong loop for full CNS: each `predict_from_history` reruns `continuous_trace`.

`continuous_trace` in `src/pdm/visualization/simulation.py` already recomputes filter gaps, then keeps only the last segment:

```python
if prep.dataset_id == "filters":
    frame = recompute_filter_gap_before(
        frame, gap_multiplier=prep.gap_multiplier,
        sampling_interval_s=prep.sampling_interval_s, causal=True)
```

`replay._continuous_bearing_predictions` (`src/pdm/replay.py`) stitches every segment for bearings and warmup-masks `predicted_rul_s` (`seen >= history_length`, else NaN; `forecast_profile is None` → `continuous_raw_readout`). That bearings fast path does not recompute filter gaps. The chart helper must not copy that skip. On filters it splits only after `recompute_filter_gap_before` with `prep.gap_multiplier` and `prep.sampling_interval_s`.

`simulate_step` has already filled `trace["timestamps_s"]` and `trace["raw_rul_s"]` for the active segment, including `_full_cns_trace` when the model has `pool_index`.

## Acceptance Criteria

- [ ] `continuous_forecast_history(measurements, model, prep, history_length, *, trace=None, cached_segments=None)` lives in `src/pdm/visualization/simulation.py`. It returns a `DataFrame` with one row per row of `measurements`, columns `timestamp_s`, `raw_rul_s`, `predicted_rul_s`. It does not accept `show_gt`, `official_rul_at_prefix_end_s`, an event time, or a profile. It does not call `window_forecast_history`, `predict_failure_interval`, or a per-row `Predictor` loop. `inspect.signature(continuous_forecast_history)` rejects `show_gt`, `official_rul_at_prefix_end_s`, and any visited-timestamp parameter, asserted inside `test_continuous_forecast_reuses_traced_segment`.
- [ ] Bearings test `test_continuous_forecast_reuses_traced_segment` in `tests/test_continuous_replay.py` composes the production call. Build a bearings prefix that contains `gap_before`. Build `trace` with `continuous_trace` on the post-gap rows only (the slice `continuous_trace` itself keeps). On the second call, `cached_segments` contains only earlier segments, not the active one; the active segment’s `raw_rul_s` equals `trace["raw_rul_s"]`, and `forward_states` is not called. A cache that already holds the active segment must not be what the test passes in, or the spy is meaningless. Assert `len(result) == len(prefix)` and `predicted_rul_s` equals `replay._continuous_bearing_predictions(prefix, predictor, None)["predicted_rul_s"]` with `equal_nan=True`, `rtol=0`, `atol=0`. The same test asserts a shorter prefix that reuses that earlier-segment cache returns no `timestamp_s` past the shorter prefix.
- [ ] The same test runs once on the `LeakyESN` from `continuous_case` and once on a 4-node `FullCNSReservoir` with `pool_index` set (shape as `_mini_full_cns_reservoir` in `tests/test_future_red_full_cns.py`). The full-CNS body must keep `pool_index` so `_full_cns_trace` runs. Do not load a real MaleCNS graph.
- [ ] Filters test `test_filter_forecast_ignores_poisoned_gap_flags` in `tests/test_continuous_replay.py`. Prefix `dataset_id == "filters"`. Stored `gap_before` disagrees with `recompute_filter_gap_before(..., gap_multiplier=prep.gap_multiplier, sampling_interval_s=prep.sampling_interval_s, causal=True)`. `predicted_rul_s` equals, row for row, `Predictor.predict_from_history` on each causal prefix (`equal_nan=True`, `rtol=0`, `atol=0`). At least one row differs from a series segmented on the stored poisoned flags. Tiny continuous model only (the `LeakyESN` path is enough here).
- [ ] Warmup rows inside a segment are NaN in `predicted_rul_s`. After warmup they equal that segment’s raw readout. A gap starts the count again. The shorter-prefix timestamp bound is the assertion inside `test_continuous_forecast_reuses_traced_segment`, not a separate test.

## Implementation Notes

- On `prep.dataset_id == "filters"`, copy the frame and call `pdm.windows.recompute_filter_gap_before` with `prep.gap_multiplier`, `prep.sampling_interval_s`, and `causal=True` before any segment split. On bearings, split the stored `gap_before` column. Same rule as `continuous_trace`: filters recompute, bearings do not.
- Segment starts: `unique([0, flatnonzero(gap_before)])` on that post-recompute frame. One `continuous_trace` per earlier segment. `seen` resets at each start. `predicted_rul_s` is the raw readout where `seen >= history_length`, else NaN.
- Active segment: if `trace["timestamps_s"]` equals that segment’s timestamps and `trace["raw_rul_s"]` has the same length, copy the trace. Do not call `forward_states` for it. If the trace does not match, score the segment; do not mis-align a stale trace.
- Return earlier-segment arrays via `cached_segments` (the object the caller stores). Include the same model identity `continuous_trace` uses (`id(model)`, weight versions, leak) plus segment start timestamp and length. Ignore cache entries that extend past this prefix or fail the identity check.
- `measurements` is already one unit and already the causal prefix (`timestamp_s <= now`). Do not read later rows from a longer frame.
- Do not stitch this frame into `predict_failure_interval`. Profiled charts are subtask 02 and use the trace arrays only.

## Dependencies

None. No Streamlit edits in this subtask.

## Verification

`test_continuous_forecast_reuses_traced_segment` passes only when the second call’s `cached_segments` omit the active segment, that segment’s `raw_rul_s` equals `trace["raw_rul_s"]`, and the `forward_states` spy stays at zero. A green result that fed the spy a cache already holding the active segment does not count.

```bash
.venv/bin/python -m pytest tests/test_continuous_replay.py::test_continuous_forecast_reuses_traced_segment tests/test_continuous_replay.py::test_filter_forecast_ignores_poisoned_gap_flags -q
.venv/bin/python -m ruff check src/pdm/visualization/simulation.py tests/test_continuous_replay.py
```
