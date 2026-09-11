# Subtask 3: Correct survival duration and endpoint masks

## Goal

Fix `build_windows` for filters (observed vs censored) and align bearings train/test/replay endpoint exclusion so survival loss and RUL targets are consistent.

## Context

P0-2: `duration_s` currently uses `observation_end_s - t` even when `event_observed=1`, while `target_rul_s` uses `event_time_s - t`. Weibull NLL uses `duration_s`. Post-event windows (`t >= event_time_s`) must be excluded.

## Acceptance Criteria

- [ ] Observed event: skip `t >= event_time_s`; `duration_s = event_time_s - t`; `event = 1`; `target_rul_s = duration_s`.
- [ ] Censored: skip `t >= observation_end_s`; `duration_s = observation_end_s - t`; `event = 0`; `target_rul_s = None` (NaN in DataFrame).
- [ ] Mid-record event, last-row event, fully censored, and event-boundary window produce correct labels (manual/synthetic checks).
- [ ] Implementation complete; `test_observed_event_duration`, `test_no_post_event_training_windows`, and `test_censored_filter_duration_and_no_post_observation_windows` pass after subtask 11a (not a gate here).
- [ ] Bearings: training windows exclude `t >= event_time_s` (already); replay/evaluate use same rule.
- [ ] Bearings: `actual_rul_s` is NaN for `t >= event_time_s` in evaluate/replay (`attach_actual_rul`, `replay_unit`); endpoint row may display sensor only, never counted in MAE.
- [ ] **Observed filter events** (same rule as bearings): `actual_rul_s` is NaN when `t >= event_time_s` in `attach_actual_rul`, `replay_unit`, and RUL MAE aggregates. Censored histories (`event_observed=0`) still have no point RUL (`actual_rul_s` NaN throughout).
- [ ] `train.py` Weibull path uses corrected `duration_s` only.

## Implementation Notes

**Files:** `src/pdm/windows.py`, `src/pdm/data/filters.py` (unit meta fields), `src/pdm/train.py`, `src/pdm/replay.py`, `src/pdm/evaluate.py`

```python
# Target logic (filters branch in build_windows)
if event_observed:
    if t >= event_time_s: continue
    duration_s = event_time_s - t
    ...
else:
    if t >= observation_end_s: continue
    duration_s = observation_end_s - t
    target_rul_s = None
```

- `filters.py`: ensure `event_time_s` is finite only when `event_observed=1`; censored units keep NaN `event_time_s`.
- Replay may still show sensor at event time; set `actual_rul_s = np.nan` when `t >= event_time_s` (bearings and observed filters).
- Regression owned by subtask 11a (`test_bearings_endpoint_actual_rul_nan`, `test_filter_observed_event_actual_rul_nan`, censored window test).
- Subtask 15 must exclude endpoint rows from all RUL MAE (including zone MAE).

## Dependencies

None (can parallel with 01–02; must finish before subtask 11a).

## Verification

```bash
.venv/bin/python -m ruff check src/pdm/windows.py src/pdm/replay.py src/pdm/evaluate.py
```

Manual: synthetic observed mid-record event → `duration_s = event_time - t`; censored unit → `target_rul_s` NaN; no windows at `t >= event_time_s` / `t >= observation_end_s`. Full gates: subtask 11a.
