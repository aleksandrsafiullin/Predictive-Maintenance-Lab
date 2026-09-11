# Subtask 11a: P0 regression tests — preprocessing & survival

## Goal

Add TZ §9 tests 1–5 plus bearings/filters endpoint `actual_rul_s` mask to `tests/test_spec_invariants.py`.

## Context

Locks subtasks 1–3 before UI/eval work. Split from former subtask 11 to stay within 1–2 h. Use `tiny_bearing_tables` / `tiny_filter_tables` only — do not invent XJTU/HSE fields.

## Acceptance Criteria

- [ ] `test_offline_online_feature_parity` — same raw window → tensor offline vs `Predictor` within 1e-6 (both dust categories, unknown category, NaN).
- [ ] `test_dust_encoding_from_raw` — dust change alters encoded features; not constant fill.
- [ ] `test_saved_median_imputation` — NaN/Inf → scaled median imputation result.
- [ ] `test_observed_event_duration` — `duration_s == event_time - t`, not `observation_end - t`.
- [ ] `test_no_post_event_training_windows` — no windows with `t >= event_time`.
- [ ] `test_bearings_endpoint_actual_rul_nan` — bearings: `actual_rul_s` NaN for `t >= event_time_s`; excluded from MAE.
- [ ] `test_filter_observed_event_actual_rul_nan` — observed filter events: `actual_rul_s` NaN at `t >= event_time_s`; censored units remain NaN throughout.
- [ ] `test_censored_filter_duration_and_no_post_observation_windows` — censored: `duration_s = observation_end_s - t`, `event=0`, `target_rul_s=None`; no windows with `t >= observation_end_s`.

## Implementation Notes

**Files:** `tests/test_spec_invariants.py`, `tests/conftest.py`

- Build minimal trained preprocessor + tiny model checkpoint in-test (`tmp_path`) for parity tests.
- Use `history_length=3` or similar for speed.
- Extend `tiny_filter_tables` with second dust category if needed for encoding test.
- Endpoint test: assert `attach_actual_rul` / replay record at last bearing timestamp has NaN `actual_rul_s`.

## Dependencies

Subtasks 1–3.

## Verification

```bash
.venv/bin/python -m pytest tests/test_spec_invariants.py -q -k "parity or dust or median or event_duration or post_event or endpoint or censored"
.venv/bin/python -m ruff check tests/test_spec_invariants.py
```
