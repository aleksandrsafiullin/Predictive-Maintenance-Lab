# Subtask 3: Last-admissible-time coverage (R4, v1)

## Goal

Score truncated histories as `insufficient_coverage` unless unit `observation_end_s` extends through the **last admissible confirmation time**. Stop treating “reached the start of the H-window” as enough to call a miss. Keep v1 scoped to `observation_end_s` (no interior-gap/K scoring this patch).

## Context

Review R4 (HIGH). `has_sufficient_coverage` is `observation_end_s >= event_time_s - max(H_trigger, minimum_action_lead_time)`. That is the start of the chance interval, not the last time a warning could still be timely.

Counterexample (must pass): event `t=100`, `H_trigger=30`, `minimum_action_lead_time=10`, observation ends `t=75`. Current code: `75 >= 70` → `miss`. A confirm at t=85 would be timely (lead=15); those samples simply do not exist. Correct: `insufficient_coverage`.

**Plan-review Warning:** `classify_alert_outcome` only sees unit `observation_end_s`. Interior-gap / K-sample tests cannot pass with obs_end-only. `min_lead==0` with `obs_end >= event` must not treat a sample **at** the event as a completed chance to warn.

`test_insufficient_coverage_not_false_miss` **must be updated** to the new definition. Do not leave it asserting `coverage_horizon = max(H, min_lead)`. Filter prefix rows with `official_rul_overlay` stay **eval annotation**, not a 600 Pa event.

Bump `METRICS_VERSION` in `src/pdm/evaluate.py` (`"v0"` → `"v1"`) because denominators change.

## Acceptance Criteria

- [ ] **v1 bound:** last admissible confirmation time is `event_time_s - minimum_action_lead_time` **inclusive** when `minimum_action_lead_time > 0`. `has_sufficient_coverage` uses unit `observation_end_s` vs that bound only.
- [ ] If `minimum_action_lead_time == 0`, a timely alert remains **strictly before** the event (`alert_time < event_time`). Coverage is **not** sufficient if the only evidence is a sample **at** the event (`observation_end_s >= event_time_s` with no timestamp `< event`). Require at least one timestamp strictly before the event (when scoring from predictions/steps, use those timestamps; obs_end-only fallback: `observation_end_s < event_time_s`).
- [ ] `H_trigger` is **not** the coverage bound. It remains the AlertEngine trigger threshold only.
- [ ] Example 100 / 75 / 30 / 10 is **not** `miss`.
- [ ] No timely alert **and** v1 coverage is sufficient **is** `miss`.
- [ ] Data coverage vs prediction availability: document as separate; missing sensors ≠ `Collecting history`. A v1-covered interval with no timely alert is still `miss` even if the model never emitted a finite RUL. Optional `n_units_no_prediction_in_window` must **not** be required for this subtask.
- [ ] Docstrings on `classify_alert_outcome` / `has_sufficient_coverage`: K-step confirm, interior gaps, and warmup are **known v1 caveats** (not scored this patch). Do **not** add dedicated interior-gap or K-sample coverage tests in this subtask.
- [ ] `METRICS_VERSION` bumped; asserts that read the constant still pass.

## Implementation Notes

**Files:** `src/pdm/alerts.py` (`coverage_horizon_s`, `has_sufficient_coverage`, `classify_alert_outcome`; add `last_admissible_confirmation_time_s` rather than `max(H, lead)`). `src/pdm/evaluate.py` (`METRICS_VERSION`; `summarize_alert_metrics` only if min_lead==0 needs step timestamps already in `steps`). Docstrings only — no new markdown files.

v1 coverage:

```text
if min_lead > 0:
    last_admissible = event_time - min_lead          # inclusive
    sufficient iff observation_end_s >= last_admissible
if min_lead == 0:
    timely confirm requires t < event_time
    sufficient iff at least one timestamp t < event_time
      (obs_end-only: observation_end_s < event_time)
```

If you keep `coverage_horizon_s` as a name, redefine it as `minimum_action_lead_time` (not `max(H, lead)`) and say so in the docstring. Prefer a new helper name.

**Tests** in `tests/test_spec_invariants.py`:

- Rewrite `test_insufficient_coverage_not_false_miss` comments and expected counts for the v1 bound. Flagship: 100/75/30/10 → insufficient, not miss. Keep overlay units as **official RUL annotation**, not as observed 600 Pa events.
- `test_coverage_boundary_at_last_admissible_timestamp` — obs_end == event - min_lead → sufficient; obs_end just below → insufficient (`min_lead > 0`).
- `test_min_lead_zero_sample_at_event_is_not_sufficient` — obs_end == event, min_lead 0, no earlier timestamp → not miss-via-coverage (insufficient); a row with obs_end strictly before event can be sufficient.
- Update `test_alert_outcome_timely_lead_time_and_coverage` if obs_end=80 / H=10 / lead=5 expectations change (still insufficient: last admissible = 95).
- **Do not** add `test_gap_in_admissible_window_is_insufficient_not_miss` or K-vs-too-few-samples coverage tests in this patch.

Do not change split protocol. Do not mix this with R5 export columns (next subtask).

## Dependencies

Subtasks 01 and 02 (orchestration order; alerts.py is new relative to those).

## Verification

```bash
.venv/bin/python -m pytest tests/test_spec_invariants.py -q -k "coverage or alert_outcome or insufficient"
.venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check src tests
```
