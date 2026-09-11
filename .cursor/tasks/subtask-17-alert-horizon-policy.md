# Subtask 17: Alert horizon policy split (Section 7)

## Goal

Separate `H_trigger` (when warning activates) from `minimum_action_lead_time` (required user reaction time); tune on validation, freeze on test. **Must complete before subtask 15b (alert metrics).**

## Context

Current `timely` = any alert in [event-H, event) including 1s before event. Real lead time must account for K-1 confirmation delay. Policy definition precedes alert metric computation.

## Acceptance Criteria

- [ ] Config/UI fields: `H_trigger` (alias existing `warning_horizon_s`), `minimum_action_lead_time`, optional max useful horizon.
- [ ] `AlertEngine` unchanged trigger logic (RUL ≤ H for K steps); evaluation uses confirmed alert time = timestamp when episode opens (after K-th qualifying step).
- [ ] **`timely` iff** `lead_time >= minimum_action_lead_time` (with K-delay accounted) **and** alert before event; `too_early` / `late` / `miss` / `insufficient_coverage` defined.
- [ ] Timely semantics covered in `test_alert_multiple_episodes_and_time_weighting` (**subtask 15b**).
- [ ] Validation policy picker saves frozen policy to **`runs/<run_id>/alert_policy.json`** (run defaults); each eval copies policy into `evaluations/<eval_id>/evaluation_config.json`.
- [ ] Policy schema consumed by subtask 15b when writing alert block to `evaluation_config.json`.

## Implementation Notes

**Files:** `src/pdm/alerts.py`, `src/pdm/evaluate.py`, `src/pdm/app.py`, `configs/bearings.yaml`, `configs/filters.yaml`

- `confirmation_delay_steps` property already on `AlertEngine`.
- Default `minimum_action_lead_time` e.g. fraction of `H_trigger` or fixed seconds from config.
- **`runs/<run_id>/alert_policy.json`:** canonical frozen defaults (tuned on validation); copied into each `evaluation_config.json` at alert-eval time.
- Do not retroactively change saved predictions when policy changes.

## Dependencies

Subtask 14.

## Verification

```bash
.venv/bin/python -m pytest tests/test_spec_invariants.py::test_alert_confirmation_dedup_reset_h_independent_of_prediction -q
.venv/bin/python -m ruff check src/pdm/alerts.py
```

Extend timely semantics in subtask 15b (test owner).
