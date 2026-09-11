# Subtask 15b: Evaluation metrics — alerts (Section 6 + §7)

## Goal

Implement multi-episode alert scoring with Δt weighting and `minimum_action_lead_time`-aware timely classification. Requires frozen policy from subtask 17.

## Context

Section 6 Alerts + Section 7. Must run **after** subtask 17 defines `H_trigger`, `minimum_action_lead_time`, and confirmed-alert timing with K-delay.

## Acceptance Criteria

- [ ] **Alerts:** `lead_time = event_time - confirmed_alert_time`; account K-1 step delay via `AlertEngine.confirmation_delay_steps`.
- [ ] **`timely` iff** `lead_time >= minimum_action_lead_time` and alert before event (not merely any warning in `[event - H_trigger, event)`).
- [ ] Score **all** episodes per unit; time-in-warning share via ΣΔt not row count.
- [ ] `too_early` / `late` / `miss` / `insufficient_coverage` defined consistently with subtask 17.
- [ ] Alert columns appended to `metrics.json` and `metrics_by_unit.csv` in existing eval dir.
- [ ] **`test_insufficient_coverage_not_false_miss` and `test_alert_multiple_episodes_and_time_weighting` implemented and passing here** (including timely iff `lead_time >= minimum_action_lead_time` with K-delay).

## Implementation Notes

**Files:** `src/pdm/evaluate.py`, `src/pdm/alerts.py`, `configs/bearings.yaml`, `configs/filters.yaml`

- `run_alert_evaluation(predictions.csv, policy)` → `alerts.csv` + alert metrics; re-runnable without regenerating predictions.
- Replace `_alert_summary` first-episode-only logic.
- `evaluation_config.json` alert/policy block written here (includes `minimum_action_lead_time`, `H_trigger`, K, reset policy, `metrics_version`).

## Dependencies

Subtasks 14, 17.

## Verification

```bash
.venv/bin/python -m pytest tests/test_spec_invariants.py -q -k "insufficient_coverage or multiple_episodes"
.venv/bin/python -m ruff check src/pdm/evaluate.py src/pdm/alerts.py
```
