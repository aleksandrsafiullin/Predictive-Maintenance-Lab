# Subtask 15: Evaluation metrics — RUL & baselines (Section 6)

## Goal

Implement dataset-specific RUL metrics, baseline endpoint fairness, and bearings near-event zones. No alert-policy metrics here — those are subtask 15b (after 17).

## Context

Section 6 Evaluation (RUL portion). Runs after immutable eval layout (14). Alert policy (`minimum_action_lead_time`) is defined in subtask 17 before alert metrics in 15b.

## Acceptance Criteria

- [ ] **Bearings:** primary `equal_weight_unit_mae`; pooled MAE/RMSE, mean overestimation, per test unit; baseline on same points.
- [ ] **Bearings near-event zones:** `configs/bearings.yaml` defines frozen `evaluation.near_event_zones_s: [3600, 1800, 600]` (seconds); not derived from test.
- [ ] `metrics.json` includes `equal_weight_unit_mae_by_zone` keyed by zone threshold.
- [ ] **Zone MAE row filter:** rows with finite `actual_rul_s` **and** finite `predicted_rul_s` **and** `0 < actual_rul_s <= zone_s`; endpoint rows excluded.
- [ ] Endpoint rows excluded from all RUL MAE aggregates (NaN `actual_rul_s`; per subtask 03 bearings + observed filter events).
- [ ] **Filters:** primary metric = last available point per author test prefix; all prefix points = secondary backtest; censored val: NLL all units + MAE on observed events with count.
- [ ] **Baselines:** `test_filter_endpoint_baseline_comparison` — NN and baseline on same endpoints with coverage fraction (test may land in 20).

## Implementation Notes

**Files:** `src/pdm/evaluate.py`, `src/pdm/baselines.py`, `configs/bearings.yaml`, `configs/filters.yaml`

- Fill stub `metrics.json` / `metrics_by_unit.csv` from subtask 14 with RUL columns; alert columns added in 15b.
- Depends on subtask 03 endpoint mask semantics for MAE denominators.

## Dependencies

Subtask 14.

## Verification

```bash
.venv/bin/python -m pytest tests/test_spec_invariants.py -q -k "endpoint_baseline"  # full test in subtask 20
.venv/bin/python -m ruff check src/pdm/evaluate.py src/pdm/baselines.py
```

RUL metrics smoke: run evaluate on tiny/smoke run; inspect `metrics.json` for `equal_weight_unit_mae_by_zone`.
