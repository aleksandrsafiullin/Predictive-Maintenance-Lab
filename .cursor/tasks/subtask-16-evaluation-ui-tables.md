# Subtask 16: Evaluation UI — summary and per-unit tables

## Goal

Replace JSON-primary evaluation display with human-readable summary metrics and per-unit table in Test & Replay / Evaluate flow.

## Context

Section 6 Evaluation: summary + per-unit table, not raw JSON. Link to immutable `evaluations/<eval_id>/`.

## Acceptance Criteria

- [ ] Evaluate action shows eval_id, primary metric headline, dataset-specific notes.
- [ ] `st.dataframe` for `metrics_by_unit.csv` (unit_id, MAE, NLL, alert class, lead_time, coverage).
- [ ] Bearings: show 3 test units explicitly; filters: prefix-end RUL vs official RUL distinction.
- [ ] Download buttons for predictions/alerts CSV from eval dir.
- [ ] JSON available in expander only (secondary).

## Implementation Notes

**Files:** `src/pdm/app.py` (`screen_replay` evaluate section), `src/pdm/experiments.py`

- Load latest eval or let user pick eval_id from dropdown.
- Use `compare_baseline` output for coverage callout.
- English labels.

## Dependencies

Subtasks 14, 15, 15b.

## Verification

```bash
.venv/bin/python -m pytest tests/test_worker_and_app.py -q -k "evaluat"
.venv/bin/python -m ruff check src/pdm/app.py
```

AppTest: evaluate button produces table element or metric keys.
