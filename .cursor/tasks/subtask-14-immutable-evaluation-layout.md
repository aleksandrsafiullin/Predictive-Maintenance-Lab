# Subtask 14: Immutable evaluation artifact layout (P1-1)

## Goal

Separate prediction evaluation from alert evaluation; store under `runs/<run_id>/evaluations/<eval_id>/` without overwriting prior reports.

## Context

P1-1: RUL predictions independent of H/K. Changing H/K re-runs alert engine only. Eval identity includes checkpoint, data version, split, evaluate mask, and (after subtask 17) alert policy fields.

## Acceptance Criteria

- [ ] Run root layout:
  ```
  runs/<run_id>/
    config.yaml, split.json, dataset_fingerprint.json, preprocessing.json,
    best.pt, training_history.csv
    evaluations/<eval_id>/
      evaluation_config.json, predictions.csv, alerts.csv,
      metrics.json, metrics_by_unit.csv
  ```
- [ ] `evaluate_run` creates new `eval_id` (UUID/timestamp); never overwrites existing eval dir.
- [ ] **Phase 1** writes `predictions.csv` plus stub/empty `metrics.json` and `metrics_by_unit.csv`; **RUL metrics populated in subtask 15**.
- [ ] **Phase 1** `evaluation_config.json`: prediction identity only — checkpoint hash, `dataset_fingerprint` fields, `split_hash`, evaluate mask, `metrics_version`. **No `minimum_action_lead_time` yet.**
- [ ] **Phase 2** (subtask 15b, after policy from 17): alert block appended — `H_trigger`, `minimum_action_lead_time`, K, `reset_factor`, policy hash.
- [ ] Legacy root-level `predictions.csv` / `test_metrics.json` deprecated or migrated read-only.
- [ ] CLI/UI list evaluations per run.

## Implementation Notes

**Files:** `src/pdm/evaluate.py`, `src/pdm/experiments.py`, `src/pdm/app.py`, `src/pdm/cli.py`, `src/pdm/worker.py`

- Phase 1: write `predictions.csv` (H/K-independent) + stub/empty `metrics.json` and `metrics_by_unit.csv` shells; **RUL metrics filled in subtask 15**.
- Phase 2: `run_alert_evaluation(predictions, policy from 17)` → `alerts.csv` + alert metrics (subtask 15b).
- `experiments.list_runs` / `run_dir` aware of eval subdirs.

## Dependencies

Subtasks 8, 3.

## Verification

```bash
.venv/bin/python -m ruff check src/pdm/evaluate.py src/pdm/experiments.py
```

Layout smoke: evaluate creates `evaluations/<eval_id>/` with `predictions.csv` and stub metrics files. RUL values appear after subtask 15; policy fields after 15b.
