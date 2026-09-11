# Subtask 13: Train screen Smoke/Full mode and diagnostics

## Goal

Expose training mode clearly; fix `max_windows_per_unit` stuck at 48 when smoke off; show selection metric, best epoch, windows/unit used.

## Context

Section 6 Training: smoke default in Advanced hides full-training path. First full run needs `smoke=False`, `max_windows_per_unit=None` (UI 0), reasonable `max_epochs` + val early stopping.

## Acceptance Criteria

- [ ] UI shows `Smoke` vs `Full` badge on run and in Train form.
- [ ] `max_windows_per_unit`: 0 or empty → `None` (unlimited); smoke preset uses cap (e.g. 24–32).
- [ ] Turning smoke off clears smoke window cap unless user sets one.
- [ ] Display selection metric name + unit (val MAE s / val NLL).
- [ ] Training history plot marks best epoch; shows real windows per unit used.
- [ ] Train/val diagnostic metrics on fixed masks with unit weighting (read from `training_history.csv` / status).

## Implementation Notes

**Files:** `src/pdm/app.py` (`screen_train`), `src/pdm/train.py`, `src/pdm/worker.py`, `configs/*.yaml`

- Worker job payload: `smoke`, `max_windows_per_unit`, `max_epochs`.
- Do not require full train to pass CI.
- Log `n_train_windows`, `n_val_windows` in run status.

## Dependencies

Subtask 3 (correct window counts).

## Verification

```bash
.venv/bin/python -m pytest tests/test_worker_and_app.py -q -k "train"
.venv/bin/python -m ruff check src/pdm/app.py src/pdm/train.py
```

Smoke train still works: `.venv/bin/python -m pdm train --dataset bearings --smoke`
