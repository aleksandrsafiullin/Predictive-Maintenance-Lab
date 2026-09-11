# Subtask 2: Preprocessor imputation, scaling, and parity fixes

## Goal

Fix `Preprocessor.transform_frame` so NaN/Inf use saved train medians (not zero), scaling is applied exactly once, and offline tensors match `Predictor` within 1e-6.

## Context

Builds on subtask 1. Current `transform_frame` replaces non-finite values with `0.0` before scaling, diverging from fit-time median imputation. `UnitWindowDataset` also uses `nan_to_num(..., 0.0)` on pre-transformed arrays — must align.

## Acceptance Criteria

- [ ] `transform_frame`: for each feature, `np.where(~isfinite, fill_values[col], val)` then `(x - mean) / scale`.
- [ ] `UnitWindowDataset` loads features via the unified pipeline (subtask 1), not ad-hoc `nan_to_num` to zero.
- [ ] Attempt to call `transform_frame` on already-scaled data raises or no-ops per guard from subtask 1.
- [ ] Implementation complete; parity/imputation tests pass after subtask 11a (not a gate in this subtask).

## Implementation Notes

**Files:** `src/pdm/preprocessing.py`, `src/pdm/train.py` (`UnitWindowDataset`, `fit_preprocessor`), `src/pdm/predict.py`

- Persist `fill_values` per feature (already in dataclass); ensure fit and transform use same order as `feature_names`.
- After `fit_preprocessor`, store transformed features for window building OR transform inside `window_matrix` consistently.
- `time_scale_s` unchanged (train-only median duration).

## Dependencies

Subtask 1.

## Verification

```bash
.venv/bin/python -m ruff check src/pdm/preprocessing.py src/pdm/train.py
```

Manual: fit preprocessor on synthetic train rows with NaN; confirm transform uses median not zero. Full parity gate: subtask 11a.
