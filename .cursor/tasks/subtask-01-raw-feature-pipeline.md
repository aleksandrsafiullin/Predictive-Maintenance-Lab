# Subtask 1: Unified raw→model feature pipeline entry point

## Goal

Introduce a single code path that converts **raw** measurement rows (as stored in `features.parquet` before model-specific encoding) into the model feature frame, so train and `Predictor` use identical categorical encoding.

## Context

P0-1: `fit_preprocessor` calls `filters_feature_frame` (dust one-hot) before fitting, but `Predictor.predict_from_history` passes raw rows to `Preprocessor.transform_frame`, which only fills missing columns with constants — dust categories collapse to zeros. Bearings path is simpler but should share the same staged API.

## Acceptance Criteria

- [ ] New function (e.g. `build_model_feature_frame(dataset_id, raw_df, prep_or_categories)`) in `src/pdm/features.py` or `src/pdm/preprocessing.py` performs: categorical encoding → log1p → column selection in saved order.
- [ ] `fit_preprocessor` and `Predictor` both call this function; no duplicate logic.
- [ ] Unknown dust category maps to documented behavior (e.g. all-zero one-hot columns, or explicit `dust__<unknown>` bucket if in train map).
- [ ] `Preprocessor` records `feature_pipeline_version` (e.g. `"v2_raw_first"`) in `to_dict()`.

## Implementation Notes

**Files:** `src/pdm/features.py`, `src/pdm/preprocessing.py`, `src/pdm/predict.py`, `src/pdm/train.py`

- Keep `bearings_feature_frame` / `filters_feature_frame` but expose `raw_to_feature_frame(dataset_id, df, categorical_maps)` used by both fit and inference.
- `categorical_maps` from fitted preprocessor: `{"dust": ["cat_a", "cat_b", ...]}`.
- Double-transform guard: require explicit schema flag on input (e.g. `raw_features=True` on DataFrame attrs or call parameter). If `raw_features=False` or flag absent on ambiguous input, raise clear `ValueError` — do not use near-zero mean heuristics.
- Do not add forbidden columns (`FORBIDDEN_FEATURE_NAMES` in `windows.py`).

## Dependencies

None (first subtask).

## Verification

```bash
.venv/bin/python -m ruff check src/pdm/features.py src/pdm/preprocessing.py src/pdm/predict.py
```

Manual: two-row synthetic filter frame with different `dust` values → different encoded columns before scaling. Parity/dust pytest (`test_offline_online_feature_parity`, `test_dust_encoding_from_raw`) pass after subtask 11a — **not a gate in this subtask**.
