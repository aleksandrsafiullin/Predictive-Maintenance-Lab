# Subtask 6: Dataset fingerprinting and immutable versions

## Goal

`prepare` creates a new versioned processed dataset with checkable hashes; training runs snapshot fingerprints; silent mutation of old experiment inputs is impossible.

## Context

P0-3: `evaluate_run` loads current `load_processed()` instead of the run's snapshot. Need `dataset_version`, raw/processed hashes, `feature_pipeline_version`, **`split_hash`** (canonical name; alias existing `split_fingerprint()` in `train.py` — one identifier across fingerprint JSON, `checkpoints_compatible`, and tests).

## Acceptance Criteria

- [ ] `prepare_dataset` writes `dataset_version` (timestamp + short hash), `processed_fingerprint.json` with hashes of `features.parquet`, `units.parquet`, `split.json`, `feature_schema.json`.
- [ ] New prepare writes to versioned dir OR bumps version in manifest; old runs reference saved fingerprint.
- [ ] `train.py` saves `dataset_fingerprint.json` at run start (copy from processed + `split_hash`).
- [ ] `feature_pipeline_version` included in fingerprint.
- [ ] `origin_unit_id` column preserved in `units.parquet` where applicable (filters `author_data_no`).
- [ ] **`checkpoints_compatible` accepts canonical `split_hash`; also reads legacy `split_fingerprint` from old run dirs** (same value, no break on existing runs).

## Implementation Notes

**Files:** `src/pdm/data/prepare.py`, `src/pdm/train.py`, `src/pdm/io_util.py`, `src/pdm/paths.py`

- Use SHA256 of file bytes (truncate for display).
- **`split_hash`:** move `split_fingerprint()` from `train.py` to `splits.py` or `io_util.py`; export as `split_hash()` with `split_fingerprint` as deprecated alias. Write `split_hash` in new artifacts; `checkpoints_compatible` / loaders accept either key for backward compatibility.
- `data_report.json` gains `dataset_version`, `split_protocol`, hash fields.
- Do not commit `data/`; fingerprints are runtime artifacts.

## Dependencies

Subtask 1 (`feature_pipeline_version`).

## Verification

```bash
.venv/bin/python -m pytest tests/test_spec_invariants.py::test_evaluate_rejects_changed_dataset_or_split -q  # after subtask 8
.venv/bin/python -m ruff check src/pdm/data/prepare.py src/pdm/train.py
```
