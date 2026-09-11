# Subtask 7: Split integrity and origin_unit_id checks

## Goal

Enforce disjoint splits, no duplicate/unassigned units, and no `origin_unit_id` overlap across train/val/test.

## Context

P0-3 acceptance: `train ∩ val`, `train ∩ test`, `val ∩ test` empty; check physical history IDs not just filenames. Filters reuse `Data_No` across train/test files.

## Acceptance Criteria

- [ ] `assert_disjoint_splits` extended or companion `assert_split_coverage(units, split)` fails on duplicates, unassigned units (when protocol expects full assignment), and `origin_unit_id` intersection.
- [ ] `filters_split` / `bearings_split` populate `origin_unit_id` on units table if missing.
- [ ] `prepare_dataset` calls validation before writing `split.json`.
- [ ] `split_hash` in fingerprint includes protocol name and unit lists (canonical name per subtask 06).

## Implementation Notes

**Files:** `src/pdm/splits.py`, `src/pdm/data/filters.py`, `src/pdm/data/bearings.py`, `src/pdm/data/prepare.py`

- Filters: `origin_unit_id` = `author_data_no` or stable physical key from CSV metadata.
- Bearings: `origin_unit_id` = `unit_id` (one file per unit).
- Warn on `split["unassigned"]` non-empty for bearings.
- Do not change 9/3/3 or 40/10/50 protocols.

## Dependencies

Subtask 6 (fingerprint uses split).

## Verification

```bash
.venv/bin/python -m pytest tests/test_spec_invariants.py::test_split_units_disjoint -q
.venv/bin/python -m ruff check src/pdm/splits.py
```

Add new test for `origin_unit_id` intersection if not covered.
