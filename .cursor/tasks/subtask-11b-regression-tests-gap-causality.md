# Subtask 11b: P0 regression tests — gap & causality

## Goal

Add TZ §9 tests 6–8 (gap warmup, prefix invariance, labels do not affect predictor) to `tests/test_spec_invariants.py`.

## Context

Locks subtasks 4–5. Complements subtask 11a; run after or in parallel once 04–05 are done.

## Acceptance Criteria

- [ ] `test_replay_gap_warmup` — after gap, no prediction until H consecutive valid points; alert confirmation reset.
- [ ] `test_raw_prefix_invariance` — appending future raw rows does not change transform/predict at t.
- [ ] `test_labels_do_not_affect_predictor` — mutating GT / label columns through real pipeline does not change prediction.

## Implementation Notes

**Files:** `tests/test_spec_invariants.py`, `tests/conftest.py`

- Use `tiny_bearing_tables` gap at index 8 (`Bearing2_1`).
- Prefix invariance: append synthetic future rows to filter/bearing fixture; compare tensors at fixed t.

## Dependencies

Subtasks 4–5.

## Verification

```bash
.venv/bin/python -m pytest tests/test_spec_invariants.py -q -k "gap_warmup or prefix_invariance or labels_do_not"
.venv/bin/python -m ruff check tests/test_spec_invariants.py
```
