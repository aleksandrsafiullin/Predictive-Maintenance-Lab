# Subtask 9: Filter time scale warning and metadata

## Goal

Keep `time_to_seconds=60`; surface visible warning that Time/RUL unit compatibility is unconfirmed; preserve original units in exports.

## Context

P0-5: Do not change the coefficient. Document assumption; if later confirmed/changed → new data version + retrain.

## Acceptance Criteria

- [ ] `configs/filters.yaml` documents `time_to_seconds`, `original_time_unit`, uncertainty note.
- [ ] `data_report.json` and Data UI show prominent warning when scale unverified.
- [ ] Metrics/CSV exports include `time_unit_note` or use `_s` suffix without implying real minutes.
- [ ] No UI copy promises "X minutes before failure" as calibrated wall-clock.

## Implementation Notes

**Files:** `configs/filters.yaml`, `src/pdm/data/filters.py`, `src/pdm/data/prepare.py`, `src/pdm/app.py` (Data screen)

- Inspect repo docs / `implementation_report.md` for any confirmed unit statement; cite in `data_report`.
- `test_rul_and_time_conversion` may need update only for messaging, not factor change.

## Dependencies

Subtask 6 (data_report fields).

## Verification

```bash
.venv/bin/python -m pytest tests/test_spec_invariants.py::test_rul_and_time_conversion -q
.venv/bin/python -m ruff check src/pdm/data/filters.py src/pdm/app.py
```

Manual: Data screen shows warning for filters dataset.
