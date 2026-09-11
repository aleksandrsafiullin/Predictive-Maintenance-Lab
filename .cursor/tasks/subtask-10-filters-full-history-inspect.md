# Subtask 10: Inspect filters_full_history MAT readability

## Goal

Determine whether `Train_Data_Uncensored.mat` is readable; keep protocol disabled and document outcome. Do not invent MATLAB table fields.

## Context

TZ §3.3: optional future protocol. MVP stays `filters_censored` only.

## Acceptance Criteria

- [ ] `inspect_filters()` attempts to read MAT (if file present) and records **`filters_full_history_status`** enum in inspection notes: `missing` | `unreadable` | `readable_needs_protocol`.
- [ ] `inspection.json` and `data_report.json` propagate `filters_full_history_status` and human-readable reason; protocol remains disabled.
- [ ] No code path enables `filters_full_history` training in this sprint.
- [ ] Config flag remains off or absent.

## Implementation Notes

**Files:** `src/pdm/data/filters.py`, `src/pdm/data/archive.py`, `configs/filters.yaml`, `src/pdm/data/prepare.py`

- Use `scipy.io.loadmat` or existing archive helper; catch failures gracefully.
- Do not add columns from unreadable tables.
- If readable, note required export work as future subtask (out of scope).

## Dependencies

None (parallel).

## Verification

```bash
.venv/bin/python -m pdm inspect --dataset filters  # if data present
.venv/bin/python -m ruff check src/pdm/data/filters.py
```

No new pytest required unless inspect is unit-testable with mock path.
