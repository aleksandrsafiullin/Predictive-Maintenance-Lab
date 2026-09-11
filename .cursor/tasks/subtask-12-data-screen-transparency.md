# Subtask 12: Data screen transparency (Section 6)

## Goal

Enhance Data screen with unit/window counts, exclusion reasons, events vs censoring, origin IDs, split hash, and data version.

## Context

Section 6 Data: keep units/split table; add diagnostic counts. Separate observed events from official-RUL-known test prefixes.

## Acceptance Criteria

- [ ] Table shows per-split: unit count, eligible windows, excluded windows (by reason: gap, post-event, insufficient length).
- [ ] Filters: observed 600 Pa events vs censored counts; regimes; total observation time.
- [ ] Display `origin_unit_id`, `split_protocol`, `split_hash`, `dataset_version` from `data_report.json` / fingerprint.
- [ ] Test-prefix units: distinguish observed sensor end vs evaluation-only official RUL.

## Implementation Notes

**Files:** `src/pdm/app.py` (`screen_data`), `src/pdm/data/prepare.py` (enrich `_data_report`)

- **Compute and cache** window/unit counts in `prepare.py` → `data_report.json` (eligible windows, exclusions by reason, events vs censoring, observation time). UI reads cache only — **do not call `build_windows` in Streamlit**.
- English UI copy.
- Use `st.dataframe` / metrics columns; avoid JSON blob as primary view.

## Dependencies

Subtasks 3, 6, 7.

**Soft (recommended):** 11a, 11b — Phase 2 regression tests green before Phase 3 Data UI ships.

## Verification

```bash
.venv/bin/python -m pytest tests/test_worker_and_app.py -q -k "data"  # extend AppTest if needed
.venv/bin/python -m ruff check src/pdm/app.py
```

Manual: Data screen with prepared synthetic or real data.
