# Subtask 4: Gap and input-quality rules at inference

## Goal

Apply the same gap exclusion and input validation at inference as `build_windows` uses for training; reset alert confirmation after gaps.

## Context

P0-4: `build_windows` skips windows crossing `gap_before`; `Predictor` takes last H rows without checking. After a gap, return `Collecting history` until H consecutive valid points exist.

## Acceptance Criteria

- [ ] Shared helper (e.g. `valid_history_window(prefix, history_length)`) in `src/pdm/windows.py` returns `(ok, reason)` checking: length, gap_before in window interior, monotonic unique `timestamp_s`, finite required raw columns.
- [ ] `Predictor.predict_from_history` uses helper; returns `Collecting history` when invalid.
- [ ] `replay.py` passes gap reset signal to `AlertEngine` (reset consecutive counters when window invalid after gap).
- [ ] Implementation complete; `test_replay_gap_warmup` passes after subtask 11b (not a gate in this subtask).

## Implementation Notes

**Files:** `src/pdm/windows.py`, `src/pdm/predict.py`, `src/pdm/replay.py`, `src/pdm/alerts.py`

- Gap check: same as training — `gap[start+1:end+1].any()` on sorted prefix.
- Monotonicity: `timestamp_s` strictly increasing (or allow equal with warning).
- `AlertEngine.update(collecting=True)` already clears counters; ensure replay sets `collecting` when gap warmup active.
- Do not pad with future rows.

## Dependencies

Subtask 1 (raw column names for finiteness check).

## Verification

```bash
.venv/bin/python -m ruff check src/pdm/windows.py src/pdm/predict.py src/pdm/replay.py
```

Manual: gap in prefix returns `Collecting history`. Full gate: subtask 11b (`test_replay_gap_warmup`).
