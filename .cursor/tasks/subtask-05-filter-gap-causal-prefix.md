# Subtask 5: Causal gap threshold for filter streaming

## Goal

Compute filter `gap_before` using only the available measurement prefix (or documented sampling interval), not median Δt of the full trajectory.

## Context

P0-4 causality: `extract_filters_tables` sets `gap_before` with `med_dt = median(dt[1:])` over the **entire** unit file. For strict streaming/replay, gap detection at time t must not use future timestamps.

## Acceptance Criteria

- [ ] Document gap rule in `configs/filters.yaml` (e.g. `gap_multiplier: 3.0`, `sampling_interval_s` fallback).
- [ ] Replay/predict path can recompute `gap_before` on prefix OR use precomputed flags only when causally valid (prefix-only median Δt).
- [ ] Implementation complete; `test_raw_prefix_invariance` passes after subtask 11b (not a gate in this subtask).

## Implementation Notes

**Files:** `src/pdm/data/filters.py`, `configs/filters.yaml`, `src/pdm/replay.py`, `src/pdm/predict.py`

- Option A: store per-row `delta_t_s` in parquet; at inference, `gap[i] = delta_t[i] > k * median(delta_t[1:i])`.
- Option B: fixed `sampling_interval_s` from config when PDF/export documents Hz (do not invent MAT fields).
- Prepared parquet can keep full-file gaps for offline windows; inference uses prefix recomputation.
- Do not change `time_to_seconds=60`.

## Dependencies

Subtask 4.

## Verification

```bash
.venv/bin/python -m ruff check src/pdm/data/filters.py src/pdm/replay.py src/pdm/predict.py
```

Manual: append future rows to fixture; prefix transform unchanged at fixed t. Full gate: subtask 11b (`test_raw_prefix_invariance`).
