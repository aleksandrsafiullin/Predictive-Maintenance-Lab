# Subtask 3: predict_with_trace, contributions, lazy artifacts, parity

## Goal

Expose `predict_with_trace` that returns per-frame reservoir states and per-frame contributions from the **same** state-update kernel used by ordinary `predict`. Persist traces lazily under `runs/<run_id>/traces/<unit_id>/` and lock contribution / causality / window-reset / reload tests.

## Context

Phase C. Phase B produced frozen ESN weights + linear readout and GRU-compatible `predicted_rul_s`. `Predictor.predict_from_history` currently builds one window tensor and calls `model.predicted_rul_s(x)` with no hidden-state dump. Replay (`src/pdm/replay.py`) already prefixes rows `≤ t`. Explorer (subtask 04) must not invent activity: it will play these artifacts.

There is **no** `predict_window_sequence()` in the repo today; add a single public API rather than a second parallel helper.

## Acceptance Criteria

- [ ] One function (e.g. `pdm.models.reservoir.forward_states` or `LeakyESN.forward`) implements the leaky update. `predicted_rul_s` / `forward` and `predict_with_trace` both call it. Tests grep or import-check that trace code does not reimplement the tanh map.
- [ ] `predict_with_trace(history_rows, ...)` returns at least: `predicted_rul_s` (display), `raw_prediction` (linear readout), `states` `[T, N]`, `inputs` `[T, F]`, `contributions` (intercept, per-input, per-neuron) `[T, ...]`, `frame_map` (frame index ↔ `timestamp_s` ↔ window offset), `status` matching `Predictor` collecting-history rules.
- [ ] Ordinary `predict` / `predict_from_history` does **not** allocate or write traces (lazy). Evaluate does not write `traces/` unless explicitly requested.
- [ ] Test 7: display and raw predictions from `predict` and `predict_with_trace` match within tight tolerance (`rtol=1e-6`, `atol=1e-6`) on the same prefix.
- [ ] Test 6: mutating frames **after** the current prefix does not change states, contributions, or prediction at t (same spirit as `test_future_measurements_do_not_change_prediction_at_t`).
- [ ] Test 8: `window_reset` — the first frame of window k does not equal the last frame of window k−1 when those windows are adjacent in replay; `x[0]` after reset is the leaky update from **zeros**, not from the previous window’s `x[T]`.
- [ ] Test 9: within a window, frame t’s pre-activation uses `x[t-1]` from that window. A constructed edge `j→i` drives node `i` from node `j`’s **previous** state, not from `u[t]` alone (hold `W_in=0` in the fixture).
- [ ] Test 10: for every frame (or at least the readout frame `T-1`), `intercept + sum(input_contrib) + sum(neuron_contrib) == raw_prediction` within `1e-5`. Works for scalar bearings raw and 2-vector Weibull raw (identity per output dim).
- [ ] Test 11: `raw_prediction` is the linear readout **before** Softplus / Weibull median / `time_scale_s`. Display RUL is documented and tested as a pure postprocess of `raw` (plus `time_scale_s`). Changing `time_scale_s` changes display, not the contribution identity on `raw`.
- [ ] Test 13: save trace artifact, load it, reconstruct display prediction equal to live `predict` (within tolerance). File layout `runs/<dataset_id>/<run_id>/traces/<unit_id>/`.
- [ ] Worker job kind (e.g. `trace`) is optional; if present, Stop sets `cancelled` not `completed`. Streamlit must not compute full-N traces inline on every widget click (may compute tiny test graphs in unit tests).
- [ ] GRU/LSTM: `predict_with_trace` either raises a clear “traces require a reservoir run” **or** no-ops without breaking `Predictor` for GRU. Do not attach fake neuron traces to GRU hidden units in this epic.
- [ ] Existing evaluate/replay isolation unchanged: traces are not written into `predictions.csv`; ground truth is still joined only by the evaluator.

## Key Files to Create/Modify

**Create**

- `src/pdm/visualization/__init__.py`
- `src/pdm/visualization/trace.py` — `predict_with_trace`, frame/window mapping, lazy flag
- `src/pdm/visualization/contributions.py` — decompose `raw = b + W_u u + W_x x`
- `src/pdm/visualization/export.py` — save/load npz+json atomically (`pdm.io_util`)
- `tests/test_reservoir.py` — tests 6–11, 13

**Modify**

- `src/pdm/models/reservoir.py` — public `forward_states` used by both predict paths; no duplicated loop in visualization
- `src/pdm/models/readout.py` — expose `W_x`, `W_u`, `b` for contribution math (same tensors as forward)
- `src/pdm/predict.py` — `Predictor.predict_from_history(..., with_trace=False)`; when True, delegate to `visualization.trace`
- `src/pdm/paths.py` — trace directory helper if not done in 01
- `src/pdm/worker.py` — optional `kind=trace`; honor stop.flag → `cancelled`
- `src/pdm/cli.py` — optional `--with-trace` on evaluate **off by default**
- `src/pdm/evaluate.py` — do not default to traces; if a flag is added, it must not change `predictions.csv` schema

## Implementation Notes

### Shared kernel

Bad: `forward` loops in `reservoir.py` and `trace.py` copies the equation. Good: `forward` calls `states = forward_states(...); return readout(states[:, -1], u[:, -1])` and trace returns the same `states`.

### Frame map

For history length `H` at replay step `t` (inclusive prefix):

- frames `0 .. H-1` map to the last `H` eligible measurements in the prefix
- `timestamp_s` copied from those rows
- `window_end_timestamp_s` = t’s timestamp
- `unit_id` string
- `prediction_index` / replay step integer

Equipment replay (subtask 04) will index this map; keep JSON-serializable types.

### Contributions

At frame t:

- `intercept` = `b` (broadcast)
- `input_contributions[j] = W_u[:, j] * u[t, j]` (keep output dim)
- `neuron_contributions[i] = W_x[:, i] * x[t, i]`

Sum over j and i plus intercept equals `raw[t]`. Do **not** include tanh internals in this identity. Do not use display RUL in the sum.

### Lazy + size

Default n_nodes in production is 1000; tests use tiny N. Write `float32`. Do not put traces in `predictions.csv`. Overwrite a unit’s trace only when requested for that unit.

### Causality

`Predictor` already slices `iloc[-history_length:]` of rows `≤ t`. Trace must use that same slice. Do not pad with future frames. Early windows: same `Collecting history` status as predict; no trace file.

### Artifact schema (test 13)

```
traces/<unit_id>/
  meta.json          # run_id, dataset_id, architecture, graph_hash, n_nodes, history_length, graph_mode, is_synthetic
  states.npz         # states, inputs
  contributions.npz  # intercept, input, neuron, raw
  frame_map.json
```

Loading uses stored readout weights from the run checkpoint, not live YAML.

## Dependencies

Subtask 02 (ESN + readout + train/eval).

## Verification

```bash
.venv/bin/python -m pytest tests/test_reservoir.py -q
.venv/bin/python -m pytest tests/test_spec_invariants.py -q
.venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check src tests
```

## Notes on constraints

- One state-update implementation (hard constraint 6).
- Contribution identity on **raw** (hard constraint 7), tolerance `1e-5`.
- No future frames; no official RUL in trace inputs.
- Synthetic traces still labeled synthetic in `meta.json`.
- Stop during a trace job → `cancelled` / `stopped`, never `completed`.
- Do not compute traces unless requested.
