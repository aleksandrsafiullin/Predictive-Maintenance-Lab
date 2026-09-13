# Subtask 3: predict_with_trace, contributions, lazy artifacts, parity

## Goal

Expose `predict_with_trace` that returns per-frame reservoir states and per-frame contributions from the **same** state-update kernel used by ordinary `predict` (same function object). Persist traces lazily under `runs/<run_id>/traces/<unit_id>/` and lock contribution / causality / window-reset / reload tests. Contribution identity uses **raw** linear output **before** Softplus / Weibull median / `time_scale_s`.

## Context

Phase C. Phase B3 produced frozen ESN weights + linear readout and GRU-compatible `predicted_rul_s`. `Predictor.predict_from_history` currently builds one window tensor and calls `model.predicted_rul_s(x)` with no hidden-state dump. Replay (`src/pdm/replay.py`) already prefixes rows `≤ t`. Explorer (subtask 04) must not invent activity: it will play these artifacts.

There is **no** `predict_window_sequence()` in the repo today; add a single public API rather than a second parallel helper.

Ridge (02c) already collects last-step states via `forward_states`. This subtask must **not** introduce a second tanh loop for traces.

## Acceptance Criteria

- [ ] One function (e.g. `pdm.models.reservoir.forward_states` or `LeakyESN.forward`) implements the leaky update. `predicted_rul_s` / `forward` and `predict_with_trace` both call it. Ridge collection (02c) already uses it.

- [ ] Test **`test_predict_and_trace_share_update_function`**: assert the trace path and the predict path bind the **same function object** (`is`); grep or AST/source check that `visualization/trace.py` does **not** contain a second `tanh` time loop.

- [ ] `predict_with_trace(history_rows, ...)` returns at least: `predicted_rul_s` (display), `raw_prediction` (linear readout **before** Softplus / Weibull median / `time_scale_s`), `states` `[T, N]`, `inputs` `[T, F]`, `contributions` (intercept, per-input, per-neuron) `[T, ...]`, `frame_map` (frame index ↔ `timestamp_s` ↔ window offset), `status` matching `Predictor` collecting-history rules.

- [ ] Ordinary `predict` / `predict_from_history` does **not** allocate or write traces (lazy). Evaluate does not write `traces/` unless explicitly requested.

- [ ] Test 7 `test_predict_trace_parity`: display and raw predictions from `predict` and `predict_with_trace` match within tight tolerance (`rtol=1e-6`, `atol=1e-6`) on the same prefix.

- [ ] Test 6 `test_no_future_frames`: mutating frames **after** the current prefix does not change states, contributions, or prediction at t (same spirit as `test_future_measurements_do_not_change_prediction_at_t`).

- [ ] Test 8 `test_window_reset`: the first frame of window k does not equal the last frame of window k−1 when those windows are adjacent in replay; `x[0]` after reset is the leaky update from **zeros**, not from the previous window’s `x[T]`.

- [ ] Test 9 `test_edge_drive_previous_state`: within a window, frame t’s pre-activation uses `x[t-1]` from that window. A constructed edge `j→i` drives node `i` from node `j`’s **previous** state, not from `u[t]` alone (hold `W_in=0` in the fixture).

- [ ] Test 10 `test_contribution_sum`: for every frame (or at least the readout frame `T-1`), `intercept + sum(input_contrib) + sum(neuron_contrib) == raw_prediction` within `1e-5`. Works for scalar bearings raw and 2-vector Weibull raw (identity per output dim). **`raw_prediction` is pre-Softplus / pre-median / pre-`time_scale_s`.**

- [ ] Test 11 `test_raw_vs_display_postprocess`: `raw_prediction` is the linear readout **before** Softplus / Weibull median / `time_scale_s`. Display RUL is a pure postprocess of `raw` (plus `time_scale_s`). Changing `time_scale_s` changes display, not the contribution identity on `raw`.

- [ ] Test 13 `test_trace_artifact_reload`: save trace artifact, load it, reconstruct display prediction equal to live `predict` (within tolerance). File layout `runs/<dataset_id>/<run_id>/traces/<unit_id>/`.

- [ ] Worker job kind (e.g. `trace`) is optional; if present, Stop sets **`cancelled`** not `completed` and not `stopped`. Streamlit must not compute full-N traces inline on every widget click (may compute tiny test graphs in unit tests).

- [ ] GRU/LSTM: `predict_with_trace` either raises a clear “traces require a reservoir run” **or** no-ops without breaking `Predictor` for GRU. Do not attach fake neuron traces to GRU hidden units in this epic.

- [ ] Existing evaluate/replay isolation unchanged: traces are not written into `predictions.csv`; ground truth is still joined only by the evaluator.

## Key Files to Create/Modify

**Create**

- `src/pdm/visualization/__init__.py`
- `src/pdm/visualization/trace.py` — `predict_with_trace`, frame/window mapping, lazy flag; imports `forward_states` (does not reimplement tanh)
- `src/pdm/visualization/contributions.py` — decompose `raw = b + W_u u + W_x x`
- `src/pdm/visualization/export.py` — save/load npz+json atomically (`pdm.io_util`)
- `tests/test_reservoir.py` — tests 6–11, 13, `test_predict_and_trace_share_update_function`

**Modify**

- `src/pdm/models/reservoir.py` — public `forward_states` used by predict, trace, and ridge (02c)
- `src/pdm/models/readout.py` — expose `W_x`, `W_u`, `b` for contribution math (same tensors as forward)
- `src/pdm/predict.py` — `Predictor.predict_from_history(..., with_trace=False)`; when True, delegate to `visualization.trace`
- `src/pdm/paths.py` — trace directory helper if not done in 01
- `src/pdm/worker.py` — optional `kind=trace`; honor `stop.flag` → `cancelled`
- `src/pdm/cli.py` — optional `--with-trace` on evaluate **off by default**
- `src/pdm/evaluate.py` — do not default to traces; if a flag is added, it must not change `predictions.csv` schema

## Implementation Notes

### Shared kernel (W2)

Bad: `forward` loops in `reservoir.py` and `trace.py` copies the equation. Good: `forward` calls `states = forward_states(...); return readout(states[:, -1], u[:, -1])` and trace returns the same `states`. `test_predict_and_trace_share_update_function` checks `trace_fn is predict_fn` (or both are `LeakyESN.forward` / `forward_states`).

Ridge feature collection (already 02c) must keep calling this same kernel — do not add `forward_states_ridge`.

### Frame map

For history length `H` at replay step `t` (inclusive prefix):

- frames `0 .. H-1` map to the last `H` eligible measurements in the prefix
- `timestamp_s` copied from those rows
- `window_end_timestamp_s` = t’s timestamp
- `unit_id` string
- `prediction_index` / replay step integer

Equipment replay (subtask 04) will index this map; keep JSON-serializable types.

### Contributions and ridge target space (C3.5, W4)

**Document in `contributions.py` (one paragraph / module docstring):**

Ridge (bearings) is solved in **normalized RUL** space (`y = target_rul_s / time_scale_s`), matching Smooth L1. That is a **training** target space. The identity test is **not** in seconds and **not** after Softplus.

At frame t, using the **pre-display** linear output:

- `intercept` = `b` (broadcast)
- `input_contributions[j] = W_u[:, j] * u[t, j]` (keep output dim)
- `neuron_contributions[i] = W_x[:, i] * x[t, i]`

`sum + intercept == raw[t]` within `1e-5`. Do **not** include tanh internals. Do **not** use `softplus(raw)`, Weibull median, or `raw * time_scale_s` in the sum. Display RUL is test 11 postprocess only.

### Lazy + size

Default n_nodes in production is 1000 (`real_connectome`); synthetic fixture ≥50, tests 8–16. Write `float32`. Do not put traces in `predictions.csv`. Overwrite a unit’s trace only when requested for that unit.

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

Loading uses stored readout weights from the run checkpoint / `weights.npz`, not live YAML, and **never** rebuilds reservoir weights from seed (same rule as 02c `load_trained_model`).

## Dependencies

Subtask 02c (ESN + readout + train/eval + shared `forward_states`).

## Verification Commands

```bash
.venv/bin/python -m pytest tests/test_reservoir.py -q
.venv/bin/python -m pytest tests/test_spec_invariants.py -q
.venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check src tests
```

## Notes/Constraints

- One state-update implementation — same function object (hard constraint 6, W2).
- Contribution identity on **raw** before Softplus / median / `time_scale_s` (C3.5, W4), tolerance `1e-5`.
- Ridge trained in normalized RUL; do not assert identity in that space.
- No future frames; no official RUL in trace inputs.
- Synthetic traces still labeled synthetic in `meta.json`.
- Stop during a trace job → **`cancelled`**, never `completed` or `stopped`.
- Do not compute traces unless requested.
