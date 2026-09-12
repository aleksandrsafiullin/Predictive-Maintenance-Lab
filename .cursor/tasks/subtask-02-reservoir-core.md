# Subtask 2: Reservoir core, fly + random, readout, train/eval

## Goal

Implement the leaky Echo State Network (fixed `W_in` / `W_res` / `b_res`, trainable linear readout only), the two architecture strings `fly_connectome_reservoir` and `random_reservoir`, and wire them through `run_training`, `load_trained_model`, `evaluate_run`, and `Predictor` for **both** bearings and filters without changing GRU/LSTM behavior.

## Context

Phase B. Phase A shipped graph orientation, synthetic fixture, YAML defaults, and `build_model()` passthrough. `src/pdm/models.py` is still a module: convert it to a package so reservoir files can live at the requested paths. `run_training()` currently always builds `PDMNet` and `AdamW(model.parameters())`. `_run_epoch` already branches bearings Smooth L1 vs filters `weibull_nll` — reservoir `forward()` must match those return types.

## Acceptance Criteria

- [ ] `src/pdm/models.py` is replaced by package `src/pdm/models/` with `__init__.py` re-exporting `PDMNet`, `RecurrentEncoder`, `RULHead`, `WeibullHead`. Existing imports keep working. `RecurrentEncoder` still rejects anything except `gru`/`lstm`.
- [ ] Shared update (CPU-correct, vectorized over batch, loop over time):

  `x[t] = (1 - alpha) * x[t-1] + alpha * tanh(W_res @ x[t-1] + W_in @ u[t] + b_res)`

  Default `state_mode=window_reset` zeros `x` at the start of each window. No second copy of this formula in train vs predict (predict_with_trace in subtask 03 will call this same function).
- [ ] `W_res[i, j]` = scaled weight of edge **j→i**. Test 1: one-edge graph `src→dst` ⇒ `W_res[dst, src] != 0` and `W_res[src, dst] == 0`. Code review must not add `.T` without a failing test that demanded it.
- [ ] Weight policy: `A[i,j] = log1p(synapse_count j→i)`, then scale so spectral radius equals `spectral_radius` (default 0.9). Empty/zero graphs fail loudly.
- [ ] `FlyConnectomeReservoir`: loads the run’s connectome artifact (or builds it once at train start), freezes `W_in`, `W_res`, `b_res` (`requires_grad=False`), trainable readout only.
- [ ] `RandomReservoir`: degree-preserving **directed** rewiring of the **same** node set and edge count (matched N/E control). Same update, readout, `state_mode`, leak, spectral radius, input scale, seed protocol. `graph_mode` is `random_rewire` (never `real_connectome`, never unlabeled synthetic).
- [ ] Readout raw: `raw = W_x @ x[T] + W_u @ u[T] + b` (direct input term required so contribution tests in 03 have three parts). Display: bearings non-negative RUL in seconds; filters Weibull λ, k via `softplus` then `predicted_rul_s` = `weibull_median_rul`.
- [ ] Bearings readout solver: `ridge` (default, `ridge_alpha=0.001`) **or** `gradient` linear layer + existing Smooth L1. Filters: **gradient + `weibull_nll` only**. Selecting ridge on filters raises. Censored rows stay `event=0`; never target `RUL=0` at planned end.
- [ ] Frozen reservoir weights are **not** updated by AdamW (assert `requires_grad` and that `W_res` bytes are identical after a gradient step). Ridge path does not run backprop through `W_res`.
- [ ] `build_model()` dispatches all four architectures. `run_training()` / `load_trained_model()` use it. GRU/LSTM constructor arguments and checkpoint `state_dict` layout stay the same.
- [ ] `compatibility_dict` / `checkpoints_compatible` for reservoirs also compare `n_nodes`, `graph_mode`, `graph_hash`, `state_mode`, `leak`, `spectral_radius`, `input_scale`, `seed`, `readout`. Wrong `n_nodes` vs artifact size raises (test 3). Bearings checkpoint cannot load as filters (existing `dataset_id`/`head` plus new graph keys).
- [ ] Smoke uses `smoke_n_nodes` (≈300) when `smoke=True`; full default `n_nodes=1000`. Tests use tiny graphs (e.g. 8–16 nodes), not 1000.
- [ ] Train writes `runs/<dataset_id>/<run_id>/connectome/` (`provenance.json`, `graph.json`, `weights.npz`, `layout.json`). `experiment_snapshot.json` stores the resolved reservoir block.
- [ ] `Predictor.predict_from_history` works for reservoir models via `predicted_rul_s` without seeing future rows or official RUL.
- [ ] Tests 1, 2, 3, 4, 5, 12 in `tests/test_reservoir.py` pass. Existing invariant + AppTest suites pass. Ruff clean.

## Key Files to Create/Modify

**Create**

- `src/pdm/models/__init__.py` — re-exports + `build_model`
- `src/pdm/models/recurrent.py` — current `models.py` contents moved verbatim
- `src/pdm/models/reservoir.py` — `LeakyESN` / `reservoir_update(...)` shared kernel; `window_reset`
- `src/pdm/models/fly_reservoir.py` — `FlyConnectomeReservoir`
- `src/pdm/models/random_reservoir.py` — `RandomReservoir` (rewire in `src/pdm/connectome/` or here; keep rewiring deterministic)
- `src/pdm/models/readout.py` — ridge closed form (train units/windows only) + `nn.Linear` gradient readout
- `tests/test_reservoir.py` — tests 1–5, 12 (extend Phase A file)

**Modify**

- **Delete** leftover `src/pdm/models.py` after the package exists
- `src/pdm/connectome/weights.py` — spectral-radius scaling (orientation already fixed in 01)
- `src/pdm/connectome/sampling.py` — if rewiring belongs with sampling, keep fly vs random sharing one edge-count
- `src/pdm/train.py` — `build_model`; optimizer over **trainable** params only; ridge branch for bearings; still `_run_epoch` for gradient; save frozen matrices in `connectome/weights.npz` **and** checkpoint; extend `compatibility_dict`, `_save_ckpt` `meta`, `load_trained_model`
- `src/pdm/evaluate.py` — no special-case leakage; `load_trained_model` must succeed for reservoir runs; `IncompatibleDataError` if graph/dataset mismatch
- `src/pdm/predict.py` — type hint `PDMNet` → protocol/`nn.Module` with `predicted_rul_s`; do not add a second update loop
- `src/pdm/config.py` — readout solver default by dataset
- `src/pdm/worker.py` — pass architecture through (already `job.get("architecture")`); stop → not `completed`
- `src/pdm/cli.py` — train already has `--arch`; no new heavy flags required
- `pyproject.toml` — only if package-data is needed for the synthetic fixture file

**Do not modify**

- Split protocol, `time_to_seconds`, `filters_full_history`
- `RecurrentEncoder` allowed names
- Alert H/K policy writers from the previous sprint

## Implementation Notes

### Package conversion (do this first, verify GRU tests, then add reservoirs)

1. Create `src/pdm/models/recurrent.py` with the current classes.
2. `__init__.py`: `from pdm.models.recurrent import PDMNet, RecurrentEncoder, RULHead, WeibullHead`.
3. Remove `src/pdm/models.py`.
4. Run `tests/test_spec_invariants.py::test_gru_lstm_both_heads_change_weights` and checkpoint reload tests **before** writing ESN code.

### Orientation

`W_res @ x` is standard `sum_j W_res[i,j] * x[j]`, so column `j` is the **source**. Edge `j→i` must sit at `[i,j]`. Hand test (test 2): 2 nodes, `alpha=1`, `W_in=0`, `b=0`, `x[0]=[1,0]`, `W_res[1,0]=c`, others 0 → `x[1,1] = tanh(c)` (or the leaky combination if `alpha≠1`). Compute expected values in the test with numpy, not by calling the production function.

### Ridge vs gradient

- **Ridge (bearings only):** collect last-step `[x; u]` on **train windows only** (same `UnitWindowDataset` / train split). Solve `(ZᵀZ + λI) W = Zᵀ y` for normalized RUL. Do **not** use val/test windows to fit. Then freeze readout for eval or keep it as buffers.
- **Gradient:** `nn.Linear` on concatenated `[x; u]`; `requires_grad` only there; reuse `_run_epoch` and existing losses.
- Filters: if `event==0`, loss is survival term only (`weibull_nll`). Add a test that a censored batch with `event=0` does not equal a batch that sets `target=0` as observed failure (test 12).

### Isolation (tests 3–5)

- Building a model with `n_nodes=16` against a 8-node artifact raises.
- `checkpoints_compatible` false across `dataset_id` and across `fly_connectome_reservoir` vs `random_reservoir` vs `gru`.
- Fit preprocessor on train; mutate test-unit features; scaler unchanged; `W_in` unchanged (seed-only).
- Same seed twice → identical `W_res` (after scaling) and identical predictions on a frozen readout.

### Random rewiring

Directed Maslov–Sneppen / degree-preserving swaps with a seeded RNG. Preserve in- and out-degree sequences as far as the algorithm guarantees; document if self-loops/parallels are forbidden. Do not copy `W_res` from the fly graph after rewiring — rebuild `log1p` + spectral scale on the rewired edges. Label provenance `graph_mode=random_rewire` and never the synthetic-biology disclaimer unless the **source** graph was synthetic (then **both** disclaimers apply: synthetic **and** rewired).

### `hidden_size` in checkpoints

For reservoirs set `meta["hidden_size"] = n_nodes` so existing meta keys remain, **or** add `n_nodes` and teach `load_trained_model` to prefer it. GRU checkpoints must still load with `hidden_size` from training. Do not require `recurrent_layers` to mean stacked ESNs.

### Leakage

`unit_id` / RUL / official overlay still not inputs. `W_in` shape `[n_nodes, n_features]` uses preprocessor `feature_names` order. Features already scaled by train-only scaler before `u[t]`.

## Dependencies

Subtask 01 (graph, provenance, YAML, architecture names, `build_model` hook).

## Verification

```bash
.venv/bin/python -m pytest tests/test_spec_invariants.py -q
.venv/bin/python -m pytest tests/test_reservoir.py -q
.venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check src tests
```

Optional (not a quality gate): tiny synthetic-graph smoke on CPU:

```bash
.venv/bin/python -m pdm train --dataset bearings --arch fly_connectome_reservoir --smoke --epochs 1 --max-windows-per-unit 8
```

Only if processed bearings exist locally; skip in CI if data missing.

## Notes on constraints

- GRU/LSTM `state_dict` keys and `PDMNet.forward` signatures must stay stable.
- Filters: **no ridge**; **no** censored-as-failure.
- Do not transpose `W_res` to match literature that uses the opposite convention; this repo’s convention is `W_res[i,j]=j→i`.
- `synthetic_fixture` runs are allowed for tests; they must still carry the synthetic label in `compat` and provenance.
- Cancel/stop during train still must not emit `completed`.
