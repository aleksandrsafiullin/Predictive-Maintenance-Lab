# Subtask 2b: Leaky ESN, weights, fly + random graphs

## Goal

Implement the leaky Echo State Network (fixed `W_in` / `W_res` / `b_res`, linear readout module) and the two architecture classes `FlyConnectomeReservoir` and `RandomReservoir`, with `forward()` return types that match existing `_run_epoch` branches. Do **not** wire `run_training` / ridge solving / `compatibility_dict` yet (02c).

## Context

Phase B part 2. 02a made `src/pdm/models/` a package with GRU/LSTM in `recurrent.py`. 01 shipped graph orientation, a ≥50-node synthetic fixture, clamp-not-raise for `synthetic_fixture`, and `build_model` that still errors on reservoir train. This subtask adds math + graph construction that unit tests can call directly.

`src/pdm/train.py` `_run_epoch` (~884–890) does:

- bearings: `pred_norm = model(x)` then Smooth L1 vs `target / time_scale_s`
- filters: `lam, k = model(x)` then `weibull_nll`

Reservoir `forward()` **must** match those shapes **before** 02c reuses `_run_epoch` for the gradient path.

## Acceptance Criteria

- [ ] Shared update (CPU-correct, vectorized over batch, loop over time):

  `x[t] = (1 - alpha) * x[t-1] + alpha * tanh(W_res @ x[t-1] + W_in @ u[t] + b_res)`

  Default `state_mode=window_reset` zeros `x` at the start of each window. Public kernel name: `forward_states` and/or `LeakyESN.forward`. **One** implementation — no second tanh loop in a sibling helper.

- [ ] `W_res[i, j]` = scaled weight of edge **j→i**. Test `test_graph_orientation`: one-edge graph `src→dst` ⇒ `W_res[dst, src] != 0` and `W_res[src, dst] == 0`. Code review must not add `.T` without a failing test that demanded it.

- [ ] Weight policy: `A[i,j] = log1p(synapse_count j→i)`, then scale so spectral radius equals `spectral_radius` (default 0.9). Empty/zero graphs fail loudly.

- [ ] `forward()` contracts (test both; these names should exist in `tests/test_reservoir.py`):
  - bearings / `head=rul`: 1D `[B]` **non-negative normalized RUL**. For ridge, `forward()` is `relu(W_out @ state + b_out)` with **no extra Softplus** (diverges from `RULHead`; see Ridge vs Softplus below). Filters unchanged.
  - filters / `head=weibull`: **tuple** `(lam, k)` after Softplus (same as `WeibullHead`).
  - Tests: `test_forward_bearings_returns_1d_nonneg_norm_rul`, `test_forward_filters_returns_lam_k_tuple`.

- [ ] Ridge vs Softplus (bearings only). Ridge readout for bearings: targets are `y_norm = target_rul_s / time_scale_s` (same space as Smooth L1). `forward()` returns `relu(W_out @ state + b_out)` (no extra Softplus). This makes the ridge solution exact. Document clearly in code comment that this diverges from RULHead's Softplus. Add test `test_ridge_bearings_no_double_softplus`: confirm `forward(x)` is non-negative and ridge residual is zero on training states to within 1e-3.

- [ ] Linear readout **raw** (used later by contributions): `raw = W_x @ x[T] + W_u @ u[T] + b`. Direct input term required. Identity tests in 03 use this raw **before** Softplus / Weibull median / `time_scale_s`.

- [ ] `FlyConnectomeReservoir`: builds from a connectome artifact (or in-memory graph in tests), freezes `W_in`, `W_res`, `b_res` (`requires_grad=False`), trainable readout only.

- [ ] `RandomReservoir`: degree-preserving **directed** rewiring of the **same** node set and edge count (matched N/E control). Same update, readout, `state_mode`, leak, spectral radius, input scale, seed protocol. Provenance: `graph_mode=random_rewire` (never `real_connectome`, never unlabeled synthetic) **and** `parent_graph_hash` of the graph that was rewired. Test: `test_random_reservoir_parent_graph_hash`.

- [ ] `synthetic_fixture` still clamps via 01 `resolve_n_nodes`; this subtask must not re-raise. `real_connectome` still validates 500–2000. Tests use **explicit** `n_nodes=8` or `16`, never 300/1000 against the fixture without expecting clamp.

- [ ] `build_model()` **may** construct reservoir modules for unit tests, but `run_training` still raises the 01 dedicated error until 02c (so no accidental AdamW-on-everything train). If `build_model` starts returning reservoirs, document that train is still blocked.

- [ ] Tests in `tests/test_reservoir.py`: `test_graph_orientation`, `test_state_update_hand_calculation`, `test_seed_reproducibility`, `test_random_reservoir_parent_graph_hash`, the two `forward()` contract tests, `test_ridge_bearings_no_double_softplus`, plus 01 clamp/label tests still pass. GRU suite still green. Ruff clean.

## Key Files to Create/Modify

**Create**

- `src/pdm/models/reservoir.py` — `LeakyESN` / `forward_states(...)` shared kernel; `window_reset`
- `src/pdm/models/fly_reservoir.py` — `FlyConnectomeReservoir`
- `src/pdm/models/random_reservoir.py` — `RandomReservoir` (rewire may live in `src/pdm/connectome/` ; keep it deterministic)
- `src/pdm/models/readout.py` — linear readout (`nn.Linear` or explicit `W_x`, `W_u`, `b`); ridge **solver function** may be sketched here but **must not** be called from `run_training` yet
- `tests/test_reservoir.py` — extend with tests listed in AC

**Modify**

- `src/pdm/models/__init__.py` — export new classes; `build_model` may dispatch to them for non-train callers
- `src/pdm/connectome/weights.py` — spectral-radius scaling (orientation already fixed in 01)
- `src/pdm/connectome/sampling.py` / new `rewire.py` — degree-preserving directed rewiring; record parent hash
- `src/pdm/connectome/provenance.py` — `parent_graph_hash` + `graph_mode=random_rewire` fields

**Do not modify**

- `run_training` epoch loop, `compatibility_dict`, `load_trained_model` (02c)
- Split protocol, `time_to_seconds`, `filters_full_history`
- `RecurrentEncoder` allowed names
- `configs/filters.yaml` (must still lack `readout: ridge`)

## Implementation Notes

### Orientation

`W_res @ x` is standard `sum_j W_res[i,j] * x[j]`, so column `j` is the **source**. Edge `j→i` must sit at `[i,j]`. Hand test (test 2): 2 nodes, `alpha=1`, `W_in=0`, `b=0`, `x[0]=[1,0]`, `W_res[1,0]=c`, others 0 → `x[1,1] = tanh(c)` (or the leaky combination if `alpha≠1`). Compute expected values in the test with numpy, **not** by calling the production function.

### forward() vs raw (C3 + W4)

Ridge (02c) fits **normalized RUL** (`y = target_rul_s / time_scale_s`), the same space `_run_epoch` uses for Smooth L1 on bearings.

Contribution identity (03) uses **pre-display raw** `W_x @ x + W_u @ u + b` **before** Softplus, Weibull median, and `time_scale_s`.

So this subtask must expose both:

1. `raw` affine output (vector length 1 for bearings, 2 for filters)
2. `forward()` after a non-negative map: bearings **ridge** uses `relu` (no extra Softplus; see below); filters use Softplus like `WeibullHead`

Do not implement contribution tests here beyond making `raw` accessible (e.g. `forward_raw`). Do not `nan_to_num` targets (no training loop yet).

### Random rewiring (W8)

Directed Maslov–Sneppen / degree-preserving swaps with a seeded RNG. Preserve in- and out-degree sequences as far as the algorithm guarantees; document if self-loops/parallels are forbidden. Rebuild `log1p` + spectral scale on the rewired edges — do not copy `W_res` from the fly graph. Always store `parent_graph_hash` and `graph_mode=random_rewire`. If the parent was synthetic, **both** disclaimers apply.

### Ridge vs Softplus (bearings only)

Ridge readout for bearings: targets are `y_norm = target_rul_s / time_scale_s` (same space as Smooth L1). `forward()` returns `relu(W_out @ state + b_out)` (no extra Softplus). This makes the ridge solution exact. Document clearly in code comment that this diverges from RULHead's Softplus. Add test `test_ridge_bearings_no_double_softplus`: confirm `forward(x)` is non-negative and ridge residual is zero on training states to within 1e-3.

### Isolation (partial; train isolation is 02c)

- Building a model whose `n_nodes` disagrees with a **provided artifact** `weights.npz` / graph size raises (artifact mismatch is not synthetic clamp).
- Same seed twice → identical `W_res` (after scaling) on the same parent graph.

### `hidden_size`

For reservoirs, `n_nodes` is the reservoir width. Do not require `recurrent_layers` to mean stacked ESNs. Checkpoint `meta` mapping is 02c.

## Dependencies

Subtask 01 (graph, provenance, clamp, YAML, names). Subtask 02a (models package).

## Verification Commands

```bash
.venv/bin/python -m pytest tests/test_spec_invariants.py -q
.venv/bin/python -m pytest tests/test_reservoir.py -q
.venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check src tests
```

Do **not** run `pdm train --arch fly_connectome_reservoir` as a quality gate here (still blocked until 02c). Tiny in-memory graphs only.

## Notes/Constraints

- GRU/LSTM `state_dict` keys and `PDMNet.forward` signatures must stay stable.
- `forward()` must be unpackable by today’s `_run_epoch` (1D vs `(lam, k)`).
- Do not transpose `W_res` to match literature that uses the opposite convention; this repo’s convention is `W_res[i,j]=j→i`.
- `synthetic_fixture` clamps; never auto-promote; tests pass explicit 8–16 node sizes.
- `random_reservoir` always records `parent_graph_hash` + `graph_mode=random_rewire`.
- Do not wire ridge into filters; do not call `run_training` for reservoirs yet.
- One tanh kernel only — 03 will bind `predict_with_trace` to this same function object.
- Ridge vs Softplus (bearings only): `forward()` is `relu(W_out @ state + b_out)` with no extra Softplus; test `test_ridge_bearings_no_double_softplus`.
