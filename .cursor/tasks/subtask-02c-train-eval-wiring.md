# Subtask 2c: Train/eval wiring (bearings + filters)

## Goal

Wire `fly_connectome_reservoir` and `random_reservoir` through `run_training`, `load_trained_model`, `evaluate_run`, and `Predictor` for **both** datasets without changing GRU/LSTM behavior, checkpoint `compat` bit-layout, or filters censoring. Bearings may use ridge **or** gradient readout; filters are gradient + `weibull_nll` only.

## Context

Phase B part 3. 02b produced frozen ESN weights, `forward_states`, and `forward()` contracts that match `_run_epoch`. Today `run_training()` still errors on reservoir strings (01), always does `AdamW(model.parameters())`, and `compatibility_dict()` writes only GRU keys (`dataset_id`, `architecture`, `history_length`, `hidden_size`, `recurrent_layers`, `head`, `feature_names`, `time_scale_s`, `split_hash`, plus optional `dataset_version` / `features_hash` / `units_hash`). `load_trained_model` always constructs `PDMNet`. Worker evaluate already checks `stop.flag` from 01.

## Acceptance Criteria

- [ ] `build_model()` dispatches all four architectures. `run_training()` / `load_trained_model()` use it. The 01 “not yet implemented” train error is gone for the two reservoir names.

- [ ] Optimizer iterates **trainable readout params only**. Frozen `W_in` / `W_res` / `b_res` have `requires_grad=False`. After a gradient step, `W_res` bytes are identical. Test: `test_frozen_reservoir_weights_not_updated`.

- [ ] **Bearings readout:** `ridge` (default from `model_defaults(dataset_id=bearings)`) **or** `gradient` + existing Smooth L1.
  - Ridge: collect last-step states via the **same** `forward_states` / `LeakyESN.forward` kernel on **TRAIN windows only**; solve `(ZᵀZ + λI) W = Zᵀ y` with `y = target_rul_s / time_scale_s` (normalized RUL, same space as Smooth L1); freeze readout.
  - Ridge path does **not** call `loss.backward()`.
  - Ridge still writes `best.pt`, `last.pt`, `status.json`, `experiment_snapshot.json`, `connectome/` artifacts (`provenance.json`, `graph.json`, `weights.npz`, `layout.json`).
  - `should_stop()` on the ridge path emits **`cancelled`**, not `completed`, not `stopped`.
  - Tests: `test_ridge_uses_forward_states_kernel`, `test_ridge_no_backward`, `test_ridge_writes_artifacts`, `test_ridge_stop_flag_sets_cancelled`.

- [ ] **Filters readout:** gradient + `weibull_nll` only. `readout=ridge` (YAML, CLI, or kwargs) **raises before any target is used** (before `UnitWindowDataset` targets are consumed / before `nan` filling). Clear message, e.g. ridge is unsupported for censored filters. Test: `test_filters_ridge_raises_before_targets`. `configs/filters.yaml` still has **no** `readout: ridge`. `model_defaults()` sets filters readout to `gradient`.

- [ ] **Never** `nan_to_num` censored `target_rul_s` to 0 for filters. Censored rows stay `event=0`; duration remains the observed prefix length. Test: `test_filters_censoring_not_rul_zero` (test 12) — a censored batch with `event=0` does not equal treating `target=0` as an observed failure.

- [ ] `compatibility_dict` / `_save_ckpt` `compat`: extra keys `n_nodes`, `graph_mode`, `graph_hash`, `state_mode`, `leak`, `spectral_radius`, `input_scale`, `seed` are **written only** when `architecture` is `fly_connectome_reservoir` or `random_reservoir`. GRU/LSTM blobs remain **bit-compatible** with today’s fixtures in `tests/test_spec_invariants.py` (same key set as now).

- [ ] `checkpoints_compatible(saved, current)` compares reservoir keys **only if both** saved and current architectures are reservoir (same optional pattern as `dataset_version` / `features_hash`: compare when present on **both** **and** both are reservoir). A GRU saved blob vs a current dict that happens to contain YAML reservoir defaults remains compatible. Test in `tests/test_spec_invariants.py`: **`test_gru_checkpoint_compat_ignores_reservoir_yaml_defaults`**. Also: `load_trained_model` still loads a GRU `best.pt` after `model_defaults()` has reservoir keys in YAML.

- [ ] `load_trained_model` for reservoir: read `connectome/weights.npz` + checkpoint readout **only**. If `weights.npz` is missing, **raise clearly** — never rebuild `W_in`/`W_res` from seed. Test: `test_load_trained_model_missing_weights_npz_raises`.

- [ ] Wrong `n_nodes` vs **saved artifact** size raises (test 3). Synthetic clamp already happened at graph build; this is checkpoint/artifact mismatch. Bearings checkpoint cannot load as filters (existing `dataset_id`/`head` plus reservoir graph keys when both are reservoirs).

- [ ] `random_reservoir` runs persist `parent_graph_hash` and `graph_mode=random_rewire` in provenance and reservoir `compat` as applicable.

- [ ] Smoke / CLI: honor explicit `--n-nodes` (8–16 in tests). Do not assume `smoke_n_nodes=300`. `synthetic_fixture` still clamps.

- [ ] `Predictor.predict_from_history` works for reservoir models via `predicted_rul_s` without seeing future rows or official RUL. Type hint `PDMNet` may become a protocol/`nn.Module`; do not add a second update loop.

- [ ] Train writes `runs/<dataset_id>/<run_id>/connectome/` as in the master plan. `experiment_snapshot.json` stores the resolved reservoir block.

- [ ] Tests: 3 (artifact mismatch), 5 (split/preprocess isolation), 12 (censoring); plus all named tests in this AC. Existing invariant + AppTest suites pass. Ruff clean.

## Key Files to Create/Modify

**Create**

- Ridge helpers may live in `src/pdm/models/readout.py` (called from train)

**Modify**

- `src/pdm/models/__init__.py` — `build_model` dispatches four architectures for train
- `src/pdm/train.py` — `build_model`; optimizer over trainable params; ridge branch for bearings (no `backward`); still `_run_epoch` for gradient; save frozen matrices in `connectome/weights.npz` **and** checkpoint; extend `compatibility_dict` **conditionally**; `_save_ckpt` `meta`; `load_trained_model` (no seed rebuild); `should_stop` → `cancelled` on ridge and epoch loops
- `src/pdm/evaluate.py` — no special-case leakage; `load_trained_model` must succeed for reservoir runs; `IncompatibleDataError` if graph/dataset mismatch
- `src/pdm/predict.py` — protocol/`nn.Module` with `predicted_rul_s`; no second tanh loop
- `src/pdm/config.py` — readout default by `dataset_id`; raise if filters + ridge
- `src/pdm/worker.py` — pass architecture / n_nodes through; stop → `cancelled` (already 01; keep for train ridge jobs)
- `src/pdm/cli.py` — pass `--n-nodes` / `--graph-mode` into `run_training`
- `tests/test_spec_invariants.py` — `test_gru_checkpoint_compat_ignores_reservoir_yaml_defaults` (and GRU `load_trained_model` with reservoir YAML defaults present)
- `tests/test_reservoir.py` — tests 3, 5, 12 and ridge/load tests named above

**Do not modify**

- Split protocol, `time_to_seconds`, `filters_full_history`
- `RecurrentEncoder` allowed names
- `configs/filters.yaml` adding `readout: ridge`
- GRU `compat` key set

## Implementation Notes

### Ridge vs gradient (C3)

- **Ridge (bearings only):** last-step `[x; u]` from `forward_states` on `UnitWindowDataset` **train split only**. `y` is **normalized RUL** (`target_rul_s / time_scale_s`). Do **not** use val/test windows to fit. Then freeze readout buffers.
- **Do not** `loss.backward()`; do not step AdamW on the ridge path. Optional eval of val MAE after solve to fill `best.pt` metrics is fine (`torch.no_grad`).
- Still emit `status.json`, write snapshot, `best.pt`/`last.pt`. If `should_stop()` fires, final status **`cancelled`**.
- **Gradient:** `nn.Linear` on concatenated `[x; u]`; `requires_grad` only there; reuse `_run_epoch` and existing losses. `forward()` already matches RULHead / WeibullHead from 02b.

### Filters ridge (C3.3, W3)

Raise at the start of `run_training` / `model_defaults` / readout factory when `dataset_id=="filters"` and `readout=="ridge"`, **before** iterating windows or reading `target_rul_s`. Do not construct a ridge target vector “and then notice censoring”.

### Censoring (C3.4)

`UnitWindowDataset` already keeps `target` as NaN when `target_rul_s` is missing (`src/pdm/train.py` ~76) and uses `event`. Do **not** add `nan_to_num(target, nan=0)`. Test 12 compares `event=0` NLL vs a counterfactual `event=1` and `target=0`.

### GRU compat (C2)

```text
# compatibility_dict
if is_reservoir(architecture):
    out["n_nodes"] = ...
    out["graph_mode"] = ...
    # etc.

# checkpoints_compatible
# existing required keys unchanged
if is_reservoir(saved["architecture"]) and is_reservoir(current["architecture"]):
    for k in reservoir_keys:
        if k in saved and k in current and saved[k] != current[k]:
            return False
```

`test_gru_checkpoint_compat_ignores_reservoir_yaml_defaults`: build a GRU `compat` like today’s fixture (no reservoir keys); build a “current” dict from `model_defaults()` on YAML that now contains `model.reservoir`; `checkpoints_compatible` is True; `load_trained_model` on a tiny GRU checkpoint succeeds.

### load_trained_model (W8)

```text
if is_reservoir(meta["architecture"]):
    weights_path = run_path / "connectome" / "weights.npz"
    if not weights_path.exists():
        raise FileNotFoundError("... weights.npz missing; will not rebuild from seed")
    # load arrays; attach frozen buffers; load readout from state_dict
```

### Isolation (tests 3–5)

- `n_nodes=16` model vs 8-node `weights.npz` raises.
- `checkpoints_compatible` false across `dataset_id` and across `fly_connectome_reservoir` vs `random_reservoir` vs `gru`.
- Fit preprocessor on train; mutate test-unit features; scaler unchanged; `W_in` unchanged (from file, not refit).
- Same seed twice → identical `W_res` and identical predictions on a frozen readout.

### Ridge target space vs contribution identity (W4)

**Write this distinction into code comments on the ridge solver and readout:**

Ridge closed-form fits **normalized RUL** (`y = target_rul_s / time_scale_s`), matching Smooth L1 in `_run_epoch`. The contribution identity test (03) does **not** use that display/loss space; it sums intercept + input + neuron terms on the **pre-Softplus / pre-median / pre-`time_scale_s` linear `raw`**. `forward()` still returns the `_run_epoch` contract (bearings: non-negative normalized RUL after Softplus; filters: `(lam, k)` after Softplus). Do not mix the two spaces in assertions.

### Leakage

`unit_id` / RUL / official overlay still not inputs. `W_in` shape `[n_nodes, n_features]` uses preprocessor `feature_names` order. Features already scaled by train-only scaler before `u[t]`.

## Dependencies

Subtask 02b (ESN, `forward()` contracts, fly/random, `parent_graph_hash`). Subtask 01 (clamp, CLI `--n-nodes`, `cancelled`, filters YAML without ridge).

## Verification Commands

```bash
.venv/bin/python -m pytest tests/test_spec_invariants.py::test_gru_checkpoint_compat_ignores_reservoir_yaml_defaults -q
.venv/bin/python -m pytest tests/test_spec_invariants.py -q
.venv/bin/python -m pytest tests/test_reservoir.py -q
.venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check src tests
```

Optional (not a quality gate; processed bearings must exist; **must** pass explicit tiny n_nodes):

```bash
.venv/bin/python -m pdm train --dataset bearings --arch fly_connectome_reservoir --smoke --epochs 1 --max-windows-per-unit 8 --n-nodes 8
```

Skip in CI if data missing. Never omit `--n-nodes` and assume 300 fixture nodes.

## Notes/Constraints

- GRU/LSTM `compat` must stay bit-compatible with current invariant fixtures (C2).
- Filters: **no ridge**; **no** censored-as-failure; **no** `nan_to_num` to 0 (C3).
- Ridge: shared `forward_states` kernel, train windows only, no `backward`, artifacts + `cancelled` on stop (C3).
- Missing `weights.npz` → raise, never seed-rebuild (W8).
- `synthetic_fixture` clamp from 01 still applies; smoke uses `--n-nodes 8`.
- Do not transpose `W_res`.
- Cancel/stop during train still must not emit `completed` or `stopped`.
