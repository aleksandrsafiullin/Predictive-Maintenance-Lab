# Subtask 2a: models.py → package (GRU/LSTM stay green)

## Goal

Replace the `src/pdm/models.py` **module** with package `src/pdm/models/` that re-exports `PDMNet`, `RecurrentEncoder`, `RULHead`, `WeibullHead`, and `build_model` unchanged so every existing import and GRU/LSTM test stays green. Do **not** add Echo State Network math in this subtask.

## Context

Phase B part 1. Subtask 01 added `build_model()` on the module and architecture name constants. Reservoir files in 02b need `src/pdm/models/reservoir.py` beside the GRU code, which is impossible while `models.py` is a single file. This conversion is high import-risk and must be proven green **before** ESN code.

Today `from pdm.models import PDMNet` is used by `src/pdm/train.py`, `src/pdm/predict.py`, `src/pdm/cli.py`, and `tests/test_spec_invariants.py`. `RecurrentEncoder` still rejects anything except `gru`/`lstm`.

## Acceptance Criteria

- [ ] `src/pdm/models.py` **does not exist**. Package `src/pdm/models/` exists.
- [ ] `src/pdm/models/recurrent.py` contains the previous module body (GRU/LSTM encoder + heads + `PDMNet`) with the same `forward` / `predicted_rul_s` signatures.
- [ ] `src/pdm/models/__init__.py` re-exports at least: `PDMNet`, `RecurrentEncoder`, `RULHead`, `WeibullHead`, and `build_model` (from 01). `from pdm.models import PDMNet` works.
- [ ] `RecurrentEncoder` still raises `architecture must be gru or lstm` for any other string (including `fly_connectome_reservoir`).
- [ ] `build_model("gru"|"lstm")` still returns `PDMNet` with the same constructor arguments as 01. Reservoir strings still raise the dedicated not-yet-implemented error from 01 (02c removes that raise).
- [ ] GRU/LSTM `state_dict` key names are unchanged. Checkpoint reload tests in `tests/test_spec_invariants.py` pass without rewriting fixtures.
- [ ] `tests/test_spec_invariants.py::test_gru_lstm_both_heads_change_weights` passes without assertion edits.
- [ ] No reservoir modules are required yet (`reservoir.py` / `fly_reservoir.py` may be absent). Do not add reservoir keys to `compatibility_dict`.
- [ ] Full `tests/test_spec_invariants.py` and `tests/test_worker_and_app.py` pass. Ruff clean.

## Key Files to Create/Modify

**Create**

- `src/pdm/models/__init__.py` — re-exports + `build_model` if it moves here
- `src/pdm/models/recurrent.py` — current `models.py` contents moved verbatim (plus 01 `build_model` if it lived in `models.py`)

**Modify**

- **Delete** `src/pdm/models.py` in the same change that adds the package (never leave both on disk)
- Import sites need **no** edits if re-exports are complete (`train.py`, `predict.py`, `cli.py`, tests)

**Do not modify**

- `src/pdm/train.py` loop, `compatibility_dict`, `_run_epoch`
- `src/pdm/losses.py`, splits, windows
- Adding `LeakyESN` / readout / fly-random classes (02b)

## Implementation Notes

Do this **first**, run GRU tests, **then** stop. Suggested order:

1. Create `src/pdm/models/recurrent.py` with the current classes (`importlib` path `pdm.models.recurrent`).
2. `__init__.py`:

   ```python
   from pdm.models.recurrent import (
       PDMNet,
       RecurrentEncoder,
       RULHead,
       WeibullHead,
       build_model,  # if defined in recurrent.py
   )
   __all__ = [...]
   ```

3. Delete `src/pdm/models.py` immediately (Python will otherwise prefer the module over the package, or fail ambiguously).
4. Run `tests/test_spec_invariants.py` before writing any ESN file.

If `build_model` was added on the old module in 01, move it with the file. Do not change its reservoir error string in 02a.

Keep `PDMNet.forward` return types exactly: bearings 1D tensor; filters `(lam, k)` tuple. `_run_epoch` in `src/pdm/train.py` unpacks those today (~884–890).

## Dependencies

Subtask 01 (`build_model` hook and architecture names). If 01’s `build_model` is still missing, include a pass-through `build_model` here that only handles gru/lstm so imports do not break — still no ESN.

## Verification Commands

```bash
.venv/bin/python -c "from pdm.models import PDMNet, RecurrentEncoder, RULHead, WeibullHead; from pathlib import Path; assert not Path('src/pdm/models.py').exists()"
.venv/bin/python -m pytest tests/test_spec_invariants.py::test_gru_lstm_both_heads_change_weights -q
.venv/bin/python -m pytest tests/test_spec_invariants.py tests/test_worker_and_app.py -q
.venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check src tests
```

## Notes/Constraints

- NEVER leave both `src/pdm/models.py` and `src/pdm/models/` on disk.
- NEVER change GRU/LSTM `state_dict` keys, `forward` signatures, or `compat` blobs.
- NEVER start ESN / ridge / connectome weight code in this subtask.
- NEVER treat this as done if `test_spec_invariants.py` is red.
- Smoke training is not a gate.
