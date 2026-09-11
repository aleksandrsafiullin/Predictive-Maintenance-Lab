# Subtask 1: Epoch-aware unit-balanced sampler (R1)

## Goal

Stop `UnitBalancedSampler` from drawing the **same** with-replacement window set every epoch. Make the schedule reproducible per `(seed, epoch)` and continue correctly on resume. Never draw under default epoch 0.

## Context

Review R1 (HIGH), commit `45a45968`. `__iter__` currently does `np.random.RandomState(self.seed)` every epoch; `run_training` never calls `set_epoch`. Isolated probe: 10 units × 100 windows, 1000 draws, seed=42, 30 epochs → identical sequences; 356 windows never get a gradient. Replacement sampling itself is allowed. Do **not** promise 100% coverage per epoch.

`seed + 0` is **not** the first training epoch. Training epochs are 1-based (`for epoch in range(start_epoch, max_epochs + 1)` with `start_epoch` 1 on a new run).

## Acceptance Criteria

- [ ] `UnitBalancedSampler` exposes `set_epoch(epoch: int)` and draws with `RandomState(seed + epoch)` (or a persistent RNG whose state is saved/restored).
- [ ] `run_training` calls `sampler.set_epoch(epoch)` **inside** the 1-based epoch loop, **before** iterating `train_loader` that epoch. Do not rely on a default epoch of 0. `__iter__` without `set_epoch` for that epoch must not be used for gradient steps (raise or leave epoch unset until `set_epoch` — pick one and test that training never samples `seed+0` as “epoch 1”).
- [ ] Resume from `last.pt` (`start_epoch = meta["epoch"] + 1`) uses `set_epoch(start_epoch)` at the top of that loop iteration, not epoch 0. Epoch 11 after a stop at 10 must match a fresh sampler at epoch 11, not epoch 1.
- [ ] Diagnostics logged **per epoch** (status.json emit and/or that epoch’s `training_history.csv` row): `n_eligible_windows`, `n_gradient_draws`, `n_unique_sampled_windows`. Unique count is that epoch’s sampled index set, **not** run-cumulative. Do not claim 100% window coverage for replacement sampling.
- [ ] Different epochs → different index sequences (assert inequality on the review-style fixture). Same `(seed, epoch)` → identical sequence.
- [ ] Smoke cap `UnitWindowDataset(..., max_windows_per_unit=..., seed=)` is unchanged (one-time cap, not per-epoch).

## Implementation Notes

**Files:** `src/pdm/train.py` (`UnitBalancedSampler.__init__`, `__iter__`, new `set_epoch`; `run_training` epoch loop ~596; `windows_per_unit_summary` / `emit` / history row)

- Preferred: `rng = np.random.RandomState(int(self.seed) + int(self.epoch))` inside `__iter__` after `set_epoch`. Simpler than pickling RNG state into `last.pt`.
- Initialize `self.epoch` to `None` (or a sentinel) so accidental iteration before `set_epoch` fails loudly rather than silently using 0.
- `n_draws` stays `max(len(train_ds), 1)` unless you have a reason to change it. Replacement is OK.
- Unique indices: collect from the sampler yield **this epoch** only (`train_loader` / sampler), not `train_diag_loader`.
- Copy: never say “every window is updated every epoch”.

**Tests** in `tests/test_spec_invariants.py` (new, synthetic, no real data):

- `test_unit_balanced_sampler_epoch_changes_sequence`
- `test_unit_balanced_sampler_seed_epoch_reproducible`
- `test_unit_balanced_sampler_resume_does_not_restart_at_epoch_zero` — iterate epoch 1 then 2 on A; B with `set_epoch(2)` matches A’s second epoch, not the first.
- `test_unit_balanced_sampler_epoch_one_is_not_seed_plus_zero` — sequence at `set_epoch(1)` ≠ sequence at epoch 0 / unset-equivalent.
- Unique-count < n_eligible on a many-windows/few-draws fixture for **one** epoch (diagnostics are distinct; not cumulative across epochs).

Do not add extra NN architectures. Do not change split protocol.

## Dependencies

None.

## Verification

```bash
.venv/bin/python -m pytest tests/test_spec_invariants.py -q -k "sampler or unit_balanced"
.venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check src tests
```

`--smoke` training is **not** a quality claim and is not required to close this subtask.
