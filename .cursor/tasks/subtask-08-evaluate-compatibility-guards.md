# Subtask 8: Evaluate/replay compatibility guards

## Goal

`evaluate_run`, `load_trained_model`, and replay entry points abort when checkpoint data/split/preprocessing fingerprints do not match the run snapshot; on success, bind evaluation to the run's saved split and fingerprinted data.

## Context

P0-3: Prevents evaluating an old checkpoint on re-prepared data that includes former train units in test. TZ requires evaluation use the run's split/data snapshot **or abort** — never silently substitute live processed split.

## Acceptance Criteria

- [ ] `evaluate_run` loads `dataset_fingerprint.json` + `split.json` from run dir; compares field-by-field to current processed fingerprint; mismatch aborts (CLI `--force` optional, not UI default).
- [ ] **On success path:** `test_ids` and split protocol come from `rdir/split.json`; processed `features`/`units` must match `dataset_fingerprint.json` field-by-field. **Never use live split from `load_processed()` when a run snapshot exists.**
- [ ] Mismatch raises clear error listing differing fields (`dataset_version`, `split_hash`, `features_hash`, `units_hash`, `feature_pipeline_version`, `checkpoint_hash`).
- [ ] `checkpoints_compatible` extended with fingerprint fields; uses canonical `split_hash` name; **also accepts legacy `split_fingerprint` from old run dirs**.
- [ ] **`test_evaluate_rejects_changed_dataset_or_split` passes (hard gate for this subtask).**

## Implementation Notes

**Files:** `src/pdm/evaluate.py`, `src/pdm/train.py`, `src/pdm/experiments.py`, `src/pdm/replay.py`, `src/pdm/cli.py`

- `checkpoint_hash` = hash of `best.pt` bytes or state_dict keys+shapes.
- Load features/units from processed dir only after fingerprint match; filter rows to `split.json` test IDs from run dir.
- UI: show "incompatible data" message instead of silent wrong metrics.
- Replay historical mode: require fingerprint match; load `evaluations/<eval_id>/predictions.csv` from matching run.

## Dependencies

Subtasks 6, 7.

## Verification

```bash
.venv/bin/python -m pytest tests/test_spec_invariants.py::test_evaluate_rejects_changed_dataset_or_split -q
.venv/bin/python -m ruff check src/pdm/evaluate.py src/pdm/train.py
```

This subtask is **not complete** until `test_evaluate_rejects_changed_dataset_or_split` passes.
