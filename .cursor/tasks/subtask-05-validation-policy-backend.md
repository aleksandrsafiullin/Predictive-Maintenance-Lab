# Subtask 5: Validation vs test policy backend (R3, no UI chrome)

## Goal

Generate predictions for `split_name='validation'|'test'` with an explicit evaluate mask. Freeze is the **only** writer of `source='validation_ui'` and the only way to create `alert_policy.json`. Validation Evaluate is `policy_mode='research'` (never writes). Test evaluation uses the frozen file or **fails**. CLI with no H/K flags is `frozen` (fail if missing), not a first-write.

## Context

Review R3 (HIGH). `screen_replay` only lists `split['test']`. Freeze still writes `source='validation_ui'`. `evaluate_run` is test-bound; explicit CLI/worker H/K **override** frozen policy. Not weight leakage (checkpoint still chosen on val) but it invites tuning alerts on test.

**Plan-review Critical:** `ensure_alert_policy(overwrite=False)` is still a write. Research CLI/widget H/K with no `alert_policy.json` would freeze those values on first evaluate. That is not allowed.

This subtask is **backend + tests**. UI modes are subtask 06. Do not change 9/3/3 or 40/10/50.

## Acceptance Criteria

- [ ] Shared helper (e.g. `evaluate.generate_split_predictions` or `replay.replay_split_units`) takes `split_name: Literal['validation','test']` (and explicit `unit_ids`). `evaluate_run(..., split_name='test')` keeps today’s default. Evaluate mask in `evaluation_config.json` records `split`, `protocol`, `unit_ids`, and `blind_benchmark` (true only for frozen test eval).
- [ ] Writes split by `policy_mode` (do **not** use `ensure_alert_policy` as a catch-all):
  - **Validation Evaluate:** `split_name='validation'`, `policy_mode='research'` (or a dedicated `'validation'` alias that is write-identical). Widget/CLI H/K **allowed**. `evaluate_mask.blind_benchmark: false`. **Never** call `ensure_alert_policy` or `save_alert_policy`. Missing `alert_policy.json` stays missing. evaluation_config `source` is `research` / `validation_eval` — **not** `validation_ui`.
  - **`research` / CLI H/K overrides (test split):** same write ban. Missing file stays missing. `blind_benchmark: false`.
  - **`frozen` test:** if `alert_policy.json` is missing, **fail** with an error that requires Validation Freeze first. Do **not** write resolved CLI/widget args. Do **not** write YAML `source='evaluate_default'`. Ignore any present H/K args; use the frozen file only.
  - **`validation_ui`:** **Freeze** (`save_alert_policy` from the validation freeze path) is the **only** writer of that source. Validation **Evaluate** must not create it.
- [ ] Frozen payload after `AlertPolicy.to_dict()`: **merge provenance** so it is not stripped — `source`, `split='validation'`, `unit_ids` = **full** `split['validation']` from the run snapshot (not the UI selectbox unit), `checkpoint_hash`, `policy_hash`, `frozen_at`. `from_mapping` / `load_alert_policy` **round-trip** those keys.
- [ ] Overwriting policy copies the previous `alert_policy.json` into `runs/<run_id>/alert_policies/<frozen_at>_<hash8>.json` (or equivalent) **before** replacing the current file. Old provenance is not erased. Round-trip test is required in addition to the archive copy test.
- [ ] `save_alert_policy` / `build_alert_policy` refuse `source='validation_ui'` when the recorded split is `test`.
- [ ] Worker `evaluate` job accepts `split_name` and `policy_mode`. CLI `pdm evaluate` grows `--split {validation,test}` (default test). **No** `--horizon-s` / `--k` / `--min-action-lead-s` / `--max-useful-horizon-s` → `policy_mode='frozen'` (fail if no `alert_policy.json` — cannot silently first-write). **Any** of those flags present → `policy_mode='research'`, never write.
- [ ] Test evaluation with a freeze present reproduces frozen H/K/lead (same `policy_hash` in `evaluation_config.json` as `alert_policy.json`) even when H/K args are passed.
## Implementation Notes

**Files:** `src/pdm/evaluate.py` (`evaluate_run`, `_evaluation_config`, `bind_evaluation_to_run` test_ids vs val_ids — **stop calling `ensure_alert_policy` from research/frozen-missing paths**), `src/pdm/alerts.py` (`save_alert_policy`, `AlertPolicy.to_dict` / `from_mapping` / `load_alert_policy` provenance merge, `resolve_alert_policy`, `build_alert_policy`), `src/pdm/cli.py`, `src/pdm/worker.py`, `src/pdm/replay.py` if the loop is extracted there.

- `bound["test_ids"]` stays for test. Add `validation_ids` from `split['validation']` in the run snapshot — never from a live split when the snapshot exists.
- `resolve_alert_policy` currently prefers explicit args over frozen. Frozen test eval **ignores args**. Research and Validation Evaluate use args and **do not write**. Default CLI (no H/K flags) is frozen even when `split_name='validation'` unless flags are passed — UI Validation Evaluate always sends `policy_mode='research'` (or `'validation'`) so it is not frozen.
- Policy hash stays over H/K/lead/K/reset/max_useful only. Provenance keys are extra JSON **re-attached after `to_dict()`**.
- Error string for missing freeze (frozen test): include **Freeze from Validation first** (UI will surface the same text in subtask 06).
- Do not rewrite `predictions.csv` when only the policy changes. Validation vs test are **different** `evaluations/<eval_id>/` directories.

**Tests** in `tests/test_spec_invariants.py` (tmp_path run dir, no AppTest yet):

- `test_evaluate_mask_validation_vs_test_unit_lists`
- `test_frozen_test_eval_ignores_hk_overrides` — **extend:** H/K args present; policy_hash still matches frozen file; file contents unchanged.
- `test_frozen_test_eval_without_policy_file_fails` — no `alert_policy.json` → error, file still absent.
- `test_research_eval_does_not_create_alert_policy_json` — missing file stays missing after research/CLI-override eval.
- `test_validation_evaluate_does_not_create_alert_policy_json` — `split_name='validation'`, `policy_mode='research'` (or `'validation'`), H/K args present, no freeze file → eval may succeed as non-blind; file still absent.
- `test_cli_evaluate_without_hk_flags_is_frozen` — no `--horizon-s`/etc. → `policy_mode='frozen'`; missing file fails; no write.
- `test_cli_evaluate_with_hk_flags_is_research_never_writes`
- `test_research_eval_does_not_overwrite_alert_policy_json` — existing file unchanged.
- `test_save_alert_policy_archives_previous_version`
- `test_alert_policy_provenance_round_trip` — `save` then `load_alert_policy` keeps `split`, **full** `unit_ids`, `checkpoint_hash`, `frozen_at`, `source='validation_ui'`.
- `test_validation_ui_source_rejected_for_test_split`

## Dependencies

Subtask 04 (export/rescore contract used when generating split predictions).

## Verification

```bash
.venv/bin/python -m pytest tests/test_spec_invariants.py -q -k "alert_policy or evaluate_mask or frozen or research"
.venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check src tests
```
