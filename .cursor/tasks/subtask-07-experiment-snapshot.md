# Subtask 7: Experiment snapshot semantics (R6)

## Goal

Persist the full resolved task config, preprocessing hash, source commit, and metric-policy version with the run. Resume and evaluate must not silently replace saved preprocessing or live-YAML task settings. `--force` records the checkpoint bytes **actually used**. Distinguish “evaluation method changed” from “data incompatible”.

## Context

Review R6 (second priority). `evaluate_run()` still reads live dataset YAML for pressure limit and near-event zones. Run `config.yaml` has model/gap but not a full task snapshot. With `--force`, `checkpoint_hash` in the new evaluation can still be the **previous expected** hash (`run_fp.get("checkpoint_hash") or checkpoint_hash(best.pt)`).

Keep this subtask small. Do not Full-train. Do not enable `filters_full_history`.

**DAG:** this subtask depends on **05** (`evaluate_run` split/policy_mode API), **not** 06. 06 is UI-only. Orchestration may still run 06 before 07 to serialize `evaluate.py` if both would edit it; that is a merge convenience, not a logical dependency.

## Acceptance Criteria

- [ ] At train (new run), write `runs/<run_id>/experiment_snapshot.json` (name may vary but must be loaded by evaluate/resume) containing: resolved dataset config used (including `pressure_limit_pa`, `evaluation.near_event_zones_s`, `alerts`, `gap`, `model`), `preprocessing_hash` (SHA256 of `preprocessing.json`), `source_commit` (best-effort `git rev-parse HEAD`; null if unavailable — do not fail train), `metrics_version`, `gap_rule_version`, `feature_pipeline_version`.
- [ ] Resume: if `preprocessing.json` exists, **load** it (`Preprocessor.from_dict`) and transform with it. Do **not** overwrite it with a newly fitted preprocessor from live YAML/data when hashes match. If live fit / live YAML would disagree with the snapshot on scaler/maps/`time_scale_s`/`feature_pipeline_version`, abort with a clear error (same family as checkpoint incompat) — do not silently replace.
- [ ] `evaluate_run` / `build_rul_metrics` read `pressure_limit_pa` and `near_event_zones_s` from the run snapshot, not `load_dataset_config` live YAML, when the snapshot exists.
- [ ] When `experiment_snapshot.json` exists, `src/pdm/app.py` reads `pressure_limit_pa` from it for rescore / Δp line (same source as `evaluate_run`), not live YAML. Legacy runs without a snapshot keep the 04 fallback.
- [ ] `bind_evaluation_to_run`: fingerprint / file-hash / split / **`gap_rule_version` vs expected constant** mismatches stay `IncompatibleDataError` (**data incompatible**). Metrics/schema/`METRICS_VERSION` / coverage-definition drift is **evaluation method changed**: allow a new `eval_id`, set `evaluation_config.evaluation_method_changed=true` (and record snapshot vs current `metrics_version`). Do not raise `IncompatibleDataError` solely because `METRICS_VERSION` bumped after R4.
- [ ] `--force`: `evaluation_config.checkpoint_hash` is `checkpoint_hash(rdir / "best.pt")` of the **bytes loaded**. If it differs from `run_fp["checkpoint_hash"]`, also store `expected_checkpoint_hash`. Never write only the old expected hash when bytes differ.
- [ ] UI must not pass `force=True` (already true).

## Implementation Notes

**Files:** `src/pdm/train.py` (`run_training` after writing `preprocessing.json` / `config.yaml`; `_environment` may gain `source_commit`; resume branch before `fit_preprocessor`), `src/pdm/evaluate.py` (`evaluate_run`, `_evaluation_config`, `bind_evaluation_to_run`, `load_near_event_zones_s` call sites), `src/pdm/app.py` (`pressure_limit_pa` on the replay view ~913 — snapshot over live YAML), `src/pdm/io_util.py` (`checkpoint_hash` already hashes file bytes)

- Prefer loading snapshot in evaluate **and** in `app.py` for `pressure_limit_pa`; fall back to live YAML only for **legacy** runs that lack the file, and record `snapshot_missing: true` in evaluation_config. Shared helper is fine (e.g. `load_run_pressure_limit_pa(rdir, fallback)`).
- Git commit: `subprocess` best-effort from `project_root()`; never `git config` writes.
- Do not treat preprocessing hash mismatch as “evaluation method changed” — that is data/run incompatibility.
- Do not reopen R3 policy_mode rules; call the 05 API.

**Tests** in `tests/test_spec_invariants.py`:

- `test_evaluate_uses_snapshot_pressure_limit_not_live_yaml` (tmp_path: snapshot 500, live yaml 600).
- `test_resume_does_not_rewrite_preprocessing_json` (hash stable).
- `test_force_eval_checkpoint_hash_is_actual_bytes`.
- `test_metrics_version_drift_is_method_changed_not_incompatible_data`.

## Dependencies

Subtask **05** (`evaluate_run` API: `split_name`, `policy_mode`, no research writes). **Not** subtask 06.

## Verification

```bash
.venv/bin/python -m pytest tests/test_spec_invariants.py -q -k "snapshot or checkpoint_hash or preprocessing"
.venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check src tests
```
