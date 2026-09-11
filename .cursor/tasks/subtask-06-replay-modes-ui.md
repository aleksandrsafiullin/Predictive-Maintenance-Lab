# Subtask 6: Validation / Test / Research replay modes (R3 UI)

## Goal

Add Validation / Test / Research modes on the existing Test & Replay screen (no redesign). Freeze only from Validation (full val unit list). Test Evaluate is a blind frozen benchmark or an error if nothing is frozen. Research is **test-split** exploratory, labeled not-blind, and never writes `alert_policy.json`. Bind Evaluation picker and Play to the mode’s split / mask.

## Context

Review R3 (HIGH). Today `screen_replay` (`src/pdm/app.py`) unit picker is `split['test']` only; Freeze always `source='validation_ui'`; Evaluate worker job always sends widget H/K. Existing AppTest clicks Freeze on that screen — **must** be updated.

**Plan-review Warnings:** Validation Evaluate must not first-write `alert_policy.json`. `_render_evaluation_panel` must not reindex to `test_units`. Default mode Validation. Test Evaluate with missing freeze must not `spawn_worker`.

AppTest is the UI gate. Browser playback is optional follow-up, not acceptance.

## Acceptance Criteria

- [ ] Mode control on `screen_replay`: **Validation**, **Test**, **Research**. **Default = Validation** (so Freeze AppTest has one path; do not require switching mode first). English copy. No new screens / no FastAPI.
- [ ] Unit dropdown: Validation → `split['validation']`; **Test and Research → `split['test']` only**. Do **not** offer validation units in Research.
- [ ] **Validation:** H/K/lead widgets enabled. Freeze writes `source='validation_ui'` plus `split='validation'`, `unit_ids` = **full** `split['validation']` (not the selectbox unit), checkpoint hash, policy hash, timestamp (backend from subtask 05). Evaluate job: `split_name='validation'`, `policy_mode='research'` (or write-identical `'validation'`), **may include H/K**, `blind_benchmark` false. Must **not** create `alert_policy.json`.
- [ ] **Test:** Freeze disabled (no button, or button no-ops with an error). Cannot persist `source='validation_ui'`. Evaluate job: `split_name='test'`, `policy_mode='frozen'`, **no H/K keys** (`H_trigger`, `warning_horizon_s`, `confirmation_count`, `minimum_action_lead_time`, `max_useful_horizon_s` absent). If `alert_policy.json` is missing, show error **Freeze from Validation first** and **do not** `spawn_worker`. Caption: frozen model+policy test evaluation — only after a freeze exists.
- [ ] **Research:** widgets enabled for in-memory rescore on **test** units. Freeze does not write `alert_policy.json`. Any evaluate from this mode is `policy_mode='research'`, `blind_benchmark` false. Missing freeze stays missing (05).
- [ ] **Evaluation selectbox** lists only evals whose `evaluate_mask.split` matches the mode. In **Test**, also require `evaluate_mask.blind_benchmark` true (exclude research and validation evals).
- [ ] **Play** is disabled/blocked if selected `uid not in evaluate_mask.unit_ids` for the chosen eval (or no matching eval). Test mode must **not** offer a validation eval as the Play source.
- [ ] `_render_evaluation_panel` and in-view `metrics_by_unit_display_frame` take selected eval `evaluate_mask.unit_ids` (fallback: current mode split list). Do **not** hardcode `test_units` / `split['test']`.
- [ ] Existing freeze AppTest: default mode is already Validation — Freeze without a mode switch; assert `source`, `split`, **full val `unit_ids`**, hashes. New AppTests:
  - Test mode: no successful Freeze-to-`validation_ui`.
  - Validation Evaluate captured job has `split_name='validation'`, `policy_mode='research'` (or `'validation'`), may include H/K; after click, `alert_policy.json` is **not** created.
  - Test Evaluate with **missing** freeze does **not** `spawn_worker` (no job queued).
  - Test Evaluate with freeze present: captured job has `policy_mode='frozen'` and **no H/K keys**.
  - Test mode Evaluation options do not include a validation `eval_id`.
  - Validation mode + validation `eval_id` → per-unit table lists a **validation** uid, not only `split['test']`.
  - Play disabled when `uid not in evaluate_mask.unit_ids`.
- [ ] Research vs frozen policy mismatch still rescores from predictions (subtask 04 `pressure_limit_pa`) and does not rewrite `alerts.csv` of a prior blind eval.
- [ ] Do not change splits to improve numbers. Already-viewed test results are not relabeled as a new blind check.

## Implementation Notes

**Files:** `src/pdm/app.py` (`screen_replay`, Freeze ~736, Evaluate spawn ~751, unit selectbox ~647, evaluation selectbox ~773, `_render_evaluation_panel` ~925, `metrics_by_unit_display_frame` ~894, Play gating, captions), `src/pdm/worker.py` (pass through `split_name`, `policy_mode`; Test job must not include widget H/K), `src/pdm/experiments.py` (`list_evaluations` filter helper if cleaner than inline), `tests/test_worker_and_app.py` (freeze test ~519 and neighbors)

- Keep three screens (Data / Train / Test & Replay). Mode is a radio/select on Test & Replay only.
- Play remains blocked until matching eval artifacts exist. Do not run Predictor on the Streamlit request thread.
- Worker job from Validation Evaluate: `{kind: evaluate, dataset_id, run_id, split_name: validation, policy_mode: research}` (or `validation`) **plus** widget H/K. Must not call Freeze/save.
- Worker job from Test Evaluate: `{kind: evaluate, dataset_id, run_id, split_name: test, policy_mode: frozen}` only — **omit** H/K fields. If freeze file missing, **return before** `spawn_worker`.
- Copy: “Test evaluation uses the frozen validation-selected policy.” / “Freeze from Validation first.” / “Research — not a blind benchmark.”
- Pressure limit for rescore: 04 threads `pressure_limit_pa`. When `experiment_snapshot.json` exists, **07** switches the source off live YAML (do not re-read live config here if 07 has landed).
- `st.rerun` after Freeze stays; archive + provenance round-trip is backend (05).

**Tests:** `tests/test_worker_and_app.py` AppTest. Keep invariant tests from 05.

## Dependencies

Subtask 05.

## Verification

```bash
.venv/bin/python -m pytest tests/test_worker_and_app.py -q
.venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check src tests
```

Optional (not a gate): click through Validation freeze → Test evaluate on `http://127.0.0.1:8501`.
