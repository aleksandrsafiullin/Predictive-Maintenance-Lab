# Subtask 8: Full regression gate and smoke-hash note (R7 optional)

## Goal

Run the full pytest + ruff gate after R1–R6. Fix leftover breaks from `METRICS_VERSION`, Freeze AppTest, gap-version gates, and policy-write rules. Replace hardcoded `"v0"` in AppTest fixtures with `METRICS_VERSION`. Optionally add a **short** report note that old smoke hashes must not be reused as quality — **no new metrics, no Full training**.

## Context

Review step 4 = full regression; R7 quality rewrite is **out of scope**. `reports/implementation_report.md` still cites smoke blob hash `33d60a985e5bfd0bae49762ac88764ef92b07bb3`. Do not invent replacement MAE/NLL. Do not run Full 30-epoch training. `--smoke` is not a quality claim. Browser playback is optional, not a gate.

`tests/test_worker_and_app.py` currently hardcodes `"metrics_version": "v0"` in evaluation fixtures (~436, ~442, ~651, ~825). After R4 those must follow `METRICS_VERSION`.

## Acceptance Criteria

- [ ] `.venv/bin/python -m pytest tests -q` passes.
- [ ] `.venv/bin/python -m ruff check src tests` passes.
- [ ] `tests/test_spec_invariants.py` includes the review fixtures: sampler epoch/resume and epoch-1 ≠ seed+0; timestamps `[0,6,12,42,142,242,342]` offline/online match; poisoned stored flags cannot drive `build_windows`; missing `gap_rule_version` refuses train/eval; coverage 100/75/30/10 not miss; sensor-limit CSV vs in-memory vs rescore at limit 500 / Δp 550; research eval does not create `alert_policy.json`; provenance round-trip.
- [ ] `tests/test_worker_and_app.py`: Validation Freeze provenance (full val `unit_ids`); Test mode cannot save `source='validation_ui'`; Test Evaluate job `policy_mode='frozen'` with no H/K keys; Test mode does not Play from a validation eval. Fixtures use `METRICS_VERSION` (import from `pdm.evaluate`), **not** hardcoded `"v0"`.
- [ ] Optional (do it): one short paragraph or callout in `reports/implementation_report.md` that post-R1–R5 pipeline versions **must not** reuse the old smoke checkpoint/blob hashes as quality evidence; next Full run is a **future** human step after a new prepare. **Do not** add fake tables or copied smoke numbers.
- [ ] No `filters_full_history` enablement. No split protocol edits. No new architectures.

## Implementation Notes

**Files:** leftovers that fail the suite after 01–07; `tests/test_worker_and_app.py` (`metrics_version` fixtures); optionally `reports/implementation_report.md` (a few sentences, not a rewrite).

- If a test still encodes `coverage_horizon = max(H, min_lead)`, full-file prepare eligibility, or `ensure_alert_policy` first-write as OK for research, fix the test to the new contract.
- Do not queue `pdm train --smoke` or Full as verification of model quality.
- Do not treat this subtask as a place to “improve” test MAE by changing splits.

## Dependencies

Subtasks 01–07.

## Verification

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check src tests
```

Optional follow-up (not a gate): Streamlit at `http://127.0.0.1:8501` Validation freeze → Test evaluate → Research rescore.
