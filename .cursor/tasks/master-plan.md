# Master Plan: Review patch 45a45968 (Predictive Maintenance Lab)

**Source of truth:** `Predictive_Maintenance_Lab_Update_Review_45a45968.md` against commit `45a45968c4b2688b374db207b072b5bce8fa74ba`  
**Repo:** `src/pdm/` · **Stack:** Python ≥3.11 · PyTorch · Streamlit · Plotly · pytest · ruff  
**Date:** 2026-09-11 (revised after plan-review **REVISE**, then remaining Warnings)  
**Replaces:** previous sprint plan (P0/P1 preprocessing, fingerprints, immutable eval, replay UI). That sprint is done. This file is a **new patch**, not a continuation of those subtask numbers.

## Overview

Close remaining train/eval/replay discrepancies **without** changing split protocol, architectures, or enabling `filters_full_history`. No FastAPI/React/Docker/MLflow/Transformers. No Full 30-epoch training as an acceptance gate. `--smoke` is not a quality claim. Real data only; do not invent MAT/CSV fields. UI English.

| ID | Priority | Theme |
|----|----------|--------|
| R1 | HIGH | `UnitBalancedSampler` repeats the same with-replacement draw every epoch |
| R2 | HIGH | Prepare uses full-file median Δt; inference uses causal → eligibility mismatch. Stale parquet must not keep training on full-file flags. |
| R3 | HIGH | Replay UI tunes H/K on test while Freeze writes `source='validation_ui'`. Research/first-eval must not freeze H/K. |
| R4 | HIGH | Truncated history scored as `miss` too early. v1 coverage = last-admissible bound on `observation_end_s` only. |
| R5 | MEDIUM | Sensor-limit / prediction status dropped on `predictions.csv`; UI rescore must not default to 600. |
| R6 | SECOND | Run snapshot missing resolved task config; live YAML still used at evaluate; `--force` hash |
| R7 | OUT | Do **not** Full-train or rewrite quality numbers. Optional one-line note that old smoke hashes are not quality |

### Product constraints (non-negotiable)

- Splits stay bearings **9/3/3** and filters **40/10/50** (author test 50 held out). Do not retune splits to improve numbers.
- GRU or LSTM only; never mix bearings/filters checkpoints.
- No leakage: train-only scalers; whole-unit splits; predictor never sees future rows or reference RUL.
- `filters_full_history` stays disabled.
- Heavy jobs stay in `pdm.worker`. AppTest is the UI gate; browser playback is optional follow-up, not a gate.

---

## Phases

### Phase 1 — Train coverage + causal eligibility (R1, R2)

Fix what windows actually get gradients, and make offline/online window legality identical **even if this patch never re-prepares real data**.

1. **R1 sampler** — `set_epoch(epoch)` **inside** the 1-based epoch loop with `seed + epoch`. Do not iterate with default epoch 0 (`seed+0` ≠ epoch 1). Log `n_unique_sampled_windows` **per epoch**. Replacement sampling stays. Resume must not restart the schedule at epoch 0.
2. **R2 gaps** — prepare writes **causal** `gap_before`. Full-file median is diagnostic-only. **Belt-and-suspenders:** `build_windows` / `count_window_eligibility` recompute causal gaps from timestamps (same helper as inference) so stored full-file flags cannot drive train eligibility. **Gate train and eval** against expected constant `GAP_RULE_VERSION` (`"causal_v1"`): missing ≠ expected is incompatible (`fingerprint_mismatches` field-to-field equality is not enough). Plan fingerprint bump; **do not** run prepare on real datasets in this patch.

**Deliverables:** Subtasks 01–02  
**Risk:** Existing local processed dirs lack `gap_rule_version`. Train/eval must refuse them until the next human prepare. Data screen may still load. Tests use synthetic timestamps only.

### Phase 2 — Alert scoring contract (R4, R5)

Make miss/coverage and replay/CSV/UI rescore mean the same thing.

3. **R4 coverage (v1)** — observability through `event_time - minimum_action_lead_time` inclusive, scored from unit `observation_end_s` only. `H_trigger` is not the coverage bound. `min_lead==0` requires a timestamp **strictly before** the event. Interior-gap / K-sample coverage is a **documented caveat**, not this patch. Bump `METRICS_VERSION`. **Rewrite** `test_insufficient_coverage_not_false_miss` (keep `official_rul_overlay` as eval annotation, not a 600 Pa event).
4. **R5 sensor state** — persist H/K-independent observed fields; one rescore contract; thread `pressure_limit_pa` through `alerts_from_predictions`, `rescore_replay_alerts`, **and** the `app.py` UI call. Tests use 500 vs 550. Never a 600-only path in the UI.

**Deliverables:** Subtasks 03–04  
**Depends on:** Phase 1 preferred (tests file + replay/eval touch points). R4/R5 share `alerts.py` / `evaluate.py` / `replay.py` — sequential. R5 also edits `app.py` (pressure limit only; modes stay in 06).

### Phase 3 — Validation vs test policy workflow (R3)

Stop labeling test-tuned policies as validation. Freeze is the only writer of `source='validation_ui'` and the only creator of `alert_policy.json`. **Validation Evaluate** is `policy_mode='research'` (or write-identical `'validation'`): widget H/K allowed, `blind_benchmark: false`, **never** `ensure_alert_policy`. CLI `pdm evaluate` with no H/K flags is `frozen` (fail if no file). Test Evaluate with no freeze **fails** and must **not** `spawn_worker`. Default replay mode = **Validation**. Eval summary tables use `evaluate_mask.unit_ids`, not hardcoded `split['test']`.

**Deliverables:** Subtasks 05 (backend) → 06 (UI AppTest)  
**Depends on:** Phase 2 so rescore/export fields exist before UI modes rely on them.  
**07 (R6) depends on 05**, not 06 (`evaluate_run` API). 06 is UI-only relative to 07 except `app.py` pressure-limit source (07 reads snapshot when present).

### Phase 4 — Experiment snapshot (R6) + suite

Persist resolved config / preprocessing hash / commit / metric-policy version. Resume must not silently replace saved preprocessing. `--force` records **actual** checkpoint bytes. Distinguish data-incompatible vs evaluation-method-changed. Full pytest + ruff. AppTest fixtures use `METRICS_VERSION`, not hardcoded `"v0"`. Optional report note (no fake metrics).

**Deliverables:** Subtasks 07–08  
**Depends on:** 07 ← 05; 08 ← 01–07. Sequential 06 then 07 only if a merge conflict appears; prefer 07 after 05 while 06 is in flight or after 06 solely to serialize `evaluate.py` edits.

---

## Dependencies (DAG)

```text
01 (R1 sampler) ──┐
                  ├──→ 03 (R4 coverage) → 04 (R5 export + app.py limit)
02 (R2 causal)  ──┘         │
                            ↓
                     05 (R3 backend) ──→ 06 (R3 UI)
                            │
                            └──→ 07 (R6 snapshot) ──→ 08 (suite + R7 note)
                                       ↑
                                  06 does not block 07
```

01 and 02 do not share production files (`train.py` vs `filters.py`/`windows.py`/`prepare.py`) but **both edit** `tests/test_spec_invariants.py`. Run **01 then 02** under orchestration. Do not parallelize 03/04 or 05/06. **07 depends on 05**, not 06. If 06 and 07 would both touch `evaluate.py`, run 06 then 07; otherwise 07 may follow 05 immediately.

---

## Execution order

| # | Subtask file | Est. |
|---|--------------|------|
| 1 | `subtask-01-sampler-epoch-rng.md` | 1 h |
| 2 | `subtask-02-causal-gap-eligibility.md` | 1.5–2 h |
| 3 | `subtask-03-alert-coverage-horizon.md` | 1.5 h |
| 4 | `subtask-04-sensor-state-export.md` | 1.5–2 h |
| 5 | `subtask-05-validation-policy-backend.md` | 1.5–2 h |
| 6 | `subtask-06-replay-modes-ui.md` | 1.5–2 h |
| 7 | `subtask-07-experiment-snapshot.md` | 1–1.5 h |
| 8 | `subtask-08-regression-and-report-note.md` | 1 h |

**Total estimated effort:** ~11–15 hours (implementation + review cycles)

Orchestration serial order stays 01…08 so `evaluate.py` / `app.py` / `test_spec_invariants.py` do not fork. DAG permission: 07 may start after 05 if 06 has not started editing `evaluate.py`.

---

## Verification (every subtask)

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check src tests
```

| Area | Gate |
|------|------|
| Leakage / sampler / gaps / coverage / export / policy | `tests/test_spec_invariants.py` |
| UI modes, Freeze provenance, eval picker | `tests/test_worker_and_app.py` AppTest |
| Env (only if env-related) | `.venv/bin/python -m pdm doctor` |
| Browser | Optional follow-up on `http://127.0.0.1:8501` — **not** an acceptance gate |
| Smoke / Full train | **Not** a quality gate. Do not treat `--smoke` metrics as quality. Do not run Full 30-epoch as acceptance. |
| Real prepare | **Not** required this patch. Missing `gap_rule_version` **refuses train/eval** until the next human prepare. |

---

## Leakage & gotchas (carry through all subtasks)

- Fit encoder, imputer, scaler, `time_scale_s` on **train units only**; persist with the run.
- `unit_id`, `event_time_s`, `RUL`, split labels, official filter RUL are never model inputs. `official_rul_overlay` is eval annotation, not a 600 Pa event.
- Predictor receives **raw** rows ≤ t; evaluator joins GT afterward.
- `observed_limit_reached`, `differential_pressure`, `prediction_status`, `valid_history_reason` are **current** sensor/model state, not future ground truth. Allowed in `predictions.csv`.
- `alert_status` / warning_active remain H/K-dependent — **drop** from prediction export.
- `fingerprint_mismatches` equality of missing fields is **not** a version gate. Compare to expected `GAP_RULE_VERSION`.
- `ensure_alert_policy(overwrite=False)` is still a **write**. Research, Validation Evaluate, and CLI-with-flags must not call it. CLI with no flags is `frozen` (fail if missing), not a first-write.
- `_render_evaluation_panel` / `metrics_by_unit_display_frame` must use selected eval `evaluate_mask.unit_ids`, not always `split['test']`.
- Changing gap rules or coverage definition invalidates old processed data / old alert denominators. Version them (`gap_rule_version`, `METRICS_VERSION`). Do not silently reuse old smoke hashes as the new pipeline’s quality.
- Never load a bearings checkpoint into filters.
- `split_hash` remains canonical; loaders still accept legacy `split_fingerprint`.

---

## Out of scope

- R7 Full training, grouped CV, demo split, new NN architectures
- Enabling `filters_full_history` or inventing MATLAB fields
- Changing `time_to_seconds` or split counts
- FastAPI / React / Docker / MLflow / Transformers / live SCADA
- Rewriting `reports/implementation_report.md` with new quality numbers
- Browser-only verification as a gate
- R4 v1 interior-gap / K-sample coverage scoring (documented caveat)

---

## Next step

Re-run **plan-reviewer** on `.cursor/tasks/`, then `/orchestration` (planner → reviewer → per-subtask developer → code-reviewer).
