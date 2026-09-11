# Master Plan: Next Sprint TZ (Predictive Maintenance Lab)

**Source of truth:** `Predictive_Maintenance_Lab_Review_Next_Sprint.md`  
**Repo:** `src/pdm/` · **Stack:** Python ≥3.11 · PyTorch · Streamlit · pytest · ruff  
**Date:** 2026-09-10 (revised after plan-review)

## Overview

This sprint closes the train/inference gap, fixes filter survival labels, freezes data/split identity for evaluation, unifies gap rules at inference, and delivers transparent UI + immutable evaluation artifacts. Existing split protocols are preserved (bearings 9/3/3; filters author test 50 held out; author train 40/10). No new NN architectures (GRU/LSTM only). Full 30-epoch training runs are **out of scope** as a verification gate; UI must support Full mode.

### In scope (P0 + P1 + Section 6 + Section 7 + Section 9 tests)

| ID | Theme |
|----|-------|
| P0-1 | Unified preprocessing (train + inference) |
| P0-2 | Correct survival duration for filters; bearings endpoint mask alignment |
| P0-3 | Evaluate compatibility guards (dataset/split fingerprints) |
| P0-4 | Unified gap/quality rules at inference |
| P0-5 | Filter time scale warning (keep `time_to_seconds=60`) |
| P1-1 | Immutable evaluation layout (`evaluations/<eval_id>/`) |
| P1-2 | Replay UI fixes (H/K cache, log filter, fragment, precomputed predictions, state keys) |
| §6 | Data / Train / Evaluation / Demo presentation |
| §7 | Alert horizon: `H_trigger` vs `minimum_action_lead_time` |
| §9 | 15 named regression tests |

### Out of scope

- Full GRU/LSTM training on real data as acceptance gate
- Grouped 4-fold CV, `filters_full_history` enablement (inspect only), extra models, anomaly detection
- FastAPI/React/Docker/MLflow/Transformers/OpenAI/live SCADA
- Changing bearings 9/3/3 or moving author filter test into train
- Changing `time_to_seconds` away from 60
- Committing `data/`, `runs/`, `.venv/`

---

## Phases

### Phase 1 — P0 foundation (data correctness)

Fix the pipeline before any new metrics or UI polish.

1. **Unified raw→model feature path** — one serializable pipeline used by `fit_preprocessor`, `UnitWindowDataset`, and `Predictor`.
2. **Survival window labels** — correct `duration_s` / `event` / `target_rul_s`; bearings endpoint mask in train/replay/evaluate.
3. **Gap/quality at inference** — same `gap_before` logic as `build_windows`; prefix-only gap estimation; warmup + alert reset.
4. **Frozen versions** — `prepare.py` emits versioned fingerprints; train saves `dataset_fingerprint.json`; evaluate/replay abort on mismatch and bind to run snapshot.
5. **Filter time scale** — visible UI warning; do not relabel units.

**Deliverables:** Subtasks 01–10  
**Risk:** Dust one-hot parity is the highest-leakage-adjacent bug; fix before retraining.

### Phase 2 — P0 regression tests

Lock fixes with synthetic fixtures; split into preprocessing/survival (11a) and gap/causality (11b).

**Deliverables:** Subtasks 11a, 11b  
**Depends on:** Phase 1

### Phase 3 — Transparent counts & Full training mode

Data screen reads cached counts from `data_report.json`. Train screen: Smoke/Full, selection metric, best-epoch marker.

**Deliverables:** Subtasks 12–13  
**Depends on:** Phase 1; **soft:** Phase 2 (11a, 11b) complete first

### Phase 4 — Immutable evaluation + metrics + policy

**Order matters:** layout (14) → alert policy (17) → RUL metrics (15) + alert metrics (15b) → UI tables (16).

- Prediction artifacts independent of H/K.
- Alert policy (`H_trigger`, `minimum_action_lead_time`) defined before alert metrics.
- Bearings near-event zones frozen in config.

**Deliverables:** Subtasks 14, 17, 15, 15b, 16  
**Depends on:** Phases 1–2

### Phase 5 — Replay UI & demo presentation

Precomputed `evaluations/<eval_id>/predictions.csv` for replay; block Play without artifacts. No unbounded inline inference on Streamlit thread.

**Deliverables:** Subtasks 18–19  
**Depends on:** Subtask 18 required for 19; subtasks 14, 17 (required for 18); Phase 1 gap rules; **soft:** 16 for eval tables in replay/demo UI

---

## Dependencies (DAG)

```text
01 → 02 → 11a (preprocessing / survival tests)
03 → 11a
04 → 05 → 11b (gap / causality tests)
06 → 07 → 08 (compat gate: test_evaluate_rejects_changed_dataset_or_split)
09, 10 (parallel)
12, 13 (after 03, 06; **soft:** 11a, 11b before Phase 3 UI)

14 → 15 (RUL / baseline metrics; fills stub metrics from 14)
14 → 17 (alert policy → `runs/<run_id>/alert_policy.json`)
17 → 15b (alert metrics — requires frozen policy from 17)
15 + 15b → 16
14 + 17 → 18 (core replay; **soft:** 16 for eval tables on replay screen)
18 → 19 (required) → 20

11a, 11b → before Phase 3 UI (soft gate for 12)
08: hard gate `test_evaluate_rejects_changed_dataset_or_split`
15b: owns alert episode tests; 20 = full-suite only
```

---

## Execution order

| # | Subtask file | Est. |
|---|--------------|------|
| 1 | `subtask-01-raw-feature-pipeline.md` | 1–1.5 h |
| 2 | `subtask-02-preprocessor-imputation-scaling.md` | 1 h |
| 3 | `subtask-03-survival-window-labels.md` | 1–1.5 h |
| 4 | `subtask-04-gap-quality-inference.md` | 1–1.5 h |
| 5 | `subtask-05-filter-gap-causal-prefix.md` | 45 min |
| 6 | `subtask-06-dataset-fingerprint-versioning.md` | 1–1.5 h |
| 7 | `subtask-07-split-integrity-origin-ids.md` | 45 min |
| 8 | `subtask-08-evaluate-compatibility-guards.md` | 1–1.5 h |
| 9 | `subtask-09-filter-time-scale-warning.md` | 45 min |
| 10 | `subtask-10-filters-full-history-inspect.md` | 30 min |
| 11a | `subtask-11a-regression-tests-preprocessing-survival.md` | 45–60 min |
| 11b | `subtask-11b-regression-tests-gap-causality.md` | 45–60 min |
| 12 | `subtask-12-data-screen-transparency.md` | 1–1.5 h |
| 13 | `subtask-13-train-screen-full-mode.md` | 1 h |
| 14 | `subtask-14-immutable-evaluation-layout.md` | 1.5–2 h |
| 17 | `subtask-17-alert-horizon-policy.md` | 1–1.5 h |
| 15 | `subtask-15-evaluation-rul-baseline-metrics.md` | 1–1.5 h |
| 15b | `subtask-15b-evaluation-alert-metrics.md` | 1–1.5 h |
| 16 | `subtask-16-evaluation-ui-tables.md` | 1 h |
| 18 | `subtask-18-replay-ui-fixes.md` | 1.5–2 h |
| 19 | `subtask-19-demo-replay-presentation.md` | 1 h |
| 20 | `subtask-20-regression-tests-eval-replay.md` | 2–2.5 h |

**Note:** Subtasks 15 and 17 both depend on 14; **17 must complete before 15b**. Subtask 15 (RUL) may run in parallel with 17 after 14.

**Total estimated effort:** ~24–32 hours (implementation + review cycles)

---

## Verification (every subtask)

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check src tests
```

Additional checks by area:

| Area | Command / check |
|------|-----------------|
| Env | `.venv/bin/python -m pdm doctor` |
| Prepare | `.venv/bin/python -m pdm prepare --dataset bearings` (if data present) |
| Smoke train | `.venv/bin/python -m pdm train --dataset bearings --smoke` |
| UI | `tests/test_worker_and_app.py` AppTest; manual `http://127.0.0.1:8501` optional |
| Parity | `test_offline_online_feature_parity` tolerance 1e-6 |
| P0-3 gate | `test_evaluate_rejects_changed_dataset_or_split` in subtask 08 only |

---

## Leakage & gotchas (carry through all subtasks)

- Fit encoder, imputer, scaler, `time_scale_s` on **train units only**; persist with run.
- `unit_id`, `event_time_s`, `RUL`, split labels are never model inputs.
- Censored filter rows: `event=0`, `target_rul_s=None`; do not MSE on `observation_end - t`.
- Post-event windows must not enter training or survival loss.
- Bearings endpoint: `actual_rul_s` NaN at `t >= event_time_s`; never in MAE.
- Observed filter events: same post-event NaN rule at `t >= event_time_s`; censored filters have no point RUL.
- Evaluate/replay: `test_ids` from `rdir/split.json`; never live split when run snapshot exists.
- Predictor receives **raw** measurement rows ≤ t; evaluator joins GT afterward.
- Heavy inference off Streamlit request thread; prefer precomputed eval predictions.
- Separate checkpoints per dataset; never load bearings weights into filters.
- `split_hash`: single canonical name; loaders accept legacy `split_fingerprint` on old runs.
- Smoke runs are not quality benchmarks.

---

## Next step

Re-run **plan-reviewer** on `.cursor/tasks/`, then `/orchestration` or the dev-review workflow skill for implementation.
