# Subtask 4: Persist sensor/model state for a single rescore contract (R5)

## Goal

In-memory `replay_unit`, exported `predictions.csv` evaluation, and UI `rescore_replay_alerts` produce the **same** episodes/statuses on the same prefix and policy. Thread `pressure_limit_pa` through the UI call so rescore does not default to 600. H/K changes must not change numeric `predicted_rul_s`.

## Context

Review R5 (MEDIUM). `replay_unit` knows Δp and sets Observed limit reached. `prediction_export_frame` drops `alert_status` and does not keep `observed_limit_reached` / Δp. `run_alert_evaluation` rescores from that CSV. UI rescore joins `differential_pressure` from measurements. Same H/K can open a horizon-warning on CSV that in-memory replay never opened.

**Plan-review Warning:** UI rescore still defaults to 600 unless `app.py` is in this subtask. Do not leave a 600-only path in the UI.

Counterexample: three consecutive `RUL<=H` with K=3; on the third measurement Δp exceeds the **saved** pressure limit. In-memory: sensor limit, no new horizon-warning. CSV without sensor state: may open horizon-warning. UI with Δp: matches in-memory.

These fields are current sensor/model state, **not** future ground truth. Allowed in the export. Threshold from saved config — **not** `float(rec["differential_pressure"]) > 600.0` in `alerts_from_predictions`.

## Acceptance Criteria

- [ ] `replay_unit` writes per-row: `observed_limit_reached`, `prediction_status` (Predictor status: Collecting history / ok / No valid prediction), `valid_history_reason`, and `differential_pressure` when present. Still computes `alert_status` in memory for the live engine, but that column is H/K-dependent.
- [ ] `prediction_export_frame` continues to drop `alert_status` (and any other H/K-dependent columns). It **keeps** the H/K-independent observed fields above. Changing H/K must not change exported `predicted_rul_s` or `observed_limit_reached`.
- [ ] `alerts_from_predictions` / `rescore_replay_alerts` / `run_alert_evaluation` share one contract: prefer `observed_limit_reached` if the column exists; else derive from `differential_pressure` using `pressure_limit_pa` passed in (run snapshot / dataset config), **never** a literal `600.0` in `alerts.py`.
- [ ] `rescore_replay_alerts(..., pressure_limit_pa=...)` exists. `src/pdm/app.py` passes `pressure_limit_pa` into that call (from bound/run config / `st.session_state` view — same source as the Δp plot line). No UI path that omits the arg and falls back to 600 only.
- [ ] UI rescore on a CSV **with** persisted fields matches CSV-only rescore and in-memory `replay_unit` episodes (same unit prefix, same policy).
- [ ] Tests use limit **500** and Δp **550**: 550 must trigger at 500 and must **not** trigger if a leftover 600 hardcoded path remains.

## Implementation Notes

**Files:** `src/pdm/replay.py` (`replay_unit`, `rescore_replay_alerts`; module-level `PRESSURE_LIMIT_PA` must not be the UI/rescore default when a run config exists), `src/pdm/evaluate.py` (`prediction_export_frame`, `_ALERT_DEPENDENT_PRED_COLUMNS`, `evaluate_run` replay loop, `run_alert_evaluation`), `src/pdm/alerts.py` (`alerts_from_predictions` ~238–242), **`src/pdm/app.py`** (`rescore_replay_alerts(...)` call ~857; `pressure_limit_pa` already on the replay view ~913)

- Thread `pressure_limit_pa` into `alerts_from_predictions(..., pressure_limit_pa=...)` and `rescore_replay_alerts`.
- `collecting` for AlertEngine should follow `prediction_status == "Collecting history"` (or missing finite RUL), not the dropped `alert_status` column.
- Do not put actual RUL / event_time into the export as model inputs. `actual_rul_s` may already be evaluator-joined; do not use it in `AlertEngine`.
- `evaluate_run` currently passes `pressure_limit_pa=float(cfg.get("pressure_limit_pa", 600.0))` from **live** YAML — acceptable until subtask 07 snapshots it; still pass the **same** value into export, rescore, **and** the app call in this subtask.
- Replay **modes** (Validation/Test/Research) stay in subtask 06. This subtask only threads the limit into the existing rescore call.

**Tests** in `tests/test_spec_invariants.py`:

- `test_sensor_limit_rescore_matches_in_memory_and_csv` — K=3, three RUL≤H, third Δp=550, limit=500; compare episodes from (1) `replay_unit`, (2) `prediction_export_frame` → `alerts_from_predictions` without measurements, (3) `rescore_replay_alerts` with measurements **and** `pressure_limit_pa=500`.
- `test_hk_change_does_not_change_predicted_rul_or_observed_limit` on the export frame.
- `test_pressure_limit_500_not_hardcoded_600` — Δp=550, limit=500 → observed limit; if the code still used 600, this would fail.

AppTest for modes is subtask 06. If an existing AppTest already rescored filters, it must keep working with an explicit limit. No split changes.

## Dependencies

Subtask 03 (same `alerts.py` / `evaluate.py`; coverage tests already landed).

## Verification

```bash
.venv/bin/python -m pytest tests/test_spec_invariants.py -q -k "sensor_limit or observed_limit or prediction_export or rescore or pressure_limit"
.venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check src tests
```
