# Condition monitoring v1

Condition monitoring extends the four existing screens and uses the existing worker,
training-v2 engine, immutable admitted data and `Predictor`. Old runs are read without
migration. `base_v1` and `degradation_v1` retain their original numerical definitions.

## Runtime and identity

`monitoring_step` in `src/pdm/monitoring/runtime.py` is the common computation for
sequential replay, direct prefix seek and exported observations. It removes all outcome
columns and rows later than `as_of`. Quality uses virtual replay time. No routine trains
or adapts reference data during inference. A prefix contains one equipment unit.

Quality, normality, model applicability, forecast readiness, health, urgency and the open
critical latch are separate fields. Invalid required channels are never filled with zero
or a learned average. Sensor bounds, fragment integrity, duplicate/backward clocks,
freshness, contiguous history and constant-signal suspicion have explicit reasons.
An unusual but finite amplitude alone is not a rejection. Fragment audit is also integrated
into future bearing preparations; existing snapshots are not rewritten.

A robust reference uses fixed initial 20-point train prefixes meeting a saved stability
rule. This is **candidate_healthy / provisional**, never expert-labelled health. Filter
zero-flow or zero-pressure startup prefixes cannot establish a healthy reference.
The regime domain is saved separately from reference medians, using train operating
conditions; missing reference can make condition unavailable while a trained sensor
forecast remains possible. RPM/load exact groups and laboratory flow/feed bins are
explicit and provisional. No age-dependent healthy norm or test adaptation is used.

## State and action

`state.py` implements channel-valid applicable limit → unavailable quality → unknown
regime/reference → eligible prognostic urgency → persistent deviation → normal.
Confirmation uses both counts and elapsed source time; recovery uses its own lower
threshold and longer interval. Gaps reset confirmation, never erase the open alert.
Hard limits operate before warmup and outside the ML domain. An acknowledgement is
append-only feedback, not a repair, resolution or training label.

Critical limits require signal, unit, direction, aggregation rule, applicability, provenance
and verification. Filters retain a **laboratory** 600 Pa definition. Bearing RMS has no
configured physical critical limit. This is allowed to produce no red bearing episodes.

Planning margin is `named conservative time - (action lead + buffer)`, only with compatible
verified time, endpoint, quality, domain and a permitted validated method. A wide lower
interval near zero cannot by itself authorize predicted red. Current bundles have no
verified maintenance lead time and no admitted timing/probability calibration: those
outputs remain unavailable. Rule behavior is tested with labelled synthetic fixtures.

## Commands

```sh
.venv/bin/python -m pdm condition-study --config configs/condition_bearings.yaml --plan
.venv/bin/python -m pdm condition-study --config configs/condition_bearings.yaml --run
.venv/bin/python -m pdm condition-study --resume-study STUDY_ID
.venv/bin/python -m pdm stop
.venv/bin/python -m pdm freeze-monitoring-bundle --study-id STUDY_ID --candidate-id event_final
.venv/bin/python -m pdm monitor-evaluate --bundle-id BUNDLE_ID --split validation
.venv/bin/python -m pdm monitor-evaluate --bundle-id BUNDLE_ID --split test
```

Profiles plan 22 fitting jobs each (44 paired). The ledger discloses every horizon/quantile
fit, event fold and seed. A maximum of 24 attempts per profile leaves two slots for recorded
interrupted sensor retries, staying within the combined 48 cap. Completed jobs are reused;
optimizer/scheduler/RNG resume remains training-v2. Test never selects a candidate.

`monitoring_bundle.json` binds data, model files, preprocessing, history, reference, policies,
units, calibration and source identity. Dependency edits are rejected, not silently loaded.
A policy change requires a new bundle. Evaluation files have independent hashes and IDs.
Feedback is intentionally append-only and is not part of immutable issued-forecast bytes.

## UI and interpretation

With a frozen bundle, Model Report starts with Condition & Forecast. Legacy installations
without a bundle retain their original entry point. The plot shows actual sensor forecasts,
real units, reference/statistical boundary and any configured physical limit. Issued forecasts
retain `(issued_at, target_time)`; older lines are optional. Ground truth is only an overlay.
RUL and research Weibull percentages live in Diagnostics. Existing model activity remains
available, and no connectome is loaded for the main monitoring view.

The comparison screen separates event forecasts, sensor forecasts and monitoring outcomes.
Diagnostics without a declared event/horizon are not scored as false failure-in-H alarms.
All measurements, including refusals and warmup, remain in the end-to-end clock.

This is a laboratory support tool, not equipment protection or an operating procedure.

## Recorded bounded study

The executed paired studies consumed 48 fits: 44 original jobs, two median residual-score controls and two corrected multiscale recipe fits on bearings. Four faulty multiscale v1 fits remain quarantined and counted. Additional corrected filter feature/residual ablations are blocked by the combined budget. This is recorded in `output/condition-monitoring/combined_fit_ledger.json`; no claim is made for missing results. A new plan uses corrected v2 recipes.
