# Quality, reports and model comparison

## Data admission

`prepare` saves an immutable data version with `quality_records.parquet`, `quality_report.json`, and the admission policy version/hash in `processed_fingerprint.json`. Experiments retain that fingerprint and their split. Replay resolves the experiment's saved version, even when the current data version changes.

The admission policy is `admission_v1` in `pdm.data.quality`:

| Check | Action |
|---|---|
| Unreadable vibration record, wrong channel count or fragment length, missing/nonfinite required numeric values | Exclude the record; preserve its source and reason in the audit |
| Identical unit/time measurements | Keep one measurement |
| Conflicting measurements at the same unit/time | Exclude all conflicting records |
| Rejected bearing measurement inside a series | Preserve a history boundary; windows and continuous state restart after it |
| Damaged bearing endpoint or unreliable filter event/censoring clock | Exclude the series from training and RUL evaluation; never move its event to the last retained row |
| Backward filter source timestamps | Quarantine the affected series before sorting can hide the error |
| Missing equipment identifiers | Block preparation with source row numbers; never silently lose records through groupby |
| Extreme signals or jumps | Diagnostic attention only; retain potential degradation evidence |
| Valid filter series without observed failure | Retain as right-censored survival observations |

The original train/validation/test assignment is filtered, never redistributed after exclusion. Training validates file hashes, current admission policy and nonempty eligible train and validation windows before fitting any preprocessing or model. Full CNS uses the same admission gate.

On the real source files prepared on 2026-09-15:

- Bearings: 9,216 admitted measurements, 15 series, no rejected records; split 9/3/3.
- Filters: 78,236 admitted measurements, 99 series; split 39/10/50. `Train_28` (598 records) is excluded because its first source time is 157.2, followed by 0.2. Four observed train failures and one validation failure remain. Large signals marked for attention remain available.

Bearing `last_recorded_sample` is an experiment-end approximation, not a verified industrial failure time. Filter times retain the project's unverified `Time × 60` convention.

A censored series establishes that the equipment continued to operate for its remaining observed duration. The existing Weibull survival loss uses this lower bound without inventing a failure time. See the [scikit-survival introduction](https://scikit-survival.readthedocs.io/en/stable/user_guide/00-introduction.html).

## One forecast, architecture-specific diagnostics

The ordinary `Predictor` and traced forecasts use `model_forecast` and the saved preprocessing/window rules. Results expose timestamp, point RUL, readiness/status, memory mode and optional interval provenance. Exports retain `raw_rul_s` separately from the displayed `predicted_rul_s`.

- GRU: actual normalized inputs and hidden state of every recurrent layer.
- LSTM: the same, plus actual cell memory `c` separately from hidden state `h`.
- Random reservoir: its actual computational graph and states.
- Fly reservoir: actual sampled source topology; coordinate completeness is labelled.
- Full CNS: every classified neuron computes; display samples, anatomical coordinates and readout pooling retain their existing explicit labels.

Window models reset to their saved history window. Continuous models append chronological state and reset at gaps. Seeking reconstructs the same causal prefix. Ground truth is joined only after inference; the display checkbox cannot change forecasts. A replay chart includes all eligible earlier forecasts regardless of the user's seek path.

Weibull 5–95% ranges describe the fitted distribution, without empirical calibration guarantees. An empirical interval is shown only after verifying its checkpoint, preprocessing, graph, data and split bindings. Results on calibration units are marked explicitly. Point predictions remain usable when no interval exists.

## Comparison protocol

Each selection names one `run_id` and one `eval_id`. There is no fallback to an unrelated latest metric. Saved prediction-file hashes are verified. Data version, features, units, split, quality policy/audit and metrics version must agree.

The comparison clock excludes only explicit unavailable history and invalid outcome points. A missing prediction or missing CSV row reduces coverage; it cannot remove a difficult point from the other models' scores. All evaluated units must be represented. Incomplete, smoke, synthetic and legacy-quality runs cannot win the current ranking. Historical/diagnostic runs have a separate opt-in selector.

| Dataset / split | Primary score |
|---|---|
| Bearings / validation or test | Mean of each bearing's MAE over `0 < actual RUL ≤ 1,800 s` |
| Filters / validation | Mean of each unit's mean survival NLL across all eligible observed or censored windows |
| Filters / test | Unit-equal MAE against official RUL at the last recorded prefix point |

Validation NLL uses Weibull density in the same fixed internal-second unit for every run, including different train time scales in the censoring study. Thus it may differ numerically from the training history's normalized-duration NLL. Zero remaining observation duration has no positive-duration likelihood and is excluded explicitly.

Other columns include full-history MAE, overestimation, interval width/coverage, observed failure counts, and warning outcomes. Warning columns describe saved full-history evaluations; they require identical thresholds to be compared. Node/state size, memory mode and history length are shown beside results. A 1,000-node fly/random comparison controls topology; a 166,700-node continuous model also changes size and memory.

Model selection uses validation. Test has already been inspected in prior project work and is labelled `exploratory_reused_holdout`. Alert settings are fixed by the existing train-duration rule: horizon = rounded 10% of median train duration, three confirmations, minimum lead = half the horizon. They are frozen before the test alert evaluation.

`runs/comparisons/<id>/` stores selections and all tables in `comparison.json`, `table.csv`, `per_unit.csv`, and `predictions.csv`. UI downloads also include prediction JSON. Saved compositions can be restored without changing selected evaluation identities.

## Sequential full study

`pdm train-matrix` runs:

- Bearings: GRU, LSTM, real fly reservoir 1,000, matched random reservoir 1,000, complete MaleCNS.
- Filters: GRU, LSTM, real fly reservoir 1,000, matched random reservoir 1,000.
- Filter GRU ablation: all valid training series versus observed failures only, seeds 42/43/44. The full validation cohort is identical in each pair. Main GRU seed 42 supplies the first reference, so there are 14 distinct experiments.

All use history 20, all eligible windows, real sources and seed 42 except the paired seeds. Gradient training uses up to 30 epochs with early stopping. Full CNS preserves its grouped train-only readout CV and separate validation calibration.

Full CNS also preserves the previous readout protocol's terminal zero-RUL state: 6,861 train states and 1,622 calibration states, versus 6,852 and 1,619 strictly pre-endpoint windows for the window models. The common ranking excludes terminal zero-RUL points for every bearing model. Gradient training retains the existing unit-balanced sampler with replacement: all eligible windows enter the sampling pool; each epoch reports its actual unique draws.

`runs/batches/<batch_id>/manifest.json` records data versions, task statuses, run IDs and exact validation/test evaluation IDs. `batch.log` records progress. Stop is checked at training boundaries and during evaluation. Resuming keeps completed jobs; interrupted fitting without a saved completed model restarts that configuration. Changed data versions are rejected.

`censoring_ablation.csv` reports validation NLL, observed-event MAE and independent failure counts. With only one observed validation failure, conclusions must remain limited to these GRU configurations and this split.

Additional ablation configurations are evaluated on validation only. The main seed-42 GRU already has its ordinary main-study test evaluation; that test result is not used in the paired censoring conclusion.

## Checks

Run `.venv/bin/python -m pytest tests -q` and `.venv/bin/python -m ruff check src tests`. Tests cover damaged endpoints, gaps, censoring, duplicates, compatible scoring masks, missing predictions, real recurrent/cell traces, trace/ordinary forecast equality, seek/replay/export equality and the existing no-future/no-ground-truth invariants. Browser acceptance additionally exercises both datasets, every architecture, desktop/mobile layouts, controls, exports, saved selections and console errors.

The [completed 2026-09-15 study](../reports/quality_study_20260915.md) records 14 real experiments, four complete comparisons and the paired censoring results. `scripts/verify_quality_study.py <batch_id>` checks real checkpoint/replay/trace parity and every saved comparison CSV/JSON. Run it while the worker is idle.

Official filter test truth is an evaluation overlay anchored to `observation_end_s + official_rul_at_prefix_end_s`, never an observed prefix failure or model input. The study preserves its first four filter test evaluations and publishes corrected truth annotations as new evaluation IDs, with unchanged inference and explicit provenance. `scripts/backfill_official_test_truth.py <batch_id>` reproduces that annotation-only migration when needed.
