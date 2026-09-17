# Condition monitoring implementation — 16–17 September 2026

## Scope and evidence boundaries

Implemented in the existing Predictive Maintenance Lab; training protocol v2, the four existing screens, original architectures, immutable dataset snapshots and common Predictor are retained. Condition & Forecast is the main report when a monitoring bundle exists. Fly / neural activity remains an additional experimental view.

This is a laboratory implementation. No industrial fault probability, safe operating deadline, verified healthy label or unavailable MATLAB result is asserted. Synthetic tests establish software behavior; real XJTU-SY/HSE experiments establish only the recorded empirical results. Test is the already-used exploratory holdout and never selects a candidate, recipe, horizon, threshold or policy.

Initial HEAD: `3143fabf3a11928b915f24b03bafe89bd10a6e1c` (specification named older `e026c0d`). Baseline: 369 tests passed before implementation. Initial untracked `.playwright-cli/` and `output/` were preserved. A concurrent Windows task changed `README.md`, `scripts/run.ps1`, `scripts/setup.ps1`, Windows process checks in `src/pdm/worker.py`, and `tests/test_worker_and_app.py`; those changes are not claimed as this implementation and were preserved.

## Files

| Area | Exact repository files |
|---|---|
| Memory and common inference | `src/pdm/history.py`, `src/pdm/models/recurrent.py`, `src/pdm/preprocessing.py`, `src/pdm/train.py`, `src/pdm/training_engine.py`, `src/pdm/training_readout.py`, `src/pdm/predict.py` |
| Versioned features / fragment quality | `src/pdm/multiscale_features.py`, `src/pdm/feature_recipes.py`, `src/pdm/training_protocol.py`, `src/pdm/data/bearings.py`, `src/pdm/data/quality.py` |
| Observation/reference/forecast | `src/pdm/monitoring/contracts.py`, `quality.py`, `normality.py`, `signal_forecast.py`, `runtime.py`, `calibration.py` |
| Decisions, identity, evaluation | `src/pdm/monitoring/policy.py`, `state.py`, `journal.py`, `bundle.py`, `evaluation.py`, `filter_export.py`, `study.py` |
| Existing UI / worker / CLI | `src/pdm/monitoring/ui.py`, `src/pdm/app.py`, `src/pdm/lab_ui.py`, `src/pdm/cli.py`, condition handlers in `src/pdm/worker.py` |
| Profiles | `configs/condition_bearings.yaml`, `configs/condition_filters.yaml` |
| Reproducible diagnostics | `scripts/condition_history_report.py`, `scripts/verify_condition_runtime.py`, `scripts/condition_residual_ablation.py`, `scripts/condition_policy_audit.py`, `scripts/condition_recipe_repair_study.py`, `scripts/condition_outcome_report.py`, `scripts/export_filter_tables.m` |
| Acceptance tests | `tests/test_condition_monitoring.py` (old tests were not weakened) |
| Documentation | `docs/condition_monitoring_implementation_plan.md`, `docs/condition_monitoring.md`, `docs/sensor_forecasting.md`, `docs/history_20_60_protocol.md`, `docs/data_units_and_endpoints.md`, this report |

## Implemented behavior

Runtime strips outcomes and future rows, checks raw channels before inference, refuses invalid/stale clocks and channels, and treats past invalid observations, acquisition gaps and confirmed maintenance markers as history boundaries. An applicable usable hard-limit channel bypasses model warmup/OOD; loss of observations preserves an open critical latch. Strong physically admissible vibration alone is not deleted.

Train-only initial stable prefixes establish a separate robust median/MAD reference by operating regime. It is explicitly provisional. Unknown regime and absent healthy reference are distinct from sensor quality and event-model readiness. No reference adaptation occurs during replay.

Fixed 20/40/60 and genuinely trained variable 20–60 use saved policies. Recurrent lengths mask padding; variable training samples different valid lengths throughout life, rather than using only short early-life windows. Full CNS keeps its continuous-state contract. The runtime reports actual elapsed history, not 20 points = 20 elapsed intervals.

Persistence, causal local trend and quantile boosting forecast actual future sensor labels. Horizon masks reject missing tails, gaps, repairs and ambiguous time matches. Quantile ordering/nonnegative transforms are identical at evaluation and inference. No line is derived from RUL, extended until a threshold is hit, or forced to increase. Future measured operating context never enters the inputs.

The deterministic state engine has confirmation, duration, recovery, hysteresis, cooldown, idempotent episodes, escalation and a critical latch. Feedback is append-only; acknowledgement is not fault confirmation or resolution. Action timing checks units, endpoint, method, calibration, horizon and uncertainty. Present real bundles intentionally cannot issue validated operational timing.

Frozen bundles bind preprocessing, history, reference, quality/state policies, model and dataset hashes, calibration, source commit/worktree hash and environment. Evaluation retains every clock row, including refusals; JSON/parquet/prefix/trace checks exercise the same runtime. Prior evaluation files are never overwritten.

## Engineering issues caught during verification

- Fixed histories originally omitted explicit maintenance boundaries; behavior was corrected and tested for all four modes.
- Historical sensor-range/fragment errors now reset warmup; they cannot be filled into a supposedly valid history.
- Duration confirmation advances with measurement time, not repeated redraws of the same observation.
- JSON reload of active episodes now preserves resolution/escalation identity.
- Early `multiscale_*_v1` experiments exposed pandas label alignment producing missing slopes on later objects/segments. Their artifacts remain, but they are quarantined (`quality_eligible=false`) and are **not evidence that multiscale features are worse**. Corrected recipes use explicit positional assignment and version `v2`; v1 retains its exact saved semantics for forensic replay. `base_v1` and `degradation_v1` are untouched. Replacement diagnostics are separately counted.
- Browser tests exposed playback/timer handling; Start/Pause recreate the fragment timer and paused views do not repeatedly rerun.

## Data and limits

Bearings snapshot `20260915T153526Z_04154804`: 9,216 rows, 15 bearings, original 9/3/3 split. Last-recorded-fragment endpoint remains a proxy, not a confirmed industrial failure. No engineering RMS critical limit is invented.

Filters snapshot `20260915T153833Z_4182d92b`: 78,236 rows, 99 objects, 39/10/50 original split. Existing admission excludes the backward-clock training object `Train_28`. HSE source PDF confirms differential pressure in Pa, the instrument range and laboratory 600 Pa threshold. It does **not** establish Time/RUL physical units. Runtime retains `dataset_internal`, and physical action deadlines stay unavailable.

`filters_full_history_v1` remains disabled: MATLAB verified table export, origin/prefix matching and endpoint/unit verification are unavailable. The export helper and validator do not fabricate a usable full-history dataset. Actual status is in `output/condition-monitoring/filter-export-status.json`.

A real completed 32,768-sample train fragment at 25,600 Hz was processed by the envelope helper (`output/condition-monitoring/envelope-real-fragment.json`). DC/window/Hilbert/energy conventions are recorded. Bearing defect frequencies remain unavailable without verified geometry; no envelope model was promoted or full raw snapshot rewritten.

Statistical thresholds are provisional train-prefix quantiles. They are not calibrated to a verified industrial false-episode rate: healthy negative follow-up and an operational target are missing. Grouped train audits record that limitation rather than treating candidate-healthy labels as truth. Calibration remains `insufficient_independent_calibration_data`; repeated seeds do not increase independent event counts.

## Selection, bounded fits and incomplete experiments

The main studies are `20260916T170147Z_bfd686` (bearings) and `20260916T171150Z_9177ed` (filters). Their train equipment folds select history/recipe and sensor method. A second train fold is a confirmation diagnostic. The final event fit uses validation for checkpoint stopping; consequently validation is **not independent calibration**. Seeds 42/43/44 reuse the same objects and do not create independent failure evidence. No test value changes selection.

Exactly **48 fitting jobs** were executed: 44 original event/sensor jobs, two bearing median-quantile residual-score controls, and two corrected bearing multiscale recipe fits. Horizon × quantile submodels count separately. The four faulty v1 multiscale fits are retained, quarantined and counted. Corrected filter multiscale/no-age refits and filter residual-score ablation are **blocked by the exhausted fit budget**. They are not reported as successful experiments. Full CNS refitting, additional long horizons and an envelope-model ablation were not run.

The fit ledger is `output/condition-monitoring/combined_fit_ledger.json`. Supplemental exact records are in each study's `recipe_repair_ledger.json` and, where present, `residual_ablation/fit_ledger.json`. A fresh condition study plan uses corrected v2 recipes. Existing runs and the earlier training-v2 study were not overwritten.

Main train-fold history scores (lower is better):

| History | Bearings near-last-30-minute MAE, physical seconds | Filters unit-balanced survival NLL, common-clock-60 |
|---|---:|---:|
| fixed_20 | 2570.7980 | 2.858017 |
| fixed_40 | 2640.1630 | 3.272798 |
| fixed_60 | 2571.9299 | **2.569134** |
| variable_20_60 | **2550.3519** | 2.584734 |

These are small differences, not proof that variable/longer memory generally improves prediction. On the same bearing train fold, selected-recipe seeds 42/43/44 scored 2550.3519 / 3005.1373 / 2700.8337 seconds. The corrected multiscale v2 scored 2614.8966 seconds; its no-age version scored 2866.9788 seconds. Both are worse than the selected base recipe on this fold. The defective v1 scores are deliberately excluded from that conclusion.

For bearings, the additional residual-score median-only control improved the 300-second horizon (0.130654 vs 0.146120 g MAE) and worsened 900 seconds (0.466162 vs 0.387666 g). Both controls use the same nonnegative, unrepaired median transform; this is not a re-ranking of complete quantile curves or permission to change the frozen sensor. Filter seeds scored 2.569134 / 2.316600 / 2.624232 NLL. No post-test candidate was promoted.

`history_per_unit_results.csv` and `history_comparison.csv` in both study directories report common-clock-60 and end-to-end availability. A horizon of 1800 in those generic files is **1800 saved time units**; it means 30 physical minutes only for bearings. The original filter Time conversion does not justify a physical-minute label.

## Exact commands and observed checks

All commands ran from the repository root using its `.venv`. System `/usr/bin/git` requested an Xcode license, so Git-aware commands used `PATH=/Library/Developer/CommandLineTools/usr/bin:$PATH`.

```sh
PATH=/Library/Developer/CommandLineTools/usr/bin:$PATH .venv/bin/python -m pytest tests -q
.venv/bin/python -m pdm doctor
.venv/bin/python -m pdm condition-study --config configs/condition_bearings.yaml --plan
.venv/bin/python -m pdm condition-study --config configs/condition_bearings.yaml --run
.venv/bin/python -m pdm condition-study --config configs/condition_filters.yaml --run
.venv/bin/python -m pdm stop
.venv/bin/python -m pdm condition-study --resume-study 20260916T171150Z_9177ed
.venv/bin/python scripts/condition_history_report.py runs/condition_studies/20260916T170147Z_bfd686
.venv/bin/python scripts/condition_history_report.py runs/condition_studies/20260916T171150Z_9177ed
.venv/bin/python scripts/condition_residual_ablation.py runs/condition_studies/20260916T170147Z_bfd686
.venv/bin/python scripts/condition_recipe_repair_study.py runs/condition_studies/20260916T170147Z_bfd686
.venv/bin/python scripts/condition_policy_audit.py runs/condition_studies/20260916T170147Z_bfd686
.venv/bin/python scripts/condition_policy_audit.py runs/condition_studies/20260916T171150Z_9177ed
.venv/bin/python -m pdm freeze-monitoring-bundle --study-id 20260916T170147Z_bfd686 --candidate-id event_final
.venv/bin/python -m pdm freeze-monitoring-bundle --study-id 20260916T171150Z_9177ed --candidate-id event_final
PATH=/Library/Developer/CommandLineTools/usr/bin:$PATH .venv/bin/python -m pytest tests -q --junitxml=output/condition-monitoring/junit.xml
.venv/bin/ruff check src tests scripts/condition_*.py scripts/verify_condition_runtime.py
PATH=/Library/Developer/CommandLineTools/usr/bin:$PATH git diff --check
```

The final full suite passed **441 tests, zero failures/errors/skips**, in 49.142 seconds. The two warnings concern PyTorch sparse CSR beta support and a legacy test's non-writable NumPy conversion. They are recorded, not suppressed. Ruff and whitespace checks passed. The original baseline was 369 passing tests; concurrent Windows work contributes additional tests, so the entire difference is not claimed as new monitoring tests.

`output/condition-monitoring/junit.xml`, `full-tests.log`, `test-results.json` and `ruff.log` contain the actual evidence. The monitor worker command was exercised on the earlier frozen filter validation bundle (`13d2a9e9f6330ef332f16069`, 5763 measurements, 158.561 seconds). Final repeat evaluations call the same `pdm.monitoring.evaluation.evaluate_monitoring` function directly in bounded CPU processes; this avoids refitting and preserves separate immutable evaluation IDs. CLI reproduction for the final bundles is listed below.

An endpoint metadata audit corrected `first_measurement_ge_600pa` to the source's actual `first_measurement_gt_600pa`. The independent monitoring limit remains `>=600 Pa`. No snapshot contains an exact 600 Pa measurement. `sensor_selected_source_gt.joblib` preserves identical fitted estimators, features and horizons in a new artifact; the original sensor file remains unchanged. Verification is in `source_metadata_correction_verification.json`; this correction used **zero fits**. A partial older filter test evaluation was explicitly cancelled and marked superseded, never reported as complete.

## Acceptance evidence and status

| Requirement family | Evidence | Status / limit |
|---|---|---|
| D01–D12: causal data, splits, damaged endpoint, quality, units | `tests/test_condition_monitoring.py`; existing `test_spec_invariants.py`, `test_data_quality.py`; immutable snapshot dependencies | Software verified; MATLAB import and physical HSE Time/RUL units remain blocked |
| H01–H10: regime reference, histories, padding, legacy/continuous memory | New history/normality tests; existing `test_forecasting.py`, `test_continuous_trace.py`, `test_continuous_replay.py`, `test_training_v2.py`; actual saved fits | Software verified; healthy reference provisional, not expert-confirmed |
| F01–F14: real signal target masks, quantile repair, probabilities, identity | New forecast/calibration/bundle tests; real runtime/export verifier | Software verified; probability calibration and operational interval guarantee unavailable |
| Z01–Z14: priority, confirmation, recovery, latch, acknowledgement and eligibility | New state/timing/feedback tests and actual Train_46 measured-limit replay | Software verified; real action deadline intentionally unavailable |
| E01–E12: full clock, censoring, episode matching, budget, negative results | New evaluator tests, fit ledger, per-unit files, quarantined recipe records, frozen evaluations | Measured within 48 fits; stated extra ablations not run |
| U01–U12: seek, truth overlay, controls, export, unit isolation, compatibility | Real Playwright desktop/mobile checks, downloaded-byte comparison, prefix/trace/export parity; existing UI and missing-soma tests | Available paths verified; no native Windows certification or production protection claim |

The state/reference warning thresholds are **not** validated against an industrial false-alarm target. Observed diagnostic lead times are descriptive; without confirmed healthy negatives and a configured action window, false-failure precision and timely recall are unavailable. A grey refusal, a broad interval, or a missing forecast is retained in the denominator rather than hidden.

Remaining work: verified HSE full-history MATLAB export and origin mapping; physical Time/RUL reconciliation; expert-reviewed healthy segments across regimes; an agreed false-episode burden target and action lead/buffer; independent calibration cohorts; corrected filter feature ablations under a separately authorized new fit budget; geometry-backed envelope features if pursued. None is represented as already validated.


## Final frozen evaluation results (17 September 2026)

Both final bundles and their completed evaluations use source hash `01254421d20483f418010b13379b8c25c1e933dafecb833ae0e218132c398e15`. HEAD remains `3143fabf3a11928b915f24b03bafe89bd10a6e1c`; changes are uncommitted. Full modified-file SHA-256 inventory, tracked diff and status are saved in [final_worktree_identity.json](../output/condition-monitoring/final_worktree_identity.json), [final-worktree.patch](../output/condition-monitoring/final-worktree.patch), and [final-worktree-status.txt](../output/condition-monitoring/final-worktree-status.txt). The inventory includes concurrent changes and does not claim their ownership.

| Dataset | Bundle | Split | Evaluation | Measurements |
|---|---|---|---|---:|
| bearings | `4d7312f2b157a8c714e5ca28` | validation | [20260917T042618Z_6ed0d5](../runs/monitoring_bundles/4d7312f2b157a8c714e5ca28/evaluations/20260917T042618Z_6ed0d5/evaluation.json) | 1,679 |
| bearings | `4d7312f2b157a8c714e5ca28` | test | [20260917T042634Z_df5ab4](../runs/monitoring_bundles/4d7312f2b157a8c714e5ca28/evaluations/20260917T042634Z_df5ab4/evaluation.json) | 505 |
| filters | `98c47636891c3ac099dd12ea` | test | [20260917T042408Z_6295f6](../runs/monitoring_bundles/98c47636891c3ac099dd12ea/evaluations/20260917T042408Z_6295f6/evaluation.json) | 39,414 |
| filters | `98c47636891c3ac099dd12ea` | validation | [20260917T042409Z_80a933](../runs/monitoring_bundles/98c47636891c3ac099dd12ea/evaluations/20260917T042409Z_80a933/evaluation.json) | 5,763 |

All four evaluations are completed and labelled `exploratory_reused_holdout`. Final filter test replay took **475.790 s**, with 39,414/39,414 assessments. There is no test-based promotion.

### Event forecast and availability

MAE averages equipment-level MAE, not pooled rows. Bearing values are physical seconds; filter values are **dataset-internal units**. Filter test prefixes contain no observed terminal event, so observed-event MAE is unavailable. Censored NLL is not accuracy against the withheld future endpoint.

| Dataset / split | End-to-end available / expected | Event MAE | Common-clock-60 available / expected | Common-clock MAE | Survival NLL |
|---|---:|---:|---:|---:|---:|
| bearings / validation | 1622/1679 (96.61%) | 19125.219049 | 1519/1519 (100.00%) | 24942.188827 | unavailable |
| bearings / test | 448/505 (88.71%) | 7519.371491 | 335/335 (100.00%) | 2606.776559 | unavailable |
| filters / test | 34551/39414 (87.66%) | unavailable | 34551/36464 (94.75%) | unavailable | 0.005868 |
| filters / validation | 5173/5763 (89.76%) | 185.508470 | 5173/5173 (100.00%) | 185.508470 | 0.688609 |

Official filter test RUL is evaluated **only at the observed prefix end**, separately from censored event outcomes: 41/50 final forecasts available (82%); unit MAE **6680.652871 dataset-internal units**. Worst available unit: **Test_22**, absolute error **12264.110352**. The nine unavailable predictions stay in the coverage denominator. These poor errors and the absence of observed test failures do not support useful event timing.

### Sensor forecast

Bearings select causal local trend; no empirical interval is fabricated. Filters select quantile boosting. The nominal 5–95% band is pointwise and uncalibrated. Its broad width explains why coverage alone cannot establish useful uncertainty. Horizons 30/90 for filters are dataset-internal units, not verified seconds or minutes. Prediction coverage below is among targets with admissible history and a real matching future observation, distinct from all-clock monitoring availability.

| Dataset / split | Horizon | Targets / predictions | MAE | RMSE | Band coverage | Mean band width |
|---|---:|---:|---:|---:|---:|---:|
| bearings / validation | 300 | 1607/1607 | 0.411405 | 0.671026 | unavailable | unavailable |
| bearings / validation | 900 | 1577/1577 | 1.472101 | 1.612198 | unavailable | unavailable |
| bearings / test | 300 | 433/433 | 0.553313 | 0.873799 | unavailable | unavailable |
| bearings / test | 900 | 403/403 | 1.620032 | 2.075746 | unavailable | unavailable |
| filters / test | 30 | 38214/35986 | 3.141740 | 4.564973 | 0.974416 | 115.628639 |
| filters / test | 90 | 37714/35576 | 4.515348 | 6.594070 | 0.964218 | 109.517470 |
| filters / validation | 30 | 5523/5523 | 3.087986 | 4.983492 | 0.973403 | 101.762909 |
| filters / validation | 90 | 5423/5423 | 4.299897 | 6.595284 | 0.955471 | 94.173572 |

Sensor errors/widths are in **g** for bearings and **Pa** for filters. Saved `signal_per_unit.csv` includes every equipment/horizon result.

### State coverage, episodes and observed warning times

| Dataset / split | State available | Grey / unavailable | Diagnostic episodes | Observed endpoints |
|---|---:|---:|---:|---:|
| bearings / validation | 96.61% | 3.39% | 2 | 3 |
| bearings / test | 88.71% | 11.29% | 3 | 3 |
| filters / test | 2.92% | 97.08% | 3 | 0 |
| filters / validation | 21.81% | 78.19% | 2 | 1 |

Filter validation lacks a provisional healthy reference for eight of ten equipment regimes; filter test is grey for **97.08%** of measurements. A model can still forecast a signal in a known training regime without establishing healthy-state normality. Refusals are not green states.

| Dataset / split | Grey exposure | Green exposure | Yellow exposure |
|---|---:|---:|---:|
| bearings / validation | 3420 | 92340 | 4800 |
| bearings / test | 3420 | 1500 | 25200 |
| filters / test | 229308 | 2400 | 4476 |
| filters / validation | 26988 | 1416 | 6114 |

Exposure uses the previous issued state over the interval until the next observation; no follow-up after the last row is invented. Units are physical seconds for bearings and internal units for filters. Train_46 reaches **607.9102 Pa** on its final sample at 3756 internal units: it is red with a critical latch, but contributes zero subsequent red exposure. This is distinct from absence of a red decision.

| Unit | Split | Diagnostic lead to observed endpoint | Observed unresolved episode duration | Units |
|---|---|---:|---:|---|
| Bearing2_4 | validation | 540.000000 | 540 | physical_seconds |
| Bearing3_4 | validation | 4260.000000 | 4260 | physical_seconds |
| Bearing1_5 | test | 840.000000 | 840 | physical_seconds |
| Bearing2_5 | test | 19020.000000 | 19020 | physical_seconds |
| Bearing3_5 | test | 5340.000000 | 5340 | physical_seconds |
| Test_38 | test | unavailable | 2214 | dataset_internal |
| Test_39 | test | unavailable | 1302 | dataset_internal |
| Test_45 | test | unavailable | 960 | dataset_internal |
| Train_46 | validation | 2982.000000 | 2982 | dataset_internal |
| Train_9 | validation | unavailable | 3132 | dataset_internal |

All listed episode resolutions are unobserved. There are no admitted prognostic episodes. Censored Train_9 and Test_38/39/45 have unknown event outcomes; they are not automatically false alarms. No verified false-episode rate, precision, operational timely recall or probability reliability is claimed. Bearing validation has no diagnostic alert on Bearing1_4 before its endpoint; test diagnostic leads vary from 840 to 19020 seconds.

Worst observed-event RUL objects: validation **Bearing3_4**, MAE **37046.681267 s**; test **Bearing1_5**, MAE **13969.575500 s**. Filter validation has only one observed event (Train_46, MAE 185.508470 internal units). Equipment-balanced vs pooled sensor errors and common-clock vs end-to-end RUL errors differ substantially; neither the most favorable denominator nor the broad band is used to claim improvement.

### Artifact map and reproduction

**bearings frozen bundle:** [4d7312f2b157a8c714e5ca28](../runs/monitoring_bundles/4d7312f2b157a8c714e5ca28/monitoring_bundle.json). Associated immutable files: [unit_verification.json](../runs/monitoring_bundles/4d7312f2b157a8c714e5ca28/unit_verification.json), [healthy_reference_manifest.json](../runs/monitoring_bundles/4d7312f2b157a8c714e5ca28/healthy_reference_manifest.json), [state_policy.json](../runs/monitoring_bundles/4d7312f2b157a8c714e5ca28/state_policy.json), [calibration_report.json](../runs/monitoring_bundles/4d7312f2b157a8c714e5ca28/calibration_report.json).

**filters frozen bundle:** [98c47636891c3ac099dd12ea](../runs/monitoring_bundles/98c47636891c3ac099dd12ea/monitoring_bundle.json). Associated immutable files: [unit_verification.json](../runs/monitoring_bundles/98c47636891c3ac099dd12ea/unit_verification.json), [healthy_reference_manifest.json](../runs/monitoring_bundles/98c47636891c3ac099dd12ea/healthy_reference_manifest.json), [state_policy.json](../runs/monitoring_bundles/98c47636891c3ac099dd12ea/state_policy.json), [calibration_report.json](../runs/monitoring_bundles/98c47636891c3ac099dd12ea/calibration_report.json).

**bearings study:** [study_manifest.json](../runs/condition_studies/20260916T170147Z_bfd686/study_manifest.json), [candidate_results.csv](../runs/condition_studies/20260916T170147Z_bfd686/candidate_results.csv), [history_comparison.csv](../runs/condition_studies/20260916T170147Z_bfd686/history_comparison.csv), [history_per_unit_results.csv](../runs/condition_studies/20260916T170147Z_bfd686/history_per_unit_results.csv), [sensor_selection_per_unit.csv](../runs/condition_studies/20260916T170147Z_bfd686/sensor_selection_per_unit.csv).

**filters study:** [study_manifest.json](../runs/condition_studies/20260916T171150Z_9177ed/study_manifest.json), [candidate_results.csv](../runs/condition_studies/20260916T171150Z_9177ed/candidate_results.csv), [history_comparison.csv](../runs/condition_studies/20260916T171150Z_9177ed/history_comparison.csv), [history_per_unit_results.csv](../runs/condition_studies/20260916T171150Z_9177ed/history_per_unit_results.csv), [sensor_selection_per_unit.csv](../runs/condition_studies/20260916T171150Z_9177ed/sensor_selection_per_unit.csv).

**bearings validation `20260917T042618Z_6ed0d5`:** [per_unit_results.csv](../runs/monitoring_bundles/4d7312f2b157a8c714e5ca28/evaluations/20260917T042618Z_6ed0d5/per_unit_results.csv), [event_per_unit_results.csv](../runs/monitoring_bundles/4d7312f2b157a8c714e5ca28/evaluations/20260917T042618Z_6ed0d5/event_per_unit_results.csv), [signal_forecasts.parquet](../runs/monitoring_bundles/4d7312f2b157a8c714e5ca28/evaluations/20260917T042618Z_6ed0d5/signal_forecasts.parquet), [state_history.parquet](../runs/monitoring_bundles/4d7312f2b157a8c714e5ca28/evaluations/20260917T042618Z_6ed0d5/state_history.parquet), [alert_episodes.parquet](../runs/monitoring_bundles/4d7312f2b157a8c714e5ca28/evaluations/20260917T042618Z_6ed0d5/alert_episodes.parquet), [feedback.jsonl](../runs/monitoring_bundles/4d7312f2b157a8c714e5ca28/evaluations/20260917T042618Z_6ed0d5/feedback.jsonl), [observations.jsonl](../runs/monitoring_bundles/4d7312f2b157a8c714e5ca28/evaluations/20260917T042618Z_6ed0d5/observations.jsonl), [diagnostics/event_monitoring_per_unit.csv](../runs/monitoring_bundles/4d7312f2b157a8c714e5ca28/evaluations/20260917T042618Z_6ed0d5/diagnostics/event_monitoring_per_unit.csv), [diagnostics/episode_lead_and_duration.csv](../runs/monitoring_bundles/4d7312f2b157a8c714e5ca28/evaluations/20260917T042618Z_6ed0d5/diagnostics/episode_lead_and_duration.csv), [diagnostics/summary.json](../runs/monitoring_bundles/4d7312f2b157a8c714e5ca28/evaluations/20260917T042618Z_6ed0d5/diagnostics/summary.json).

**bearings test `20260917T042634Z_df5ab4`:** [per_unit_results.csv](../runs/monitoring_bundles/4d7312f2b157a8c714e5ca28/evaluations/20260917T042634Z_df5ab4/per_unit_results.csv), [event_per_unit_results.csv](../runs/monitoring_bundles/4d7312f2b157a8c714e5ca28/evaluations/20260917T042634Z_df5ab4/event_per_unit_results.csv), [signal_forecasts.parquet](../runs/monitoring_bundles/4d7312f2b157a8c714e5ca28/evaluations/20260917T042634Z_df5ab4/signal_forecasts.parquet), [state_history.parquet](../runs/monitoring_bundles/4d7312f2b157a8c714e5ca28/evaluations/20260917T042634Z_df5ab4/state_history.parquet), [alert_episodes.parquet](../runs/monitoring_bundles/4d7312f2b157a8c714e5ca28/evaluations/20260917T042634Z_df5ab4/alert_episodes.parquet), [feedback.jsonl](../runs/monitoring_bundles/4d7312f2b157a8c714e5ca28/evaluations/20260917T042634Z_df5ab4/feedback.jsonl), [observations.jsonl](../runs/monitoring_bundles/4d7312f2b157a8c714e5ca28/evaluations/20260917T042634Z_df5ab4/observations.jsonl), [diagnostics/event_monitoring_per_unit.csv](../runs/monitoring_bundles/4d7312f2b157a8c714e5ca28/evaluations/20260917T042634Z_df5ab4/diagnostics/event_monitoring_per_unit.csv), [diagnostics/episode_lead_and_duration.csv](../runs/monitoring_bundles/4d7312f2b157a8c714e5ca28/evaluations/20260917T042634Z_df5ab4/diagnostics/episode_lead_and_duration.csv), [diagnostics/summary.json](../runs/monitoring_bundles/4d7312f2b157a8c714e5ca28/evaluations/20260917T042634Z_df5ab4/diagnostics/summary.json).

**filters test `20260917T042408Z_6295f6`:** [per_unit_results.csv](../runs/monitoring_bundles/98c47636891c3ac099dd12ea/evaluations/20260917T042408Z_6295f6/per_unit_results.csv), [event_per_unit_results.csv](../runs/monitoring_bundles/98c47636891c3ac099dd12ea/evaluations/20260917T042408Z_6295f6/event_per_unit_results.csv), [signal_forecasts.parquet](../runs/monitoring_bundles/98c47636891c3ac099dd12ea/evaluations/20260917T042408Z_6295f6/signal_forecasts.parquet), [state_history.parquet](../runs/monitoring_bundles/98c47636891c3ac099dd12ea/evaluations/20260917T042408Z_6295f6/state_history.parquet), [alert_episodes.parquet](../runs/monitoring_bundles/98c47636891c3ac099dd12ea/evaluations/20260917T042408Z_6295f6/alert_episodes.parquet), [feedback.jsonl](../runs/monitoring_bundles/98c47636891c3ac099dd12ea/evaluations/20260917T042408Z_6295f6/feedback.jsonl), [observations.jsonl](../runs/monitoring_bundles/98c47636891c3ac099dd12ea/evaluations/20260917T042408Z_6295f6/observations.jsonl), [diagnostics/event_monitoring_per_unit.csv](../runs/monitoring_bundles/98c47636891c3ac099dd12ea/evaluations/20260917T042408Z_6295f6/diagnostics/event_monitoring_per_unit.csv), [diagnostics/episode_lead_and_duration.csv](../runs/monitoring_bundles/98c47636891c3ac099dd12ea/evaluations/20260917T042408Z_6295f6/diagnostics/episode_lead_and_duration.csv), [diagnostics/summary.json](../runs/monitoring_bundles/98c47636891c3ac099dd12ea/evaluations/20260917T042408Z_6295f6/diagnostics/summary.json).

**filters validation `20260917T042409Z_80a933`:** [per_unit_results.csv](../runs/monitoring_bundles/98c47636891c3ac099dd12ea/evaluations/20260917T042409Z_80a933/per_unit_results.csv), [event_per_unit_results.csv](../runs/monitoring_bundles/98c47636891c3ac099dd12ea/evaluations/20260917T042409Z_80a933/event_per_unit_results.csv), [signal_forecasts.parquet](../runs/monitoring_bundles/98c47636891c3ac099dd12ea/evaluations/20260917T042409Z_80a933/signal_forecasts.parquet), [state_history.parquet](../runs/monitoring_bundles/98c47636891c3ac099dd12ea/evaluations/20260917T042409Z_80a933/state_history.parquet), [alert_episodes.parquet](../runs/monitoring_bundles/98c47636891c3ac099dd12ea/evaluations/20260917T042409Z_80a933/alert_episodes.parquet), [feedback.jsonl](../runs/monitoring_bundles/98c47636891c3ac099dd12ea/evaluations/20260917T042409Z_80a933/feedback.jsonl), [observations.jsonl](../runs/monitoring_bundles/98c47636891c3ac099dd12ea/evaluations/20260917T042409Z_80a933/observations.jsonl), [diagnostics/event_monitoring_per_unit.csv](../runs/monitoring_bundles/98c47636891c3ac099dd12ea/evaluations/20260917T042409Z_80a933/diagnostics/event_monitoring_per_unit.csv), [diagnostics/episode_lead_and_duration.csv](../runs/monitoring_bundles/98c47636891c3ac099dd12ea/evaluations/20260917T042409Z_80a933/diagnostics/episode_lead_and_duration.csv), [diagnostics/summary.json](../runs/monitoring_bundles/98c47636891c3ac099dd12ea/evaluations/20260917T042409Z_80a933/diagnostics/summary.json).

[Combined runtime/export verification](../output/condition-monitoring/runtime_and_export_verification.json) verifies dependency/export hashes, all four full-clock exports, JSON/parquet agreement, real prefix seek/sequential equivalence, trace/plain prediction equivalence, and independence from future rows/truth labels. [Budget preflight verification](../output/condition-monitoring/budget_preflight_verification.json) confirms both pending filter ablations fail before starting any additional fit at 48/48.

The following evaluation commands create **new evaluation IDs**, do not refit, and do not overwrite the four results above. Do not rerun training commands unless a separate additional fit budget is intended.

```sh
.venv/bin/python -m pdm monitor-evaluate --help
```
The final direct evaluation entry point used was `pdm.monitoring.evaluation.evaluate_monitoring(bundle_id, split_name, log=...)`; final IDs are in the table. The exact read-only postprocessing commands executed were:
```sh
.venv/bin/python scripts/verify_condition_runtime.py 98c47636891c3ac099dd12ea --output output/condition-monitoring/runtime_and_export_verification_filters.json
.venv/bin/python scripts/verify_condition_runtime.py 4d7312f2b157a8c714e5ca28 --output output/condition-monitoring/runtime_and_export_verification_bearings.json
.venv/bin/python scripts/condition_outcome_report.py 98c47636891c3ac099dd12ea
.venv/bin/python scripts/condition_outcome_report.py 4d7312f2b157a8c714e5ca28
```

### Real UI verification

[browser_qa_results.json](../output/condition-monitoring/browser_qa_results.json) keeps the original bundle identity for the complete control/mobile/export check and separately records the final-bundle smoke checks. Desktop 1440×1000 and mobile 390×844 passed; paused replay stayed paused, Start/Pause/Next/Reset and unit changes worked, truth overlay did not alter the issued state, and a downloaded evaluation matched its saved bytes. No horizontal overflow, Streamlit exceptions or browser console errors were observed. Final bundles were loaded again after the endpoint metadata correction. Train_46 at measurement 625 displayed **Urgent action required**, valid observation, a persistent critical latch and the actual 60-point / 354-internal-unit history.

Screenshots: [bearing desktop forecast](../output/playwright/condition-desktop-plot.png), [mobile forecast](../output/playwright/condition-mobile-plot.png), [mobile controls](../output/playwright/condition-mobile-controls.png), [final filter critical state](../output/playwright/condition-filters-critical-final.png). These are observed UI screenshots, not mockups. Earlier failed selector/timer QA logs are retained; they are superseded by the named passing checks.

### Train-only sensor candidate comparison

Unit-balanced MAE below comes from the same internal training fold, before test replay. Selection averages the configured horizon scores.

| Dataset | Method | Horizon | Unit MAE |
|---|---|---:|---:|
| bearings | causal_local_trend | 300 | 0.089917 |
| bearings | causal_local_trend | 900 | 0.237502 |
| bearings | multi_horizon_quantile_boosting | 300 | 0.130244 |
| bearings | multi_horizon_quantile_boosting | 900 | 0.466157 |
| bearings | persistence | 300 | 0.096995 |
| bearings | persistence | 900 | 0.232778 |
| filters | causal_local_trend | 30 | 3.495568 |
| filters | causal_local_trend | 90 | 6.481134 |
| filters | multi_horizon_quantile_boosting | 30 | 4.165445 |
| filters | multi_horizon_quantile_boosting | 90 | 5.081646 |
| filters | persistence | 30 | 3.314373 |
| filters | persistence | 90 | 6.581102 |

Final filter grey-state reason counts overlap: normality unavailable **35,769**, constant-signal diagnostic **2,526**, unknown operating regime **2,444**, collecting contiguous history **950**. There are **38,265 grey measurements** total, not the sum of overlapping reasons. Constant signal remains a quality diagnostic rather than automatic evidence of equipment failure.

Final CLI reproduction (one evaluation at a time; creates a new result directory):
```sh
.venv/bin/python -m pdm monitor-evaluate --bundle-id 4d7312f2b157a8c714e5ca28 --split validation
.venv/bin/python -m pdm monitor-evaluate --bundle-id 4d7312f2b157a8c714e5ca28 --split test
.venv/bin/python -m pdm monitor-evaluate --bundle-id 98c47636891c3ac099dd12ea --split validation
.venv/bin/python -m pdm monitor-evaluate --bundle-id 98c47636891c3ac099dd12ea --split test
```
The final four evaluations were actually executed by the direct shared evaluator, not those four worker dispatch commands. The CLI argument names were checked with `monitor-evaluate --help`. Worker behavior was exercised on the earlier validation bundle as described above.

Final engineering result: the implemented causal monitoring paths and compatibility checks pass. Predictive-quality improvement and operational action timing remain **unproven**. In particular, poor event errors, extensive missing filter reference coverage, broad uncalibrated bands and incomplete budget-blocked filter ablations prohibit an operational promotion.
