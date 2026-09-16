# Training improvement study · 16 September 2026

## Outcome

Completed **42 real training runs**: 25 pilot/grouped-validation runs, nine main models and eight initialization confirmations. The 17 final new checkpoints and nine saved reference checkpoints have 52 concrete validation/test evaluations. Four saved comparisons include the new and reference main models. No ensemble or architecture/memory enlargement was added.

**Useful progress, but insufficient evidence of reliable maintenance timing.** GRU leads validation on both datasets. Bearing errors and warning usefulness fail the agreed laboratory targets. Filter GRU improves NLL, observed-event error and reused-test error; its validation warning success comes from only one independent observed failure.

## Main matrix · seed 42

All nine main runs cover the complete shared eligible prediction clock after the 20-measurement warmup. Candidates were frozen before test. Test remains a previously inspected, exploratory holdout; its ordering does not select models.

### Bearings

MAE in the final 30 minutes before experiment end, equal weight per bearing. Values below are minutes. Experiment end is the existing failure approximation, not a verified industrial failure time.

| Model | Old validation | New validation | Old reused test | New reused test | Validation warning goal |
| --- | --- | --- | --- | --- | --- |
| GRU | 126.8480 | 62.7108 | 30.7907 | 51.0920 | False |
| LSTM | 133.4181 | 126.9968 | 47.1016 | 55.5366 | False |
| Fly reservoir (1,000) | 479.2207 | 151.9120 | 19.2815 | 14.9859 | False |
| Random reservoir (1,000) | 493.2923 | 152.3241 | 18.9552 | 14.9049 | False |
| Full MaleCNS (166,700) | 133.8204 | 133.8204 | 20.1738 | 20.1738 | False |

No main model meets MAE ≤10 minutes. On test, Fly, Random and Full CNS each have one useful warning for three failures (33% recall and precision); GRU/LSTM have none. Validation has no useful warnings for any main model. The GRU validation errors by bearing are 122.96, 57.97 and 7.20 minutes: aggregate improvement does not make all objects usable. Its reused-test error worsens from 30.79 to 51.09 minutes, so the validation improvement is not a demonstrated generalization improvement.

### Filters

Validation uses equal-equipment survival NLL on all ten observed/censored series. Test uses official RUL at the end of 50 prefixes, in **internal seconds (Time × 60)**; the physical time unit is unverified.

| Model | Old validation | New validation | Old reused test | New reused test | Validation warning goal |
| --- | --- | --- | --- | --- | --- |
| GRU | 0.7362 | 0.6493 | 4845.2034 | 3911.8819 | True |
| LSTM | 0.8030 | 0.7038 | 3015.6089 | 5868.6139 | False |
| Fly reservoir (1,000) | 0.7900 | 0.7987 | 1647.7242 | 1591.3638 | False |
| Random reservoir (1,000) | 0.7891 | 0.7979 | 1649.7902 | 1597.1762 | False |

GRU observed-failure validation MAE falls from 443.61 to 172.10 internal seconds. Its one validation warning is useful and it produces no other validation episodes. Seeds 43/44 also meet this warning target on the **same** validation objects. Test prefixes contain official RUL references, not observed failures inside the prefix; they do not establish warning recall.

## What each change contributed

- **Selection criterion versus extra epochs:** on one bearing GRU trajectory, the former all-history checkpoint rule stops at epoch 7 (best 2), giving 126.85 minutes under the agreed final-30-minute metric. Aligning the criterion with the same 30/5 limit selects epoch 23 (stop 28), giving 67.84 minutes. Continuing to 100 selects epoch 50, giving 62.71 minutes: an additional 7.56% improvement.
- **Filter epoch count:** both the former 30/5 rule and the 100-epoch diagnostic select epoch 1 (NLL 0.73361). More epochs did not help.
- **Adaptive schedule/stopping:** bearing initial rates 0.001 / 0.0003 give 64.11 / 94.51 minutes, behind the fixed diagnostic. Filter rate 0.0003 improves pilot NLL to 0.70580. Scheduling and stopping changed together, so their individual causal effects are not isolated.
- **Complete pass without replacement:** bearing pilot error worsens to 79.29 minutes; adding half weight to final-30-minute windows worsens it further to 95.58. Filter full-pass NLL improves to 0.66215.
- **Causal degradation features:** bearing pilot error worsens to 98.52 minutes, so the selected bearing recipe keeps base features. Filter pilot NLL improves to 0.64929. In train-only grouped validation, augmented features win three of four folds; mean NLL is 1.39749 versus 1.63651 for base features. The filter recipe uses full-pass sampling and adaptive rate 0.0003.
- **Reservoir readouts:** all three bearing reservoir architectures select log1p and ridge strength 1 from the identical train-only grid. Full CNS selects the same readout setting as its reference and its point-score result is unchanged. Fly/Random improve substantially over their former linear readouts but still fail the laboratory targets.

The retained bearing recipe is fixed-rate 100 epochs with the original equal-object replacement sampler. The full-pass and feature experiments are retained as negative results. Their scope is the tested GRU configuration.

## Initialization sensitivity

| Dataset | Model | Seed 42 | Seed 43 | Seed 44 | Sample std |
| --- | --- | --- | --- | --- | --- |
| bearings | GRU | 62.7108 | 116.6966 | 86.6666 | 27.0498 |
| bearings | LSTM | 126.9968 | 99.6268 | 78.9920 | 24.0810 |
| filters | GRU | 0.6493 | 0.6634 | 0.6789 | 0.0148 |
| filters | LSTM | 0.7038 | 0.7171 | 0.6528 | 0.0340 |

Bearing GRU performance varies substantially with initialization. Seeds are repetitions on the same equipment, not new failures. Independent validation failure counts remain **three bearings and one filter**.

## Simple controls and laboratory gates

| dataset_id | split | architecture | primary_score | prediction_coverage | primary_points_available | primary_points_expected |
| --- | --- | --- | --- | --- | --- | --- |
| bearings | validation | current_feature_ridge | 6996.9144 | 1.0000 | 82 | 82 |
| bearings | validation | age_by_regime | 29320.0000 | 1.0000 | 82 | 82 |
| bearings | test | current_feature_ridge | 1170.1291 | 1.0000 | 90 | 90 |
| bearings | test | age_by_regime | 52800.0000 | 1.0000 | 90 | 90 |
| filters | validation | current_feature_weibull | 0.8698 | 1.0000 | 5563 | 5563 |
| filters | validation | pressure_trend | — | 0.0857 | 0 | 5563 |
| filters | test | current_feature_weibull | 5981.4000 | 1.0000 | 50 | 50 |
| filters | test | pressure_trend | 491.5268 | 0.0886 | 22 | 50 |

Baseline bearing scores in this raw table are seconds, filter validation scores are NLL, and filter test scores are internal seconds. Pressure-trend errors describe available predictions only: low coverage prevents a full-cohort win. The linear feature controls have complete coverage. The 20% error-improvement gate and the additional filter NLL gate are recorded per run in results.csv.

The pressure-trend method does not produce a survival distribution, so its validation NLL is unavailable (0/5,563 NLL points). Its test error uses only 22 of 50 prefix endpoints.

Warning thresholds are 1,800 seconds for bearings and 420 internal seconds for filters; useful lead is 900–1,800 or 210–420, three confirmations, reset factor 1.2. Early, late, repeated and unknown episodes are separate. A lower RUL error without useful warnings does not establish suitability.

## Implementation and reproducibility

- Versioned protocol and feature hashes, fixed data snapshots, causal feature history before model-window slicing, gap resets, float64 Weibull likelihood and explicit nonfinite failures.
- Exact checkpoint continuation restores optimizer, scheduler, stopping state and random generators. Full-pass weights equalize equipment contribution; the final short batch preserves each window coefficient.
- Fixed-reservoir caches bind data, preprocessing, graph weights, input weights, windows and memory mode. Full CNS still computes 166,700 neurons and 25,582,938 directed connections with continuous memory.
- Test candidates, warning settings and simple-control coefficients were frozen before evaluation. Saved reference weights were re-evaluated unchanged. Earlier attempts and protocol corrections remain in the manifest; only corrected completed recipes enter the results.
- CSV/source timestamp matching tolerates only floating-point round-trip noise, rejects ambiguity and missing measurements, and does not substitute nearby observations.

## Acceptance evidence

- Full pytest: **369 passed**; Ruff clean. Two existing PyTorch warnings concern sparse CSR beta support and a non-writable test fixture array.
- Audited all 42 training histories: best checkpoint is the actual minimum selection metric; diagnostic runs have 100 epochs; adaptive runs respect their bounds; full-pass histories use every eligible window once.
- All 17 final new checkpoints pass ordinary/trace parity, deterministic seek and future/ground-truth independence. Saved forecast differences at the checked points are at most **0.001954 seconds**, within float32 tolerance and below display precision.
- All **372,856** forecast rows across four comparisons match their concrete saved evaluations. Ranking CSV/JSON values match as well.
- Browser-downloaded filter validation CSV/JSON match all **22,252** selected forecast rows. The downloaded study results CSV is byte-identical to the saved artifact.
- Browser QA covers all nine main reports at 1440×1000 and 390×844: actual architecture states, displayed RUL versus saved CSV, Start/Pause/Next/Reset, seek/replay, ground-truth toggling, model switching and mobile controls. No horizontal page overflow or JavaScript errors. Existing Streamlit iframe and bundled Three.js warnings remain.
- A legacy GRU report also opens on desktop/mobile with its original forecast. The point-only chart legend says **Forecast**; ranges retain their empirical/Weibull provenance.

Evidence: [model and export verification](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/output/training-study-verification.json>) · [pytest log](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/output/training-v2-pytest.log>).

[Browser verification](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/output/playwright/training-v2-browser-qa.json>) · [GRU report](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/output/playwright/v2-report-filters-gru-desktop.png>) · [Full CNS report](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/output/playwright/v2-report-bearings-full_cns-desktop.png>) · [Mobile report](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/output/playwright/v2-report-filters-random_reservoir-mobile.png>) · [Training controls](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/output/playwright/training-v2-controls-desktop.png>).

## Artifacts

- [Full generated study report](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/training_studies/20260915T184221Z/report.md>)
- [All results CSV](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/training_studies/20260915T184221Z/results.csv>) · [JSON](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/training_studies/20260915T184221Z/results.json>)
- [Screening and parent deltas](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/training_studies/20260915T184221Z/screening.csv>) · [Grouped folds](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/training_studies/20260915T184221Z/grouped_validation.csv>)
- [Manifest with exact run/evaluation IDs](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/training_studies/20260915T184221Z/manifest.json>)
- [Protocol documentation](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/docs/training_protocol.md>)
- [bearings / validation comparison](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/comparisons/a60eb894e58f/comparison.json>) · [Table CSV](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/comparisons/a60eb894e58f/table.csv>) · [Forecast CSV](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/comparisons/a60eb894e58f/predictions.csv>)
- [bearings / test comparison](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/comparisons/6982042a86e8/comparison.json>) · [Table CSV](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/comparisons/6982042a86e8/table.csv>) · [Forecast CSV](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/comparisons/6982042a86e8/predictions.csv>)
- [filters / validation comparison](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/comparisons/e0b2737f926f/comparison.json>) · [Table CSV](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/comparisons/e0b2737f926f/table.csv>) · [Forecast CSV](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/comparisons/e0b2737f926f/predictions.csv>)
- [filters / test comparison](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/comparisons/4672524600d5/comparison.json>) · [Table CSV](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/comparisons/4672524600d5/table.csv>) · [Forecast CSV](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/comparisons/4672524600d5/predictions.csv>)
