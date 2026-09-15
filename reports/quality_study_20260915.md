# Quality study · 15 September 2026

## Outcome

Completed 14 full experiments on real source data: nine main models and five additional GRU censoring configurations. Four saved main-study comparisons cover validation and test for both datasets. All 14 checkpoints open in the unified report.

The application now provides **Data Quality → Training → Model Report → Compare Models**, with actual architecture states, immutable data bindings, explicit evaluation selection, and CSV/JSON exports. No ensemble was added.

## Data admitted to the study

| Dataset | Admitted records | Excluded records | Attention records retained | Train / validation / test units |
| --- | --- | --- | --- | --- |
| Bearings | 9,216 | 0 | 1,121 | 9 / 3 / 3 |
| Filters | 78,236 | 598 | 2,439 | 39 / 10 / 50 |

`Train_28` is quarantined because source time starts at 157.2 and then moves backward to 0.2. Its 598 rows remain in the source and exclusion registry. No failure endpoint is moved. Large signals are retained as potential degradation evidence.

The filter training cohort retains **35 censored series and four observed failures**. Validation contains ten series and **one observed failure, `Train_46`**. Fifty test prefixes have official RUL labels; those labels are not failures observed inside the prefixes.

Data versions: bearings `20260915T153526Z_04154804`; filters `20260915T153833Z_4182d92b`. Admission policy `admission_v1`; its SHA-256 and detailed row audit are saved with the data and each experiment.

## Main matrix · seed 42

Use validation to select candidates. Test is a previously inspected holdout and remains exploratory. Every unit has equal weight; all nine main runs cover 100% of the common eligible clock after warmup. Network size and memory differ, so Full CNS versus a small window model does not isolate topology alone.

### Bearings

Primary metric: MAE in the last 30 minutes before the registered experiment end, averaged equally across three bearings. The table expresses errors in minutes.

| Model | State size / memory | Validation MAE, min | Reused test MAE, min | Timely validation warnings |
| --- | --- | --- | --- | --- |
| GRU | 64 / window_reset | 126.85 | 30.79 | 0/3 |
| LSTM | 64 / window_reset | 133.42 | 47.10 | 0/3 |
| Full MaleCNS | 166,700 / continuous | 133.82 | 20.17 | 0/3 |
| Fly reservoir | 1,000 / window_reset | 479.22 | 19.28 | 1/3 |
| Random reservoir | 1,000 / window_reset | 493.29 | 18.96 | 1/3 |

**GRU leads the specified validation metric, but accuracy is still insufficient for reliable maintenance timing:** its final-30-minute MAE is 126.85 minutes and it produces no timely warnings on the three validation bearings. The test ordering differs. This stage establishes a reproducible comparison; it does not establish deployment readiness.

Full CNS empirical interval coverage is 92.5% on its calibration units, with mean width 832.6 minutes. On reused test it is 94.4%, with mean width 208.8 minutes. Broad intervals explain why coverage alone is not useful forecast precision.

### Filters

Validation primary metric: survival NLL across all ten observed/censored series. Test primary metric: MAE at the final point of each of the fifty prefixes against official RUL. Filter errors below use **internal seconds (`Time × 60`); the original time unit remains unverified**.

| Model | State size / memory | Validation NLL | Official prefix-end test MAE, internal s |
| --- | --- | --- | --- |
| GRU | 64 / window_reset | 0.7362 | 4845.20 |
| Random reservoir | 1,000 / window_reset | 0.7891 | 1649.79 |
| Fly reservoir | 1,000 / window_reset | 0.7900 | 1647.72 |
| LSTM | 64 / window_reset | 0.8030 | 3015.61 |

GRU leads the main validation matrix. Fly/random reservoirs have lower error on reused test, which is not used to change the selected model. Weibull 5–95% ranges describe the fitted distribution and are not empirical coverage guarantees.

### Frozen warning rules

Bearings: trigger horizon 2,220 s, three consecutive confirmations, minimum action lead 1,110 s. Filters: 420 internal s, three confirmations, minimum lead 210 internal s. Reset factor 1.2. These settings were fixed before test and are shared within each dataset.

## Censoring experiment · GRU only

Each pair uses the same complete validation cohort, history 20 and all eligible training windows. Seeds change initialization/sampling, not equipment split membership. The main GRU seed-42 run supplies the first all-valid reference. Additional configurations are evaluated on validation only.

| Seed | All-valid NLL | Failures-only NLL | All-valid event MAE, internal s | Failures-only event MAE, internal s |
| --- | --- | --- | --- | --- |
| 42 | 0.7362 | 1.1788 | 443.61 | 1026.20 |
| 43 | 0.7230 | 2.6398 | 175.28 | 716.56 |
| 44 | 0.6936 | 3.6381 | 299.98 | 804.02 |

Retaining censored series improves both metrics in **all three pairs**. Mean NLL: 0.7176 versus 2.4856; mean observed-event MAE: 306.29 versus 848.93 internal seconds. This supports retaining these valid censored observations for the tested GRU setup. It is not a claim about every architecture: the validation error comes from only **one independent failure**. Repeated seeds do not create additional independent failures.

## Reproducibility and acceptance

- Real sources and connectome; history 20; all eligible windows. Gradient models use up to 30 epochs with early stopping. Fly/random reservoirs have 1,000 nodes; Full MaleCNS computes all 166,700 classified neurons and 25,582,938 directed connections.
- Full CNS retains its grouped train-only readout selection and continuous memory. Other models use their trained reset window. Display samples and anatomical context are labelled separately from computing nodes.
- Admission checks run before fitting preprocessing and training; corruption, conflicting duplicates and unreliable endpoints are excluded without deleting sources or bridging gaps. Changed snapshots block checkpoint continuation.
- Four filter test evaluations were republished with explicit official truth trajectories anchored to the saved source endpoint. Their 39,414 forecasts per model were reused unchanged. The original evaluations remain available; new configs record `source_evaluation_id` and `inference_reused_unchanged`.
- Trace/ordinary inference, rewind/replay and future/ground-truth independence passed on all 14 real checkpoints. Point and available interval values match their saved evaluation CSVs.
- All 186,428 rows in the four comparison prediction exports match their exact saved evaluations. Comparison CSV and JSON scores agree. Browser-downloaded filter validation CSV/JSON also match all 22,252 selected forecasts.
- Browser QA: all 14 reports at 1440×1000 and 390×844; all architectures; both datasets; actual state panels; point/range cards; start/pause/next/reset, truth toggle and seeking; model switching; comparison split switching, saved composition restore, downloads, and no horizontal page overflow.
- Full pytest: **348 passed, zero failed/errors/skipped**. Ruff and `git diff --check` pass. Browser: zero console errors; existing Streamlit iframe feature-policy and bundled Three.js deprecation warnings remain.

### Evidence

- [Exact batch manifest](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/batches/20260915T154129Z_78a7fa/manifest.json>)
- [Full training log](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/batches/20260915T154129Z_78a7fa/batch.log>)
- [Paired censoring CSV](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/batches/20260915T154129Z_78a7fa/censoring_ablation.csv>)
- [Real-model and comparison verification](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/output/quality-study-verification.json>)
- [Browser report verification](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/output/playwright/browser-report-qa.json>)
- [Pytest JUnit results](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/output/quality-study-tests.xml>)
- [Method and compatibility rules](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/docs/quality_and_comparison.md>)

### Saved main comparisons

- **bearings / validation**: [composition](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/comparisons/74a2f96a32da/comparison.json>) · [ranking CSV](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/comparisons/74a2f96a32da/table.csv>) · [per-unit CSV](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/comparisons/74a2f96a32da/per_unit.csv>) · [predictions CSV](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/comparisons/74a2f96a32da/predictions.csv>)
- **bearings / test**: [composition](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/comparisons/87aca636a8e7/comparison.json>) · [ranking CSV](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/comparisons/87aca636a8e7/table.csv>) · [per-unit CSV](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/comparisons/87aca636a8e7/per_unit.csv>) · [predictions CSV](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/comparisons/87aca636a8e7/predictions.csv>)
- **filters / validation**: [composition](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/comparisons/d75c949ab96f/comparison.json>) · [ranking CSV](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/comparisons/d75c949ab96f/table.csv>) · [per-unit CSV](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/comparisons/d75c949ab96f/per_unit.csv>) · [predictions CSV](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/comparisons/d75c949ab96f/predictions.csv>)
- **filters / test**: [composition](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/comparisons/c3f6c2e23e8b/comparison.json>) · [ranking CSV](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/comparisons/c3f6c2e23e8b/table.csv>) · [per-unit CSV](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/comparisons/c3f6c2e23e8b/per_unit.csv>) · [predictions CSV](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/comparisons/c3f6c2e23e8b/predictions.csv>)

### Experiment registry

| Dataset / role | Model / seed / training cohort | Run | Validation evaluation | Test evaluation |
| --- | --- | --- | --- | --- |
| bearings / main | gru / 42 / all valid | [bearings_gru_20260915_204130_89b1b3](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/bearings/bearings_gru_20260915_204130_89b1b3>) | 20260915T154140Z_c51f03c5 | 20260915T154141Z_302f8f53 |
| bearings / main | lstm / 42 / all valid | [bearings_lstm_20260915_204141_ea12fe](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/bearings/bearings_lstm_20260915_204141_ea12fe>) | 20260915T154150Z_69ac7a01 | 20260915T154151Z_3712ea7b |
| bearings / main | fly_connectome_reservoir / 42 / all valid | [bearings_fly_connectome_reservoir_20260915_204210_54bf27](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/bearings/bearings_fly_connectome_reservoir_20260915_204210_54bf27>) | 20260915T154220Z_b9b90747 | 20260915T154222Z_cdf8275a |
| bearings / main | random_reservoir / 42 / all valid | [bearings_random_reservoir_20260915_204333_493cd9](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/bearings/bearings_random_reservoir_20260915_204333_493cd9>) | 20260915T154405Z_1e5b800d | 20260915T154714Z_764f2f07 |
| bearings / main | full_cns / 42 / all valid | [bearings_fly_connectome_reservoir_20260915_213107_ebc8d0](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/bearings/bearings_fly_connectome_reservoir_20260915_213107_ebc8d0>) | 20260915T163220Z_ef47e64f | 20260915T163237Z_334a7b00 |
| filters / main | gru / 42 / all valid | [filters_gru_20260915_213239_dbf182](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/filters/filters_gru_20260915_213239_dbf182>) | 20260915T163347Z_d2e7a87c | 20260915T170101Z_42a98311 |
| filters / main | lstm / 42 / all valid | [filters_lstm_20260915_213703_840256](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/filters/filters_lstm_20260915_213703_840256>) | 20260915T163835Z_d1581c74 | 20260915T170103Z_06f8ab25 |
| filters / main | fly_connectome_reservoir / 42 / all valid | [filters_fly_connectome_reservoir_20260915_214154_741d00](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/filters/filters_fly_connectome_reservoir_20260915_214154_741d00>) | 20260915T164607Z_5e5eb458 | 20260915T170105Z_78d244ea |
| filters / main | random_reservoir / 42 / all valid | [filters_random_reservoir_20260915_215021_ae03aa](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/filters/filters_random_reservoir_20260915_215021_ae03aa>) | 20260915T165454Z_116b4ac2 | 20260915T170106Z_038c1d47 |
| filters / ablation | gru / 42 / failures only | [filters_gru_20260915_221450_3a2590](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/filters/filters_gru_20260915_221450_3a2590>) | 20260915T171513Z_a5579c84 | not evaluated |
| filters / ablation | gru / 43 / all valid | [filters_gru_20260915_221514_1dbc7a](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/filters/filters_gru_20260915_221514_1dbc7a>) | 20260915T171557Z_9fa4864c | not evaluated |
| filters / ablation | gru / 43 / failures only | [filters_gru_20260915_221558_ba538a](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/filters/filters_gru_20260915_221558_ba538a>) | 20260915T171620Z_1cbf1eab | not evaluated |
| filters / ablation | gru / 44 / all valid | [filters_gru_20260915_221621_26f10c](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/filters/filters_gru_20260915_221621_26f10c>) | 20260915T171717Z_1aeb58cf | not evaluated |
| filters / ablation | gru / 44 / failures only | [filters_gru_20260915_221718_08482b](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/runs/filters/filters_gru_20260915_221718_08482b>) | 20260915T171747Z_68ac0fed | not evaluated |

### Report previews

[GRU report](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/output/playwright/report-filters-gru-desktop.png>) · [LSTM h/c report](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/output/playwright/report-filters-lstm-desktop.png>) · [Random reservoir](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/output/playwright/report-bearings-random_reservoir-desktop.png>) · [Full CNS](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/output/playwright/report-bearings-full-cns-desktop.png>) · [Mobile report](</Users/aks/01 Development/Apps/Predictive Maintenance Lab/output/playwright/report-bearings-full-cns-mobile.png>)
