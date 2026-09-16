# Training protocol v2

## Selection and useful warnings

The bearing checkpoint metric is equal-equipment MAE on windows with `0 < RUL ≤ 1800` internal seconds. Filter checkpoints minimize equal-equipment Weibull survival NLL on every eligible observed or censored window. NLL uses a density in **internal seconds**, so scores remain comparable when a training fold changes the target normalization. Missing or nonfinite forecasts make a checkpoint ineligible.

Training loss is an optimization objective; a lower training loss does not select a checkpoint. The saved best checkpoint follows the actual minimum validation metric, even when an improvement is too small to reset the stopping counter.

Warning rules are fixed before test: RUL threshold 1800 seconds for bearings and 420 internal seconds for filters, three consecutive confirmations, reset factor 1.2, useful lead time between half and all of the threshold. One observed failure can receive at most one useful episode. Early, late, repeated and unknown episodes remain visible. A censored alarm needs enough follow-up to establish a false alarm; insufficient follow-up is unknown. All observed failures stay in the timely-recall denominator, including objects with insufficient usable history; their count is also shown separately.

Bearing laboratory targets: final-30-minute MAE ≤600 seconds, timely warning recall ≥90%, useful episode precision ≥80%. The baseline target is ≥20% error reduction at equal coverage; filter validation additionally requires lower NLL, expressed as an absolute difference. These are laboratory targets, with three validation bearings and one observed validation filter failure, not industrial reliability evidence.

## Optimization

| Setting | Diagnostic | Adaptive |
| --- | --- | --- |
| Epochs | Exactly 100 | Maximum 100, minimum 20 |
| Initial learning rate | 0.001 | 0.001 or 0.0003 |
| Scheduler | None | Halve after five non-improving epochs; floor 0.00001 |
| Stopping | Epoch limit | 20 epochs without substantial improvement |
| Substantial improvement | Not used to stop | 1 second bearing MAE; 0.0001 filter NLL |

`ReduceLROnPlateau` uses `patience=4`, because PyTorch reduces when its bad-epoch count **exceeds** patience. AdamW, batch 32, dropout 0.1, weight decay 0.0001 and gradient norm limit 1 remain fixed. The original protocol is explicitly labelled legacy and keeps its numerical loss on resume.

Sampling variants are the existing equal-object sampler with replacement and a full shuffled pass without replacement. Full-pass weights have mean one and equal total weight per object. The weighted batch sum is divided by the configured batch size, including the final short batch, to preserve each window's coefficient. The optional bearing objective assigns half each object's weight to all its windows and half to its final 30 minutes; objects without eligible final windows use their whole history.

Weibull likelihood and sensitive expressions use float64. Every run records per-epoch loss quantiles, parameter ranges, gradient norms, learning rate, window coverage and stopping state. Nonfinite outputs, likelihoods or gradients stop the run with equipment/window identifiers.

## Features and memory

Both recipes keep a 20-measurement model window. `base_v1` preserves the existing input contract. `degradation_v1` additionally uses:

- Bearings, each channel: log-RMS slopes over 5/20 points, log-peak and kurtosis slopes over 20 points, log-RMS change from the first-five-point segment median.
- Filters: pressure slopes over 5/20 points, 20-point pressure change, headroom to 600 Pa, and dust-feed integration with the previous measurement's feed and elapsed internal time.

Slopes use actual timestamps. Recipes process the object's causal prefix **before** cutting model windows. One-point slopes are zero; short prefixes use only available points. Gaps reset the initial reference and accumulated dust. Future measurements cannot change an earlier feature. The recipe name, parameters, ordered feature schema and hash are saved with preprocessing.

GRU/LSTM and sampled reservoirs reset state for each model window. Full MaleCNS preserves its continuous state and 20-point warmup. Feature history can extend beyond the model window; the report labels both histories.

## Reservoir readouts

Bearing readouts compare linear RUL and `log1p(RUL / 60)`, with ridge regularization `0.0001, 0.001, 0.01, 0.1, 1`. Selection uses three train-only folds by bearing instance, refitting preprocessing in every fold. Fly and Random use the same grid and budget. Full CNS scores its actual smoothed user forecast on the same final-30-minute metric. Negative raw outputs and clamped final outputs are recorded separately.

Fixed-state caches bind to admitted data, preprocessing, the actual recurrent/input weights and bias, windows and memory mode. Full CNS caches pooled readout trajectories **after** computing every neuron's state. Cache hashes protect stored arrays; readout parameters do not enter the state-cache key.

## Bounded study and artifacts

The sequential worker screens GRU recipes, checks two finalists on grouped train folds, trains the nine main architectures, and repeats the two best validation architectures per dataset with seeds 43/44. Filter folds each contain one observed train failure; censored objects are balanced deterministically by dust type. Mean CV scores weight equipment equally. Test begins after candidates and warning settings are frozen, retaining the `exploratory_reused_holdout` label.

The manifest retains parents, failed/superseded work, decisions, exact run/evaluation IDs and immutable data versions. Additional seeds measure initialization sensitivity, not additional independent failures. The adaptive pilot changes scheduler and stopping together; its result does not isolate those two effects from each other.

```bash
.venv/bin/python -m pdm training-study
.venv/bin/python -m pdm stop
.venv/bin/python -m pdm training-study --resume-study <study_id>

.venv/bin/python -m pdm train --dataset bearings --arch gru --protocol adaptive \
  --sampling full_pass --feature-recipe degradation_v1 --near-weight 0.5 --device cpu
```

Artifacts live in `runs/training_studies/<study_id>/`: manifest, logs, screening table, exact evaluation references, baseline forecasts, result CSV/JSON, seed dispersion and a Markdown report. Individual run directories contain preprocessing, training protocol, history, weights and diagnostics. Checkpoints save optimizer, scheduler, random generators and next sampler epoch; an interrupted partial epoch restarts from the last completed epoch. Old checkpoints cannot resume with a changed protocol or input identity. Full CNS resumes through completed trajectory caches and reruns its deterministic readout selection.

Sources: [PyTorch plateau scheduler](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html), [grouped cross-validation and preprocessing](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data).
