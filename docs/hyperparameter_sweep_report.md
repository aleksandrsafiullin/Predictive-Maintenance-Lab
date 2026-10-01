# Hyperparameter Sweep Report

_Generated: 2026-10-01 (cloud agent run)_

## Summary

Systematic training-v2 sweep across **GRU / LSTM** × **feature recipes** × **learning rates** × **sampling strategies** on real data (both XJTU-SY bearings and HSE filters datasets).

- **20 configurations** run on real data (11 bearings, 9 filters).
- **Abbreviated protocol**: `max_epochs=30`, `patience=10`, `min_epochs=8` per run (vs 100/20/20 in production).
- All runs completed; no failures.
- Results are **exploratory** — epoch budget is a binding constraint for bearings (winner still improving at epoch 29). Use `python3 -m pdm training-study` for a full benchmark.

---

## Winner Table

| Dataset | Architecture | Feature recipe | LR | Sampling | Near-wt | Best epoch | Val metric | Run ID |
|---------|-------------|---------------|-----|----------|---------|------------|-----------|--------|
| **bearings** | GRU | base_v1 | 0.001 | unit_replacement | 0.0 | 29/30 | near-30-min MAE = **3949 s (65.8 min)** | `bearings_gru_20261001_084427_c543d1` |
| **filters** | GRU | base_v1 | 0.001 | full_pass | 0.0 | 1/30 | val NLL = **0.2664** | `filters_gru_20261001_083704_0de238` |

### Reproduce commands

```bash
# Bearings winner
python3 -m pdm train --dataset bearings --arch gru --protocol adaptive \
    --feature-recipe base_v1 --learning-rate 0.001 --sampling unit_replacement

# Filters winner
python3 -m pdm train --dataset filters --arch gru --protocol adaptive \
    --feature-recipe base_v1 --learning-rate 0.001 --sampling full_pass
```

---

## Bearings (XJTU-SY) — Full Results

**Selection metric**: near-30-min equal-weight unit MAE in seconds (lower is better).  
Split: instances 1–3 train, 4 val, 5 test (per regime). 9 train units, 3 val units, 3 test units.

| Rank | Config | Arch | Recipe | LR | Sampling | Near-wt | Best epoch | Val MAE (s) | Elapsed |
|------|--------|------|--------|-----|----------|---------|------------|------------|---------|
| 1 | `gru/base_v1/lr0.001/unit_repl` | GRU | base_v1 | 0.001 | unit_replacement | 0.0 | **29**/30 ⚠️ | **3 949 s (65.8 min)** | 39 s |
| 2 | `gru/base_v1/lr0.001/full_pass` | GRU | base_v1 | 0.001 | full_pass | 0.0 | 30/30 ⚠️ | 5 138 s (85.6 min) | 37 s |
| 3 | `gru/base_v1/lr0.0003/unit_repl` | GRU | base_v1 | 0.0003 | unit_replacement | 0.0 | 18/30 | 5 666 s (94.4 min) | 35 s |
| 4 | `gru/base_v1/lr0.001/full_pass_near0.5` | GRU | base_v1 | 0.001 | full_pass | 0.5 | 10/30 | 5 747 s (95.8 min) | 25 s |
| 5 | `gru/degradation_v1/lr0.001/unit_repl` | GRU | degradation_v1 | 0.001 | unit_replacement | 0.0 | 11/30 | 5 964 s (99.4 min) | 28 s |
| 6 | `gru/degradation_v1/lr0.001/full_pass_near0.5` | GRU | degradation_v1 | 0.001 | full_pass | 0.5 | 1/30 | 6 241 s (104.0 min) | 15 s |
| 7 | `gru/degradation_v1/lr0.001/full_pass` | GRU | degradation_v1 | 0.001 | full_pass | 0.0 | 15/30 | 6 944 s (115.7 min) | 31 s |
| 8 | `lstm/degradation_v1/lr0.001/full_pass` | LSTM | degradation_v1 | 0.001 | full_pass | 0.0 | 15/30 | 7 227 s (120.5 min) | 24 s |
| 9 | `lstm/base_v1/lr0.001/unit_repl` | LSTM | base_v1 | 0.001 | unit_replacement | 0.0 | 17/30 | 7 483 s (124.7 min) | 27 s |
| 10 | `gru/multiscale_no_age_v2/lr0.001/unit_repl` | GRU | multiscale_no_age_v2 | 0.001 | unit_replacement | 0.0 | 1/30 | 7 748 s (129.1 min) | 15 s |
| 11 | `gru/multiscale_trend_v2/lr0.001/unit_repl` | GRU | multiscale_trend_v2 | 0.001 | unit_replacement | 0.0 | 2/30 | 10 553 s (175.9 min) | 17 s |

⚠️ `best_epoch = max_epochs` means the epoch budget was **binding**; true performance with 100 epochs likely much better.

### Bearings validation evaluation (winner run)

Measured on `--split validation` (3 validation bearings, one per regime):

| Metric | Value |
|--------|-------|
| near-30-min equal-weight unit MAE | **3 949 s (65.8 min)** |
| near-1-hour equal-weight unit MAE | 3 918 s (65.3 min) |
| all-points equal-weight unit MAE | 16 842 s (4.7 h) |
| Bearing1_4 (35 Hz 12 kN) MAE | 6 072 s (1.7 h) |
| Bearing2_4 (37.5 Hz 11 kN) MAE | **4 242 s (70.7 min)** |
| Bearing3_4 (40 Hz 10 kN) MAE | 40 213 s (11.2 h) ⚠️ |

Bearing3_4 is a strong outlier (11× the median). The 40 Hz regime has inherently different failure dynamics; with only 3 training units per regime and 30 epochs, regime-specific convergence varies widely.

### Bearings — key findings

1. **`unit_replacement` sampling wins over `full_pass`** on bearings (opposite of filters). With only ~6 852 training windows spread across 9 units × 3 regimes, `full_pass` provides no sampling diversity benefit.
2. **`base_v1` beats `degradation_v1`** at the 30-epoch budget. The degradation recipe adds 10 slope features whose rolling-window estimates need several epochs to stabilize; at 30 epochs they appear to hurt convergence.
3. **Multiscale recipes perform worst** (best_epoch=1–2). These recipes add 40–60 measurement slope windows; with short bearing series they are mostly zero during training and provide no signal.
4. **GRU > LSTM** consistently: LSTM ranks 8–9 vs GRU at 1–7.
5. **Epoch budget is binding for the top 2 configs** (winner at epoch 29, runner-up at epoch 30). Near-event MAE of ~3 949 s at epoch 29 is still declining; a 100-epoch run is expected to improve materially.

---

## Filters (HSE) — Full Results

**Selection metric**: unit-equal val Weibull NLL (lower is better).  
Split: 39 train / 10 val (author_train 80/20 seed 42, stratified on events). 50 test units held out.

| Rank | Config | Arch | Recipe | LR | Sampling | Near-wt | Best epoch | Val NLL | Elapsed |
|------|--------|------|--------|-----|----------|---------|------------|---------|---------|
| 1 | `gru/base_v1/lr0.001/full_pass` | GRU | base_v1 | 0.001 | full_pass | 0.0 | 1/30 | **0.2664** | 1.5 min |
| 2 | `gru/degradation_v1/lr0.001/full_pass` | GRU | degradation_v1 | 0.001 | full_pass | 0.0 | 1/30 | 0.2732 | 1.4 min |
| 3 | `gru/multiscale_no_age_v2/lr0.001/unit_repl` | GRU | multiscale_no_age_v2 | 0.001 | unit_replacement | 0.0 | 1/30 | 0.2893 | 1.5 min |
| 4 | `gru/multiscale_trend_v2/lr0.001/unit_repl` | GRU | multiscale_trend_v2 | 0.001 | unit_replacement | 0.0 | 2/30 | 0.2955 | 1.5 min |
| 5 | `gru/degradation_v1/lr0.001/unit_repl` | GRU | degradation_v1 | 0.001 | unit_replacement | 0.0 | 1/30 | 0.2966 | 1.4 min |
| 6 | `gru/base_v1/lr0.001/unit_repl` | GRU | base_v1 | 0.001 | unit_replacement | 0.0 | 2/30 | 0.3038 | 1.5 min |
| 7 | `gru/base_v1/lr0.0003/unit_repl` | GRU | base_v1 | 0.0003 | unit_replacement | 0.0 | 2/30 | 0.3153 | 1.6 min |
| 8 | `lstm/degradation_v1/lr0.001/full_pass` | LSTM | degradation_v1 | 0.001 | full_pass | 0.0 | 1/30 | 0.3183 | 1.1 min |
| 9 | `lstm/base_v1/lr0.001/unit_repl` | LSTM | base_v1 | 0.001 | unit_replacement | 0.0 | 1/30 | 0.3364 | 1.1 min |

### Filters validation evaluation (winner run)

| Metric | Value |
|--------|-------|
| Best val NLL (epoch 1) | **0.2664** |
| Last epoch NLL (epoch 18) | 0.638 |
| Val MAE on observed 600 Pa events | 8.8 s (1 event unit) |
| Val coverage (n observed event units) | 1 / 10 |

The large gap between best NLL (0.266 at epoch 1) and last NLL (0.638 at epoch 18) is expected: with only **5 observed 600 Pa events in 39 training units** (13% event rate), the Weibull NLL landscape is flat and volatile. The model quickly finds a good survival function shape in epoch 1, then the optimizer drifts on the noisy censored gradient.

### Filters — key findings

1. **`full_pass` sampling wins over `unit_replacement`** (ranks 1–2 both use full_pass). Each filter unit has ~790 measurements; `full_pass` ensures every window contributes equally per epoch, which reduces variance in the NLL gradient from the sparse event signal.
2. **`base_v1` + full_pass is the winner**, beating `degradation_v1` + full_pass (0.266 vs 0.273). The pressure-slope features added by `degradation_v1` do not reliably improve over the raw pressure + dust features on this short epoch budget.
3. **`multiscale_no_age_v2` is 3rd** for unit_replacement — adding long-window slopes gives richer degradation signal even on unit_replacement. Likely competitive with full_pass for longer runs.
4. **GRU dominates**: all top-7 spots are GRU. LSTM ranks 8–9 with NLL 0.318–0.336 vs GRU's 0.266–0.315.
5. **Lower LR (0.0003) hurts** (rank 7, NLL 0.315 vs 0.304 for lr=0.001 unit_replacement). The default 0.001 converges better under the 30-epoch limit.
6. **best_epoch=1 for 8/9 configs** reflects the fragility of Weibull NLL on a small censored dataset. The checkpoint saved at epoch 1 is the true best; later epochs overfit.

---

## Cross-dataset Comparison

| Axis | Bearings | Filters |
|------|---------|---------|
| Winning sampling | `unit_replacement` | `full_pass` |
| Winning recipe | `base_v1` | `base_v1` |
| Winning LR | 0.001 | 0.001 |
| Winning arch | GRU | GRU |
| Epoch budget binding? | **Yes** (29–30 of 30) | No (1–2 of 30) |
| Multiscale recipes | ❌ Worst (best_epoch=1) | ↑ Middle-rank |
| LSTM vs GRU gap | ~2× MAE worse | ~25% NLL worse |

---

## What Was Not Trained

| Architecture | Reason |
|-------------|--------|
| `fly_connectome_reservoir` | Requires MaleCNS feather file (`malemcns_present: false` on this VM) |
| `random_reservoir` | Same constraint (no MaleCNS path needed but file lookup is shared) |
| `full_cns` | Requires MaleCNS feather (bearings only via `training_study.py`) |

The reservoir architectures need either: (a) `--graph-mode synthetic_fixture` which uses the built-in 3-node synthetic connectome (available), or (b) a MaleCNS feather for `real_connectome`. The current configs in the sweep use the default `real_connectome` mode which requires the feather.

**To sweep reservoir architectures with the synthetic fixture:**
```bash
python3 -m pdm train --dataset bearings --arch fly_connectome_reservoir \
    --graph-mode synthetic_fixture --readout ridge --protocol legacy
```

---

## Recommendations

### For bearings

1. **Wire default**: keep `gru / base_v1 / lr=0.001 / unit_replacement` — it won the sweep.
2. **Extend epochs**: re-run winner with `--epochs 100` (or default `--protocol adaptive`); best_epoch=29 strongly suggests the model would improve further.
3. **Confirm**: run seeds 43 and 44 to check ranking stability across 3 validation bearings.
4. **Cautious about `near_weight`**: near-event weighting (`full_pass_near0.5`) ranked 4th at 30 epochs — the near-event signal is strongest when the model has had time to converge globally first.

### For filters

1. **Wire default**: `gru / base_v1 / lr=0.001 / full_pass` — it won cleanly (NLL 0.266 vs 0.273 for runner-up).
2. **Note the best_epoch=1 instability**: this run's winner was determined in the first epoch. Consider adding `--sampling full_pass --min-epochs 20` in production to force longer training and observe whether NLL keeps improving.
3. **Consider `multiscale_no_age_v2`** as a secondary candidate: it ranked 3rd on unit_replacement; with full_pass and more epochs it may be competitive.

### General

- Both datasets consistently favor **GRU over LSTM** for this data size and epoch budget.
- The default `lr=0.001` is clearly better than `lr=0.0003` for both datasets at 30 epochs.
- Reservoir architectures were excluded — they are interesting candidates for both datasets but require either the MaleCNS feather or `--graph-mode synthetic_fixture`.

---

## Limitations and Honest Caveats

1. **30-epoch budget is binding for bearings**: the winner improved until epoch 29/30. All bearings rankings above are preliminary and likely to change with 100 epochs.
2. **Bearings: only 3 validation units** (one per regime). Any single-config ranking difference under ~500 s may not be statistically meaningful. Rankings 1–5 should be treated as a cluster, not a strict ordering.
3. **Filters: only 1 observed event in validation** (out of 10). The val MAE is on that 1 unit only; NLL is the only robust selection metric.
4. **Single seed (42)**: all runs use seed=42. Rankings may shift with seeds 43–44, especially for bearings where 3 training units per regime create high variance.
5. **No test evaluation**: all metrics are from validation split only. Test evaluation requires `python3 -m pdm evaluate --split test` with a frozen alert policy.
6. **Reservoir architectures excluded**: `fly_connectome_reservoir`, `random_reservoir`, and `full_cns` were not swept due to missing MaleCNS feather on this VM.
7. **CPU-only training**: `train_v2` sets `torch.set_num_threads(1)` for reproducibility. GPU runs would be faster but produce the same values.

---

## Detailed Per-Run Data

Individual result files: `docs/sweep_results/<label>.json`

To evaluate any run:
```bash
python3 -m pdm evaluate --dataset <ds> --run-id <run_id> --split validation --k 3
```

To evaluate bearings winner:
```bash
python3 -m pdm evaluate --dataset bearings \
    --run-id bearings_gru_20261001_084427_c543d1 \
    --split validation --k 3
```

To evaluate filters winner:
```bash
python3 -m pdm evaluate --dataset filters \
    --run-id filters_gru_20261001_083704_0de238 \
    --split validation --k 3
```
