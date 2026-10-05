# Hyperparameter Sweep Report

_Generated: 2026-10-01 (cloud agent run); interpretation corrected: 2026-10-02._

These results concern the historical RUL/Weibull training task. They do not measure first-RED v2 warning quality or establish architecture defaults for that separate task.

## Summary

Systematic training-v2 sweep across **GRU / LSTM** × **feature recipes** × **learning rates** × **sampling strategies** on real data (both XJTU-SY bearings and HSE filters datasets).

- **20 configurations** run on real data (11 bearings, 9 filters).
- **Abbreviated protocol**: `max_epochs=30`, `patience=10`, `min_epochs=8` per run (vs the longer configured research budget of 100/20/20).
- All runs completed; no failures.
- Results are **exploratory** — the best bearings checkpoints were near the 30-epoch cap, so longer-budget behavior remains unmeasured. Use `python3 -m pdm training-study` for a full benchmark.

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

⚠️ A best checkpoint near the epoch cap warrants a longer-budget experiment. It does not guarantee improvement at 100 epochs; later training can plateau or worsen validation performance.

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

Bearing3_4 has the largest reported all-points error. This sweep alone cannot attribute that difference to regime dynamics, convergence or feature quality.

### Bearings — key findings

1. The measured `gru/base_v1/lr0.001/unit_replacement` configuration had the lowest validation score among the tested candidates. Its sampling advantage is specific to this sweep; no variance mechanism was measured.
2. `base_v1` scored better than the tested `degradation_v1` recipes under this budget. Rolling slopes are calculated from the observed prefix and do **not** stabilize through training epochs. Their usefulness depends on causal history, scaling and model fit; the sweep does not identify the cause of the ranking.
3. The two tested multiscale candidates ranked last. No feature sparsity analysis was reported here to explain that outcome.
4. Tested GRU candidates scored better than tested LSTM candidates. A general architectural advantage requires matched budgets, additional seeds and physical-unit folds.
5. Top checkpoints occurred at epochs 29 and 30. A 100-epoch run is an untested comparison, not a promised improvement.

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

The later validation NLL was worse than the selected early checkpoint. Sparse observed events limit interpretation; this table does not establish a particular optimizer or censoring-gradient mechanism. Event counts in this historical report must not be substituted for the RED-rule audit in the v2 report.

### Filters — key findings

1. `gru/base_v1/lr0.001/full_pass` had the lowest measured NLL in this sweep; `degradation_v1/full_pass` was next. The small difference needs replication.
2. Tested multiscale candidates did not beat that configuration. Longer budgets or alternate sampling remain untested here.
3. Tested GRU candidates ranked above tested LSTM candidates. Seed and grouped-fold replication are required before generalizing the ranking.
4. The tested lower learning rate scored worse in its sampled comparison; this does not establish a universal learning-rate preference.
5. The table shows best epoch 1 for **7/9** configurations and epoch 2 for the other two. It supports selecting the recorded best checkpoint, not requiring more epochs to improve quality.

---

## Cross-dataset Comparison

| Axis | Bearings | Filters |
|------|---------|---------|
| Winning sampling | `unit_replacement` | `full_pass` |
| Winning recipe | `base_v1` | `base_v1` |
| Winning LR | 0.001 | 0.001 |
| Winning arch | GRU | GRU |
| Best checkpoint vs cap | Near cap (29–30 of 30) | Early (1–2 of 30) |
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

## Follow-up experiments

Treat the recorded winners as candidates for this historical task. Compare longer budgets using Validation checkpoint selection, repeat multiple seeds, and use physical-unit development folds to assess ranking stability. Match input recipes and compute budgets when comparing GRU and LSTM. Preserve Test from model and threshold selection. More epochs, more features or forcing a minimum epoch count do not guarantee better forecasts.

Reservoir architectures were excluded on the cloud VM. A synthetic fixture can test software contracts; it cannot establish official MaleCNS performance or substitute for the real graph in the v2 comparison.

---

## Limitations and Honest Caveats

1. **30-epoch budget is binding for bearings**: the winner improved until epoch 29/30. All bearings rankings above are preliminary; their stability under 100 epochs is unknown.
2. **Bearings: only 3 validation units** (one per regime). No confidence interval or significance analysis establishes a reliable separation between candidates.
3. **Filters: only 1 observed event in validation** (out of 10). The val MAE is on that 1 unit only; NLL includes censored follow-up, but one event does not establish robust event-time quality.
4. **Single seed (42)**: all runs use seed=42. Rankings may shift with seeds 43–44, especially for bearings where 3 training units per regime create high variance.
5. **No test evaluation**: all metrics are from validation split only. Test evaluation requires `python3 -m pdm evaluate --split test` with a frozen alert policy.
6. **Reservoir architectures excluded**: `fly_connectome_reservoir`, `random_reservoir`, and `full_cns` were not swept due to missing MaleCNS feather on this VM.
7. **CPU-only training**: `train_v2` sets `torch.set_num_threads(1)` for reproducibility. GPU speed and numerical agreement were not checked.

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
