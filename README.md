# Predictive Maintenance Lab

Local lab for **condition monitoring, sensor forecasting and remaining useful life (RUL)** on two public datasets:

| Dataset | Task | Split (units) |
|---|---|---|
| **Bearings / XJTU-SY** | Vibration condition, sensor forecast and RUL | 9 train / 3 validation / 3 test |
| **Filters / HSE** | Differential-pressure forecast and time to laboratory 600 Pa endpoint (censored survival) | Original 40 / 10 / 50; admission retains 39 / 10 / 50 |

**Event models:** GRU, LSTM, Fly and matched Random reservoirs (1,000 nodes), plus Full MaleCNS for bearings (166,700 neurons; 25,582,938 directed connections). Reservoir activity comes from actual model computation. Fly / neural activity is an optional experimental view; condition monitoring does not require a connectome. **Sensor forecasts:** persistence, causal local trend and multi-horizon quantile boosting.

English UI (Streamlit). Heavy jobs run in a background worker. Repo: https://github.com/aleksandrsafiullin/Predictive-Maintenance-Lab

---

## Condition monitoring · 17 September 2026

With a frozen monitoring bundle, **Model Report → Condition & Forecast** is the main view: observation quality, regime reference, actual sensor forecasts and deterministic condition zones. RUL and action-timing diagnostics remain available separately. Old models, snapshots and the common inference contract are preserved.

- Observation quality, model applicability and missing healthy reference have separate reasons. Unavailable assessments stay visible rather than becoming normal states.
- Fixed histories of 20/40/60 and genuinely trained variable history 20–60 use saved policies, contiguous measurements and actual elapsed duration. Full CNS retains continuous memory.
- Sensor models predict measured future signal targets; forecast lines are not constructed from RUL or forced to cross a limit.
- State rules include confirmation, recovery, hysteresis, cooldown and a critical latch. Acknowledgement does not resolve an alert. Action timing requires compatible units, endpoint and validated calibration.

The bounded condition study executed **48 fits**, including horizon/quantile submodels, controls and retained failed-recipe experiments. Train equipment folds selected candidates; validation provided stopping/diagnostics. Four final validation/test replays were completed. Test remains an **exploratory reused holdout** and was not used for selection. Verification recorded **441 passing tests**, clean Ruff, runtime/export parity and desktop/mobile browser checks.

| Selected sensor model | Test horizons | Unit-balanced test MAE | Test condition assessment available |
|---|---|---|---:|
| Bearings: causal local trend | 300 / 900 physical seconds | 0.5533 / 1.6200 g | 88.71% |
| Filters: quantile boosting | 30 / 90 dataset-internal units | 3.1417 / 4.5153 Pa | 2.92% |

**Laboratory only; useful maintenance timing is not established.** Filter condition is unavailable for **97.08%** of test measurements, mainly because the provisional regime reference is missing. Sensor forecast availability is a different measure: approximately 94% of eligible filter targets. Broad pointwise bands are uncalibrated; independent probability/timing calibration and verified HSE Time/RUL physical units are unavailable. Additional corrected filter feature ablations were blocked by the exhausted 48-fit budget. Faulty multiscale v1 experiments remain quarantined; new experiments use corrected v2 recipes.

See the [implementation report with results, commands and limitations](reports/condition_monitoring_implementation_report.md) and [monitoring guide](docs/condition_monitoring.md). Data, checkpoints, bundles, screenshots and other generated artifacts remain local; report links into `data/`, `runs/` and `output/` require that local workspace and are not included in a Git clone.

## Previous training-v2 results · 16 September 2026

The completed training improvement study contains **42 real training runs**, 52 saved validation/test evaluations and four model comparisons. All nine main models cover the shared eligible prediction clock after warmup. Model selection uses validation; test is a **previously explored holdout**, reported separately.

| Model | Bearings validation MAE, min | Bearings test MAE, min | Filters validation NLL | Filters test MAE, internal s |
|---|---:|---:|---:|---:|
| GRU | **62.71** | 51.09 | **0.6493** | 3911.88 |
| LSTM | 127.00 | 55.54 | 0.7038 | 5868.61 |
| Fly reservoir, 1,000 | 151.91 | 14.99 | 0.7987 | 1591.36 |
| Random reservoir, 1,000 | 152.32 | 14.90 | 0.7979 | 1597.18 |
| Full MaleCNS | 133.82 | 20.17 | — | — |

Seed 42; lower is better within each column. Bearing MAE covers the final 30 minutes, with equal weight per bearing. Filter validation is equal-equipment survival negative log-likelihood (NLL), including censored series; filter test uses official RUL at each prefix endpoint. Filter time uses the unverified conversion `Time × 60`.

**Progress is real, but reliable maintenance timing is not established.** No bearing model meets the ≤10-minute MAE target or the warning targets (≥90% timely recall and ≥80% useful-episode precision). GRU leads validation on both datasets, but its bearing test error worsened from 30.79 to 51.09 minutes. Filter GRU improves validation NLL and test error; its successful validation warning is based on only **one independent failure**. Validation contains three bearing failures; extra seeds do not increase that count.

What the experiments established:

- Aligning bearing checkpoint selection with the final-30-minute metric accounts for most GRU improvement. Continuing the same trajectory to 100 epochs adds 7.56% improvement; the best checkpoint is epoch 50.
- Extra epochs alone do not help filter GRU. Its selected recipe uses adaptive learning rate, a complete pass over windows and causal degradation features.
- New features and additional near-failure loss weight did not help the tested bearing GRU. Negative results are retained.
- Bearing GRU varies substantially across seeds. Lower validation error alone does not establish useful warnings or generalization.

See the [full results, controls, seed variation and acceptance evidence](reports/training_study_20260916.md) and [training protocol](docs/training_protocol.md). That study recorded **369 passing tests**, clean Ruff, and desktop/mobile browser verification of all nine main reports. Its 42 training runs and metrics describe the earlier study, separately from the condition-monitoring results above.

---

## What you need

- macOS, Linux, or Windows 10/11
- Python **3.11 or 3.12**
- PyTorch: **CPU** everywhere; **CUDA** on NVIDIA Windows/Linux; **MPS** only on Apple
- ~**8+ GB** free disk for filters + prepared data; bearings zip alone is ~**5 GB**
- Optional: Kaggle credentials for filters download (`kagglehub`)
- Optional: ~**1 GB** MaleCNS feather for real fly connectome (not required for GRU/LSTM)

`setup.sh` / `run.sh` are tested on **macOS arm64**. The `.ps1` scripts are written for Windows PowerShell; they are not executed on the macOS development machine.

Do **not** commit `data/`, `runs/`, or `.venv/`.

---

## 1. Clone and install

```bash
git clone https://github.com/aleksandrsafiullin/Predictive-Maintenance-Lab.git
cd Predictive-Maintenance-Lab

./scripts/setup.sh
```

```powershell
# Windows PowerShell (Win10/11). If scripts are blocked:
# Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\scripts\setup.ps1
```

`setup.sh` / `setup.ps1` create `.venv`, install `requirements-lock.txt`, the editable package, and run `pdm doctor`.

Manual equivalent:

```bash
python3.12 -m venv .venv   # or python3.11
.venv/bin/python -m pip install --upgrade pip wheel
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install -e .
.venv/bin/python -m pdm doctor
```

```powershell
py -3.12 -m venv .venv   # or py -3.11
.\.venv\Scripts\python.exe -m pip install --upgrade pip wheel
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pdm doctor
```

On Windows, use `.\.venv\Scripts\python.exe` anywhere this README shows `.venv/bin/python`.

---

## 2. Download training data

### Filters (Kaggle)

Needs network access. First time, configure Kaggle for `kagglehub` (API token): https://www.kaggle.com/docs/api

```bash
.venv/bin/python -m pdm download --dataset filters
```

Or copy a local extract:

```bash
.venv/bin/python -m pdm download --dataset filters --local-path /path/to/filters_folder
```

Dataset: [prognosticshse/preventive-to-predicitve-maintenance](https://www.kaggle.com/datasets/prognosticshse/preventive-to-predicitve-maintenance) (slug typo `predicitve` is intentional).

### Bearings (XJTU-SY, ~5 GB)

```bash
.venv/bin/python -m pdm download --dataset bearings
```

Falls back to a public Hugging Face mirror of the author zip if needed. Author page: https://biaowang.tech/xjtu-sy-bearing-datasets/

Or pass a local zip/folder:

```bash
.venv/bin/python -m pdm download --dataset bearings --local-path /path/to/XJTU-SY_Bearing_Datasets.zip
```

### Both

```bash
.venv/bin/python -m pdm download --dataset all
```

Raw files land under `data/raw/{filters,bearings}/`.

### Optional: MaleCNS connectome (~1 GB)

Required for the full reference matrix and training improvement study, including Full CNS and real Fly/Random comparisons. Individual GRU/LSTM runs do not need it.

```bash
mkdir -p data/raw/connectome
curl -L --fail -o data/raw/connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather \
  "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather"
```

```powershell
New-Item -ItemType Directory -Force -Path data\raw\connectome | Out-Null
Invoke-WebRequest -Uri "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather" `
  -OutFile data\raw\connectome\connectome-weights-male-cns-v1.0-minconf-0.5.feather
```

Details: [docs/fly_connectome.md](docs/fly_connectome.md). Without the file, connectome runs use a **synthetic** graph labeled as non-biological.

---

## 3. Prepare processed features

Rebuilds parquet + split manifests. After connectome/gap-rule updates you must re-prepare before train/eval.

```bash
.venv/bin/python -m pdm prepare --dataset filters
.venv/bin/python -m pdm prepare --dataset bearings
```

Output: `data/processed/{filters,bearings}/` (versioned). Expect `gap_rule_version=causal_v1` in the fingerprint.

Optional inspect:

```bash
.venv/bin/python -m pdm inspect --dataset filters
.venv/bin/python -m pdm inspect --dataset bearings
```

---

## 4. Train

### Smoke (sanity only — not a quality claim)

```bash
.venv/bin/python -m pdm train --dataset filters --arch gru --epochs 3 --smoke --max-windows-per-unit 32
.venv/bin/python -m pdm train --dataset bearings --arch gru --epochs 3 --smoke --max-windows-per-unit 32
```

### GRU with the selected v2 recipes

```bash
# Bearings: 100 epochs, fixed learning rate, original features and sampler.
.venv/bin/python -m pdm train --dataset bearings --arch gru \
  --protocol diagnostic --learning-rate 0.001 --feature-recipe base_v1 \
  --sampling unit_replacement --device cpu

# Filters: adaptive schedule, complete window passes and degradation features.
.venv/bin/python -m pdm train --dataset filters --arch gru \
  --protocol adaptive --learning-rate 0.0003 --feature-recipe degradation_v1 \
  --sampling full_pass --device cpu
```

Both recipes use all admitted windows, history 20 and seed 42 by default. Diagnostic training completes 100 epochs and retains the best validation checkpoint. Adaptive training runs 20–100 epochs, halves the learning rate after five non-improving epochs, and stops after 20 epochs without substantial improvement. CPU execution supports reproducible continuation and float64 survival calculations.

These commands reproduce the earlier training-v2 recipes. For new history experiments, add `--history-mode fixed_20`, `fixed_40`, `fixed_60` or `variable_20_60`. Variable history is trained with different valid lengths, not enabled only at inference. See the [history 20–60 protocol](docs/history_20_60_protocol.md) and the condition-study workflow below for comparisons on matching clocks.

LSTM uses the same flags with `--arch lstm`. The default CLI protocol remains `legacy`; specify `--protocol` to use v2. Legacy `--epochs 30 --max-windows-per-unit 0` makes all windows eligible but does not imply a full pass without replacement. `--smoke` is incompatible with v2.

### Fly connectome reservoir (experimental)

Synthetic graph (demo / CI). Synthetic test graph — not a biological connectome. `--smoke` is not a quality benchmark.

```bash
.venv/bin/python -m pdm train --dataset bearings --arch fly_connectome_reservoir --smoke --n-nodes 8
```

Equivalent with explicit graph mode:

```bash
.venv/bin/python -m pdm train --dataset bearings --arch fly_connectome_reservoir \
  --graph-mode synthetic_fixture --n-nodes 8 --smoke
```

Full MaleCNS model report: train the complete classified MaleCNS from scratch
(166,700 neurons; 25,582,938 directed pairs; no sampling):

```bash
.venv/bin/python scripts/train_brain_forecast.py
```

See [Full CNS data, morphology, training and limitations](docs/full_cns.md).

Real subgraph experiments for comparisons (the full study uses 1,000 nodes):

```bash
.venv/bin/python -m pdm train --dataset bearings --arch fly_connectome_reservoir \
  --graph-mode real_connectome --n-nodes 1000 --epochs 30 --max-windows-per-unit 0

.venv/bin/python -m pdm train --dataset filters --arch fly_connectome_reservoir \
  --graph-mode real_connectome --n-nodes 1000 --epochs 30 --max-windows-per-unit 0
```

Matched control topology: `--arch random_reservoir`.

Runs are stored under `runs/{filters,bearings}/<run_id>/`. Note the printed `run_id`.

---

## 5. Evaluate and check results

### CLI evaluate

Research validation (H/K flags — does **not** write `alert_policy.json`):

```bash
.venv/bin/python -m pdm evaluate --dataset bearings --run-id <run_id> --split validation \
  --horizon-s 1800 --k 3 --min-action-lead-s 900
```

Freeze alert policy from the UI (**Model Report → Evaluation settings → Validation → Freeze alert policy**), then frozen test:

```bash
.venv/bin/python -m pdm evaluate --dataset bearings --run-id <run_id> --split test
```

List evaluations:

```bash
.venv/bin/python -m pdm evaluate --dataset bearings --run-id <run_id> --list
```

Artifacts: `runs/.../evaluations/<eval_id>/{metrics.json,predictions.csv,alerts.csv,...}`.

### UI

```bash
./scripts/run.sh
```

```powershell
.\scripts\run.ps1
```

```bash
# or
.venv/bin/python -m pdm app
```

Open **http://127.0.0.1:8501** (localhost only).

Pages:

1. **Data Quality** — admission audit, reasons, retained censoring, signal and split counts, monitoring units/endpoints and reference status.
2. **Training** — full matrix, individual training, condition/sensor studies, progress, stop and resume.
3. **Model Report** — Condition & Forecast for frozen bundles, with Start/Pause/Next/Reset, quality reasons, sensor forecasts and alert episodes. Event/action diagnostics, legacy model replay and evaluation settings remain available. Fly/neural details live in **Experimental / Model activity**. Without a monitoring bundle, the legacy entry point is retained.
4. **Compare Models** — separate event, sensor and monitoring results, explicit validation/test artifacts, common time points, unit-balanced ranking, saved selections, CSV and JSON.

### Condition and sensor forecast study

Use admitted prepared data. These profiles run through the existing training-v2 engine and worker; the default GRU condition studies do not require a connectome or the earlier reference matrix. First inspect the plan without fitting:

```bash
.venv/bin/python -m pdm condition-study --config configs/condition_bearings.yaml --plan
.venv/bin/python -m pdm condition-study --config configs/condition_filters.yaml --plan
```

For a **new** experiment budget, start one dataset at a time and wait for completion in Training:

```bash
.venv/bin/python -m pdm condition-study --config configs/condition_bearings.yaml --run
# After completion, run the other profile:
.venv/bin/python -m pdm condition-study --config configs/condition_filters.yaml --run
# Stop/resume the corresponding study when needed:
.venv/bin/python -m pdm stop
.venv/bin/python -m pdm condition-study --resume-study STUDY_ID
```

Each profile plans 22 fits (44 paired), counting folds, seeds, horizons and quantiles. The recorded implementation study already consumed its 48-fit cap; starting a new study spends additional compute and is not a replay of its saved results.

After a study completes, freeze the selected configuration and evaluate its bundle. Replace placeholders with the IDs printed by the commands/UI. Run evaluations sequentially; keep selection and policy fixed before test.

```bash
.venv/bin/python -m pdm freeze-monitoring-bundle --study-id STUDY_ID --candidate-id event_final
.venv/bin/python -m pdm monitor-evaluate --bundle-id BUNDLE_ID --split validation
# After validation replay completes:
.venv/bin/python -m pdm monitor-evaluate --bundle-id BUNDLE_ID --split test
```

Studies live in `runs/condition_studies/<study_id>/`; immutable bundles and evaluations live in `runs/monitoring_bundles/<bundle_id>/`. Bundles bind data/model hashes, history, reference, policies, units and calibration. Replaying a frozen bundle does not refit or adapt its reference. See the [monitoring guide](docs/condition_monitoring.md), [sensor forecasting](docs/sensor_forecasting.md) and [recorded study artifact map](reports/condition_monitoring_implementation_report.md).

### Reference quality study (legacy training control)

```bash
.venv/bin/python -m pdm quality --dataset bearings
.venv/bin/python -m pdm quality --dataset filters
.venv/bin/python -m pdm train-matrix
# After interruption: completed experiments are retained.
.venv/bin/python -m pdm train-matrix --resume-batch <batch_id>
```

The sequential worker runs nine main models and five additional GRU censoring-study configurations (seeds 42/43/44). The main filter GRU at seed 42 is also the sixth paired-study member. All runs use history 20, all admitted windows, and at most 30 gradient epochs with early stopping. Full CNS retains its train-only grouped readout selection.

```bash
.venv/bin/python -m pdm compare --dataset bearings \
  --evaluation <run_id>:<eval_id> --evaluation <another_run_id>:<eval_id>
```

See [the quality and comparison protocol](docs/quality_and_comparison.md) for definitions and artifact contracts. Old snapshots and runs remain accessible; new training requires `admission_v1` and verified file hashes. No ensemble is fitted.

### Training improvement study (v2)

Prerequisites: prepare both real datasets, download the real connectome, and complete `train-matrix` above. A new study requires a **completed reference matrix**; standalone GRU runs are insufficient. Heavy jobs run sequentially through the worker. Wait for the reference matrix to finish in **Training** before starting the study.

```bash
.venv/bin/python -m pdm training-study
.venv/bin/python -m pdm stop
.venv/bin/python -m pdm training-study --resume-study <study_id>
```

The bounded worker compares 100-epoch diagnostics, adaptive stopping, window weighting and causal degradation features, checks the two best recipes on train-only equipment folds, trains all nine main architectures, and repeats the two best validation architectures per dataset with seeds 43/44. Test runs only after candidates and warning rules are frozen. Use **Training → Training improvement study** for progress and CSV/JSON exports; **Model Report** shows the selected checkpoint and actual model states.

The completed study is `20260915T184221Z`. Local artifacts under `runs/training_studies/<study_id>/` include `manifest.json`, `results.csv/json`, `screening.csv`, `grouped_validation.csv`, `epoch_diagnostics.csv`, `seed_dispersion.csv` and `report.md`. The manifest pins data versions, protocols, seeds and exact evaluation IDs. A Git clone includes the [published summary](reports/training_study_20260916.md), not these local artifacts.

### Automated checks

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check src tests scripts
```

---

## Quick path: individual GRU training

```bash
git clone https://github.com/aleksandrsafiullin/Predictive-Maintenance-Lab.git
cd Predictive-Maintenance-Lab
./scripts/setup.sh

.venv/bin/python -m pdm download --dataset all
.venv/bin/python -m pdm prepare --dataset filters
.venv/bin/python -m pdm prepare --dataset bearings

.venv/bin/python -m pdm train --dataset bearings --arch gru --protocol diagnostic --device cpu
.venv/bin/python -m pdm train --dataset filters --arch gru --protocol adaptive \
  --learning-rate 0.0003 --sampling full_pass --feature-recipe degradation_v1 --device cpu

# Inspect validation in Model Report; freeze candidates and warning rules before test.
.venv/bin/python -m pdm app
```

---

## Project layout

```text
configs/          dataset and condition-study profiles
src/pdm/          CLI, train, evaluate, Streamlit app, models, connectome, visualization
src/pdm/monitoring/ observation quality, reference, sensor forecast, state, bundles, replay
data/raw/         downloaded datasets (gitignored)
data/processed/   prepared features + splits (gitignored)
runs/             checkpoints, metrics, evaluations, traces (gitignored)
docs/             monitoring, sensor/history, training, quality and connectome protocols
reports/          published study summaries
tests/            pytest suite
scripts/          setup.sh / run.sh (and Windows .ps1)
```

---

## More documentation

- [Condition monitoring](docs/condition_monitoring.md) — quality, regime reference, deterministic zones, action eligibility, bundles and replay
- [Condition monitoring implementation report · 17 September 2026](reports/condition_monitoring_implementation_report.md) — 48 fits, actual evaluations, verification and unresolved limitations
- [Sensor forecasting](docs/sensor_forecasting.md) — real future targets, horizon masks, baselines and quantile bands
- [History 20–60 protocol](docs/history_20_60_protocol.md) — trained variable lengths, fixed-window comparisons and continuous CNS memory
- [Data units and endpoints](docs/data_units_and_endpoints.md) — physical vs internal time, event definitions and blocked MATLAB export
- [Training protocol v2](docs/training_protocol.md) — 100-epoch diagnostics, adaptive stopping, causal degradation features and the bounded study worker
- [Training improvement results · 16 September 2026](reports/training_study_20260916.md) — actual experiments, epoch attribution, seed stability and laboratory targets
- [Completed quality study · 15 September 2026](reports/quality_study_20260915.md) — 14 real runs, four comparisons, censoring experiment and acceptance evidence
- [Quality and comparison protocol](docs/quality_and_comparison.md) — admission, immutable snapshots, shared forecasts and ranking rules
- [docs/fly_connectome.md](docs/fly_connectome.md) — MaleCNS import, graph orientation, sampling  
- [docs/fly_connectome_demo.md](docs/fly_connectome_demo.md) — short connectome demo commands  
- [docs/neural_activity_explorer.md](docs/neural_activity_explorer.md) — Explorer meaning and limits  
- [docs/malecns_visualization.md](docs/malecns_visualization.md) — optional soma viz (CC-BY; not a required setup step)  
- [docs/fly_connectome_validation.md](docs/fly_connectome_validation.md) — what was verified  

---

## Important caveats

- `--smoke` is **not** a quality benchmark.  
- Healthy regime references and statistical thresholds are provisional. Missing reference, uncalibrated bands and unavailable action timing must not be interpreted as safe operation.
- Filter `Time` units are an assumed conversion (`time_to_seconds=60`); do not treat alert horizons as calibrated wall-clock minutes without verification.  
- `filters_full_history_v1` remains disabled pending verified MATLAB table export, origin matching and endpoint/unit checks; an export helper is not evidence of a completed import.
- Connectome Explorer shows **computational** reservoir activity, not a biophysical fly-brain recording.  
- Do not compare models across incompatible dataset versions / gap-rule versions / splits.

---

## Citation

Wang et al., IEEE Transactions on Reliability, 2020 (XJTU-SY).  
Hagmeyer, Mauthe, Zeiler, IJPHM 2021 (HSE filters), CC BY 4.0.  
MaleCNS / FlyEM Male CNS connectome — see [male-cns.janelia.org](https://male-cns.janelia.org/) when using `real_connectome`.
