# Predictive Maintenance Lab

Local lab for **remaining useful life (RUL)** and filter degradation forecasting on two public datasets:

| Dataset | Task | Split (units) |
|---|---|---|
| **Bearings / XJTU-SY** | RUL from vibration features | 9 train / 3 validation / 3 test |
| **Filters / HSE** | Time to 600 Pa (censored survival) | 40 train / 10 validation / 50 author test held out |

**Models:** `gru`, `lstm`, plus experimental `fly_connectome_reservoir` and `random_reservoir` (leaky Echo State Networks).

English UI (Streamlit). Heavy jobs run in a background worker. Repo: https://github.com/aleksandrsafiullin/Predictive-Maintenance-Lab

---

## What you need

- macOS or Linux (scripts tested on **macOS arm64**, Python **3.11 or 3.12**, PyTorch with CPU or MPS)
- ~**8+ GB** free disk for filters + prepared data; bearings zip alone is ~**5 GB**
- Optional: Kaggle credentials for filters download (`kagglehub`)
- Optional: ~**1 GB** MaleCNS feather for real fly connectome (not required for GRU/LSTM)

Do **not** commit `data/`, `runs/`, or `.venv/`.

---

## 1. Clone and install

```bash
git clone https://github.com/aleksandrsafiullin/Predictive-Maintenance-Lab.git
cd Predictive-Maintenance-Lab

./scripts/setup.sh
# Windows (unverified on this machine): .\scripts\setup.ps1
```

`setup.sh` creates `.venv`, installs `requirements-lock.txt`, editable package, and runs `pdm doctor`.

Manual equivalent:

```bash
python3.12 -m venv .venv   # or python3.11
.venv/bin/python -m pip install --upgrade pip wheel
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install -e .
.venv/bin/python -m pdm doctor
```

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

Only for `fly_connectome_reservoir` / `random_reservoir` with `real_connectome`. GRU/LSTM do **not** need this.

```bash
mkdir -p data/raw/connectome
curl -L --fail -o data/raw/connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather \
  "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather"
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

### Full GRU (recommended baseline)

```bash
.venv/bin/python -m pdm train --dataset filters --arch gru --epochs 30 --max-windows-per-unit 0
.venv/bin/python -m pdm train --dataset bearings --arch gru --epochs 30 --max-windows-per-unit 0
```

`--max-windows-per-unit 0` = use all windows. Early stopping uses validation. Omit `--smoke`.

LSTM: same flags with `--arch lstm`.

### Fly connectome reservoir (experimental)

Synthetic graph (demo / CI):

```bash
.venv/bin/python -m pdm train --dataset bearings --arch fly_connectome_reservoir \
  --graph-mode synthetic_fixture --n-nodes 8 --smoke
```

Real MaleCNS subgraph (after step 2 optional download; `n_nodes` in 500–2000):

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
  --horizon-s 2220 --k 3 --min-action-lead-s 1110
```

Freeze alert policy from the UI (**Test & Replay → Validation → Freeze alert policy**), then frozen test:

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
# or
.venv/bin/python -m pdm app
```

Open **http://127.0.0.1:8501** (localhost only).

Pages:

1. **Data** — units, splits, prepare status  
2. **Train** — architecture, Smoke/Full, start/resume jobs  
3. **Test & Replay** — evaluate, freeze H/K on validation, replay  
4. **Neural Activity Explorer** — connectome reservoir state visualization (reservoir runs only)

### Automated checks

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check src tests
```

---

## Quick path (copy-paste)

```bash
git clone https://github.com/aleksandrsafiullin/Predictive-Maintenance-Lab.git
cd Predictive-Maintenance-Lab
./scripts/setup.sh

.venv/bin/python -m pdm download --dataset all
.venv/bin/python -m pdm prepare --dataset filters
.venv/bin/python -m pdm prepare --dataset bearings

.venv/bin/python -m pdm train --dataset bearings --arch gru --epochs 30 --max-windows-per-unit 0
.venv/bin/python -m pdm train --dataset filters --arch gru --epochs 30 --max-windows-per-unit 0

.venv/bin/python -m pdm evaluate --dataset bearings --run-id <run_id> --split test
.venv/bin/python -m pdm app
```

---

## Project layout

```text
configs/          bearings.yaml, filters.yaml
src/pdm/          CLI, train, evaluate, Streamlit app, models, connectome, visualization
data/raw/         downloaded datasets (gitignored)
data/processed/   prepared features + splits (gitignored)
runs/             checkpoints, metrics, evaluations, traces (gitignored)
docs/             fly connectome + Neural Activity Explorer guides
tests/            pytest suite
scripts/          setup.sh / run.sh (and Windows .ps1)
```

---

## More documentation

- [docs/fly_connectome.md](docs/fly_connectome.md) — MaleCNS import, graph orientation, sampling  
- [docs/fly_connectome_demo.md](docs/fly_connectome_demo.md) — short connectome demo commands  
- [docs/neural_activity_explorer.md](docs/neural_activity_explorer.md) — Explorer meaning and limits  
- [docs/fly_connectome_validation.md](docs/fly_connectome_validation.md) — what was verified  

---

## Important caveats

- `--smoke` is **not** a quality benchmark.  
- Filter `Time` units are an assumed conversion (`time_to_seconds=60`); do not treat alert horizons as calibrated wall-clock minutes without verification.  
- `filters_full_history` is disabled (uncensored MATLAB table not readable by scipy).  
- Connectome Explorer shows **computational** reservoir activity, not a biophysical fly-brain recording.  
- Do not compare models across incompatible dataset versions / gap-rule versions / splits.

---

## Citation

Wang et al., IEEE Transactions on Reliability, 2020 (XJTU-SY).  
Hagmeyer, Mauthe, Zeiler, IJPHM 2021 (HSE filters), CC BY 4.0.  
MaleCNS / FlyEM Male CNS connectome — see [male-cns.janelia.org](https://male-cns.janelia.org/) when using `real_connectome`.
