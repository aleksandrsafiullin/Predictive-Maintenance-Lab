# Predictive Maintenance Lab

Local **GRU/LSTM** training prototype for two tasks:

- **Bearings / XJTU-SY** — remaining useful life (RUL) from bearing vibration
- **Filters / HSE** — time to 600 Pa on a gas filter (censored training histories)

The UI is in English. Training and data preparation run in a separate worker process.

## Environment

Tested on macOS arm64, Python **3.12**, PyTorch with MPS. Windows scripts (`setup.ps1`, `run.ps1`) have not been run on this machine.

```bash
./scripts/setup.sh
./scripts/run.sh
```

Manual equivalent:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install -e .
.venv/bin/python -m pdm doctor
.venv/bin/python -m pdm download --dataset filters
.venv/bin/python -m pdm download --dataset bearings   # or --local-path to the author's archive
.venv/bin/python -m pdm prepare --dataset filters
.venv/bin/python -m pdm prepare --dataset bearings
.venv/bin/python -m pdm train --dataset filters --arch gru --epochs 3 --smoke
.venv/bin/python -m pdm train --dataset bearings --arch gru --epochs 3 --smoke --max-windows-per-unit 32
.venv/bin/python -m pdm evaluate --dataset filters --run-id <run_id>
.venv/bin/python -m pdm app
```

UI: `http://127.0.0.1:8501` (localhost only; Streamlit telemetry disabled).

## Data

- **Bearings:** author page https://biaowang.tech/xjtu-sy-bearing-datasets/  
  Raw files live in `data/raw/bearings/`. Split is 9/3/3: per operating condition, units 1–3 train, 4 val, 5 test.
- **Filters:** Kaggle `prognosticshse/preventive-to-predicitve-maintenance` (typo `predicitve` in the slug).  
  Mode `filters_censored`. The 50 held-out test runs are untouched. `Train_Data_Uncensored.mat` is a MATLAB table that scipy cannot read; `filters_full_history` is disabled.

Do not commit `data/`, `runs/`, or `.venv/`.

## Citation

Wang et al., IEEE Transactions on Reliability, 2020 (XJTU-SY).  
Hagmeyer, Mauthe, Zeiler, IJPHM 2021 (HSE filters), CC BY 4.0.
