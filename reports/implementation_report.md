# Implementation report — Predictive Maintenance Lab

Date: 2026-09-10  
Host: macOS 26.6.2 arm64 (Apple M3), Python 3.12.14, PyTorch 2.14.0, device **mps**.

Python 3.11 was not installed; 3.12 was used as allowed by the spec.

## 1. What was implemented / commands actually run

Working local package `src/pdm` with Streamlit UI (3 screens), CLI, file-backed experiments, GRU/LSTM, bearing RUL head, filter Weibull NLL with censoring, baselines, replay isolation, alerts.

Executed:

```text
.venv/bin/python -m pdm doctor                          # pass, mps, forward/backward ok
.venv/bin/python -m pdm download --dataset filters       # Kaggle version 8 via kagglehub
.venv/bin/python scripts/download_xjtu_zip.py            # 5 447 931 446 byte author zip
.venv/bin/python -m pdm inspect --dataset filters
.venv/bin/python -m pdm prepare --dataset filters
.venv/bin/python -m pdm prepare --dataset bearings
.venv/bin/python -m pdm train --dataset filters --arch gru --epochs 3 --smoke --max-windows-per-unit 32
.venv/bin/python -m pdm train --dataset filters --arch lstm --epochs 2 --smoke --max-windows-per-unit 24
.venv/bin/python -m pdm train --dataset bearings --arch gru --epochs 3 --smoke --max-windows-per-unit 32
.venv/bin/python -m pdm train --dataset bearings --arch lstm --epochs 2 --smoke --max-windows-per-unit 32
.venv/bin/python -m pdm evaluate --dataset filters --run-id smoke_filters_gru_20260910_153840_cedeed
.venv/bin/python -m pdm evaluate --dataset bearings --run-id smoke_bearings_gru_20260910_154538_b29432
.venv/bin/python -m pdm evaluate --dataset bearings --run-id smoke_bearings_lstm_20260910_154559_cea159
.venv/bin/python -m pdm evaluate --dataset filters --run-id smoke_filters_lstm_20260910_153912_eb941f
.venv/bin/python -m pytest tests -q                      # 14 passed
.venv/bin/python -m pdm app / scripts/run.sh             # Streamlit 127.0.0.1:8501
```

## 2. How to launch / URL checked

From project root, after `./scripts/setup.sh` (already done here):

```bash
./scripts/run.sh
# or
.venv/bin/python -m pdm app
```

UI: **http://127.0.0.1:8501** (localhost only, Streamlit telemetry off).

Verification:

- `GET /` → HTTP 200; `GET /_stcore/health` → `ok` / 200.
- Streamlit `AppTest` walked Data / Train / Test & Replay for Bearings and Filters without exceptions. Filters Data showed train/val/test units 40 / 10 / 50, 78834 measurements, 5 observed 600 Pa events.
- IDE browser MCP could not create a tab in this session; no screenshot pass. Curl + AppTest used instead.

Windows `scripts/setup.ps1` and `run.ps1` are present and **not executed**.

## 3. Environment

| Item | Value |
|---|---|
| OS | macOS-26.6.2-arm64 |
| Python | 3.12.14 (`.venv`) |
| torch | 2.14.0 (macOS ARM wheel, MPS) |
| Device | mps (CUDA unavailable) |
| numpy / pandas / sklearn | 2.5.3 / 3.0.5 / 1.9.0 |
| streamlit | 1.63.0 |

Lockfile: `requirements-lock.txt`.

## 4. Data obtained

### Bearings / XJTU-SY

- Author page: https://biaowang.tech/xjtu-sy-bearing-datasets/
- Google Drive folder from that page was not used for the completed fetch (gdown not relied on). Downloaded `XJTU-SY_Bearing_Datasets.zip` from a public copy of the **same author package** (15 bearings, CSV with `Horizontal_vibration_signals,Vertical_vibration_signals`, 32 768 samples/file).
- Path: `data/raw/bearings/XJTU-SY_Bearing_Datasets.zip`
- Size: 5 447 931 446 bytes
- SHA-256: `3cc815649a315ac7da202980c489f33db44ca2db0317bbe3bcb9dcf415375e10`
- Fragments: **9216** CSV, **15** units, **0** file-index gaps
- Split (spec 9/3/3): train `Bearing{1,2,3}_{1,2,3}`; val `*_4`; test `*_5`
- Time: `timestamp_s = file_index * 60`; `endpoint_definition=last_recorded_sample`
- 20-step history = **20 minutes** of operating time

### Filters / HSE

- Kaggle `prognosticshse/preventive-to-predicitve-maintenance` version **8**, CC BY 4.0
- CSV columns (inspected): train `Data_No, Differential_pressure, Flow_rate, Time, Dust_feed, Dust`; test adds `RUL` on every row (evaluation uses **last row per unit only** to reconstruct event time)
- 50 author-train + 50 author-test units. IDs namespaced `Train_*` / `Test_*` because `Data_No` 1–50 is reused across files (not the same physical units).
- Observed 600 Pa events in author train: **5** (units 11, 43, 44, 46, 47); 45 right-censored
- Split: test 50 held out; author train 80/20 units, seed 42, stratified on events → 40 train (4 events) / 10 val (1 event)
- `Time` unit is **not named** in the PDF. Recorded decision: original Time treated as **minutes** (`× 60` → seconds), because Sampling is documented in Hz, maintenance-interval length scales as 1/`Dust_feed` around 33–179, and hours would imply implausible dust volume. CRAN `degradr` labels RUL as hours; that is noted as a conflict.
- `*.mat` files are MATLAB `table` (MCOS opaque). scipy cannot read them. **`filters_full_history` is disabled.**

## 5. Models trained (real artifacts)

All marked `Smoke test — not a quality benchmark`. Limited windows/unit. Not a quality claim.

| run_id | dataset | arch | epochs | best val metric | checkpoint |
|---|---|---|---|---|---|
| `smoke_filters_gru_20260910_153840_cedeed` | filters | GRU | 3 | NLL 0.0694 (unit-equal) | `runs/filters/.../best.pt` |
| `smoke_filters_lstm_20260910_153912_eb941f` | filters | LSTM | 2 | NLL 0.2232 | `runs/filters/.../best.pt` |
| `smoke_bearings_gru_20260910_154538_b29432` | bearings | GRU | 3 | MAE 19291.7 s (unit-equal val) | `runs/bearings/.../best.pt` |
| `smoke_bearings_lstm_20260910_154559_cea159` | bearings | LSTM | 2 | MAE 23907.0 s | `runs/bearings/.../best.pt` |

Test-set (not used for epoch pick or scalers):

- Bearings GRU: MAE 6639 s, equal-unit MAE 7976 s, n=448 points / 3 units. Age-only baseline equal-unit MAE 52800 s (NN better overall; baseline better on Bearing1_5). H=2220 s, K=3, no timely warnings (smoke overestimate).
- Bearings LSTM: MAE 16898 s / equal-unit 19288 s.
- Filters GRU primary (RUL at official prefix end, 50 units): MAE 7012 s. Linear Δp baseline coverage 8.6% of points (unreliable slope → no number). On overlap, baseline MAE 939 s vs NN 7269 s (**baseline better where it fires**). H=420 s. 43 units `Insufficient observed coverage` relative to H; 7 scored, 0 timely.
- Filters LSTM primary prefix-end MAE 6562 s (also smoke; not a quality benchmark).

## 6. Tests

`pytest tests -q`: **14 passed**.

Covers: numeric CSV sort + missing index time scale; disjoint splits; test mutation does not change train scaler; future rows do not change pred at t; official RUL does not change pred; windows vs gaps/units; RUL/time conversion; censored ≠ event; Weibull NLL vs hand values + finite grads; GRU/LSTM both heads update weights; checkpoint reload; H/K confirmation/dedup/reset; truncated test not an event; AppTest startup.

## 7. Limitations

- Smoke runs only (2–3 epochs, ≤32 windows/unit). Full 30-epoch training not executed.
- Filter Weibull is weak with 5 events; results are stored as-is.
- Filter Time unit inferred (minutes); PDF does not state it.
- `Train_Data_Uncensored.mat` unused (unreadable MATLAB table).
- Author Google Drive not used for the successful zip fetch; HF mirror of the same zip was.
- Windows launchers unverified.
- Browser tool in this agent session could not open a tab; UI checked with curl + AppTest.
- No live equipment, no notifications, no hyperparameter search (out of MVP).

## File map

```
configs/bearings.yaml, filters.yaml
scripts/setup.sh, setup.ps1, run.sh, run.ps1, download_xjtu_zip.py
src/pdm/cli.py app.py worker.py train.py evaluate.py replay.py alerts.py
src/pdm/models.py losses.py preprocessing.py windows.py splits.py features.py baselines.py
src/pdm/data/download.py bearings.py filters.py prepare.py archive.py
tests/test_spec_invariants.py test_worker_and_app.py
data/raw/{bearings,filters}  data/processed/{bearings,filters}  runs/{bearings,filters}
```
