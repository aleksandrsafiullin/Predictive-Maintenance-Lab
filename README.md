# Predictive Maintenance Lab

Локальный прототип обучения **GRU/LSTM** для двух задач:

- **Bearings / XJTU-SY** — RUL по вибрации подшипника
- **Filters / HSE** — время до 600 Па на газовом фильтре (цензурированные обучающие истории)

Интерфейс на английском. Обучение и подготовка данных идут в отдельном worker-процессе.

## Среда

Проверено на macOS arm64, Python **3.12**, PyTorch с MPS. Скрипты Windows (`setup.ps1`, `run.ps1`) не выполнялись на этой машине.

```bash
./scripts/setup.sh
./scripts/run.sh
```

Эквивалент вручную:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install -e .
.venv/bin/python -m pdm doctor
.venv/bin/python -m pdm download --dataset filters
.venv/bin/python -m pdm download --dataset bearings   # или --local-path к архиву автора
.venv/bin/python -m pdm prepare --dataset filters
.venv/bin/python -m pdm prepare --dataset bearings
.venv/bin/python -m pdm train --dataset filters --arch gru --epochs 3 --smoke
.venv/bin/python -m pdm train --dataset bearings --arch gru --epochs 3 --smoke --max-windows-per-unit 32
.venv/bin/python -m pdm evaluate --dataset filters --run-id <run_id>
.venv/bin/python -m pdm app
```

UI: `http://127.0.0.1:8501` (только localhost, телеметрия Streamlit выключена).

## Данные

- Подшипники: страница автора https://biaowang.tech/xjtu-sy-bearing-datasets/  
  Оригиналы в `data/raw/bearings/`. Разбиение 9/3/3: в каждом режиме экземпляры 1–3 train, 4 val, 5 test.
- Фильтры: Kaggle `prognosticshse/preventive-to-predicitve-maintenance` (опечатка `predicitve` в slug).  
  Режим `filters_censored`. Тестовые 50 испытаний не трогаются. `Train_Data_Uncensored.mat` — MATLAB table, scipy не читает; `filters_full_history` выключен.

Не коммить `data/`, `runs/`, `.venv/`.

## Цитирование

Wang et al., IEEE Transactions on Reliability, 2020 (XJTU-SY).  
Hagmeyer, Mauthe, Zeiler, IJPHM 2021 (HSE filters), CC BY 4.0.
