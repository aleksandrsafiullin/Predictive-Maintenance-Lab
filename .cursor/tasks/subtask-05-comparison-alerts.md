# Subtask 5: Comparison table, alerts integration, demo scenario hooks

## Goal

Add a same-split comparison surface (GRU/LSTM vs fly vs random vs baselines), wire Alert inspection to real traces of the triggering window, and add demo scenario hooks that never present `synthetic_fixture` as a biological connectome.

## Context

Phase E. Phases B–D shipped reservoir metrics, traces, and the explorer page. Existing comparison logic lives in `src/pdm/evaluate.py` (`compare_baseline`, `summarize_rul_table`) and the Test & Replay metrics panel. Alerts are `src/pdm/alerts.py` + eval `alerts.csv` / episodes. Replay isolation must stay: explorer and comparison **read** frozen predictions/alerts; they do not recompute RUL from ground truth.

## Acceptance Criteria

- [ ] Comparison table can place, on the **same** dataset split and evaluation mask: `gru`, `lstm`, `fly_connectome_reservoir`, `random_reservoir`, plus existing baselines. Metrics reuse current unit-equal MAE / NLL / alert outcomes — do not invent a new unofficial test split.
- [ ] Rows with `graph_mode=synthetic_fixture` or `is_synthetic=True` are **excluded** from any table titled or captioned as a biological / MaleCNS connectome comparison. Those runs appear only in a section labeled `Synthetic test graph — not a biological connectome`.
- [ ] Fly vs random comparison is valid only when `n_nodes`, leak, spectral radius, `state_mode`, history length, dataset, and split hash match (matched N/E). Mismatched controls show a warning and are not silently averaged together.
- [ ] Never load a bearings run into a filters comparison (dataset isolation).
- [ ] Alert inspection mode (explorer): picking an alert episode jumps to that `unit_id` + timestamp’s prediction window and loads **that** window’s trace (building it via worker if missing). Predicted RUL shown is the stored prediction, not a rescore that uses future rows.
- [ ] Demo hooks: a documented function or UI button `load_demo_scenario()` that selects a synthetic-fixture reservoir run + one unit + one window. Demo copy states the graph is synthetic. No canned fake states.
- [ ] Test & Replay still Freeze/research rules from the previous sprint (no new `ensure_alert_policy` writes from explorer).
- [ ] Stop during demo/trace still `cancelled` not `completed`.
- [ ] Tests in `tests/test_reservoir.py` and/or `tests/test_neural_explorer.py`: synthetic exclusion from “real” comparison; matched-N/E check; alert jump uses prefix `≤ t`. AppTest: comparison widget / explorer alert mode does not exception.

## Key Files to Create/Modify

**Create**

- `src/pdm/visualization/comparison.py` — collect run metrics, filter synthetic, matched-control checks
- `src/pdm/visualization/demo.py` — demo scenario loader (synthetic only unless a real local connectome provenance exists)

**Modify**

- `src/pdm/app.py` — comparison table on Test & Replay **or** Explorer (one canonical table, not two conflicting ones); Alert inspection controls; demo button
- `src/pdm/alerts.py` — **read-only** helpers if needed (episode → timestamp/unit). Do not change H/K math, coverage v1, or freeze provenance
- `src/pdm/evaluate.py` — optional read of existing `test_metrics.json` / eval dirs; do not change `predictions.csv` schema
- `src/pdm/replay.py` — only if sharing session mapping; keep `ReplaySource.prefix` causal
- `src/pdm/experiments.py` — `list_runs` may expose `architecture` / `graph_mode` from `status.json` or snapshot if missing
- `tests/test_neural_explorer.py`, `tests/test_reservoir.py` — comparison + synthetic exclusion + alert mapping
- `tests/test_worker_and_app.py` — only if new widgets break AppTest selectors

## Implementation Notes

### Comparison inputs

Read finished eval artifacts (`evaluations/<eval_id>/` metrics + `evaluation_config.json`). Require matching `split_hash` / `dataset_id`. Include baseline columns already produced by `compare_baseline`. Label architectures with `graph_mode`.

Fly vs random: both must be reservoir runs; random must be the rewire of the **same** `graph_hash` parent when that parent is `real_connectome`. If the parent is synthetic, the pair may be shown only in the synthetic section.

### Alerts

Use stored `predicted_rul_s` and episode `timestamp_s`. Map to frame via `frame_map.json`. If trace missing, worker `kind=trace` for that unit/window only (lazy).

Do not call `ensure_alert_policy`. Explorer is not Freeze.

### Demo

Prefer fixture graph + tiny in-memory model from unit tests for AppTest. Production demo button looks for a local run with `architecture=fly_connectome_reservoir` and `graph_mode=synthetic_fixture`. If none, tell the user to train smoke with that arch — do not generate a silent GRU demo.

### Leakage

Comparison is post-hoc metrics. Table code must not refit scalers or peek at test labels to choose which runs to show (selection is by run_id the user picks, or all compatible runs).

## Dependencies

Subtask 04 (explorer modes + component). Subtask 02 (metrics). Subtask 03 (traces for alert jump).

## Verification

```bash
.venv/bin/python -m pytest tests/test_reservoir.py tests/test_neural_explorer.py -q
.venv/bin/python -m pytest tests/test_worker_and_app.py -q
.venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check src tests
```

Browser (if available): open explorer Alert inspection on a synthetic run with a fake episode fixture; confirm jump + synthetic banner. Comparison table lists GRU run without requiring MaleCNS.

## Notes on constraints

- Synthetic fixture never appears as biological connectome (hard constraint 4).
- Dataset checkpoints never mixed.
- Alert jump is causal (no future frames).
- Do not change split protocol or alert freeze rules.
- Demo must not fake reservoir flashes.
- `--smoke` numbers in any table must keep the smoke badge / “not a quality benchmark” copy if the run was smoke.
