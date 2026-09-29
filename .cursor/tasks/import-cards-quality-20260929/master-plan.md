# Import cards, manual unit moves, traffic-light overlay — master plan

## Overview

Three user requests on the project workflow (`src/pdm/project_ui.py` → `project_quality_ui.py`), not the legacy `app.py` `_render_import`:

1. **Import data** looks like Apple Create ML: three equal bordered cards — Training Data, Validation Data, Testing Data. Each card takes its own source. Validation/Testing can be "Automatically split from training data". Each card shows **Units · Admitted rows · Gaps** from the active snapshot (empty state before import).
2. **Data Quality** lets the user move whole physical units between Training / Validation / Testing. Persisted as a new immutable snapshot.
3. **Data Quality** chart paints green / yellow / red zone labels on the sensor series using the snapshot's own threshold rule, with honest captions and zone counts.

## Current behavior vs target

| Area | Current | Target |
|---|---|---|
| Import layout | `_source_widgets`: "Training source" box; second "Validation and Testing" box with radios `Automatic holdout` / `Separate folder`; collapsed "Automatic split weights · 70 / 15 / 15" expander | Three cards in `st.columns(3)`, each `st.container(border=True)`; source row at the bottom of each card; Validation/Testing selector "Split from training" / "Separate folder"; auto cards say "Automatically split from training data" + effective % |
| Import counts | none | Units / Admitted rows / Gaps from `part_summary(active snapshot)`; "No data yet" before any snapshot; "View" → Data Quality on that tab |
| Split after import | Fixed until re-import | Whole-unit moves on Data Quality publish a new snapshot |
| Quality chart | Single `series_observed` line via `gap_safe_trace`, no zones | Same gap-safe line + per-row zone markers (green/yellow/red/unknown) + yellow/red limit lines/bands shared with Results; counts per unit and per split |

## Investigation findings (load-bearing)

- `prepare_project` (`data/project_prepare.py`) writes `features.parquet`, `units.parquet`, `split.json`, `feature_schema.json`, `data_report.json`, `processed_fingerprint.json` into a staging dir, renames to `snapshots/<uuid>`, then `store.update(active_snapshot_id=…, selected_run_id=None, state="ready")`. `load_snapshot` verifies every hash and `assert_split_coverage`.
- `ProjectStore.update` refuses a `selected_run_id` whose run manifest `snapshot_id` ≠ `active_snapshot_id`. `project_training_ui.py` (line ~142) and `project_results_ui.render_results` (line ~222) already filter runs by `snapshot_id`. `load_signal_run` loads the run's own snapshot by id. ⇒ old runs stay valid against their old snapshot as long as the old snapshot dir is never deleted.
- Unit provenance: `units.parquet.source_group` ∈ {`primary`, `validation`, `test`, `author_test`}. HSE official `Test_Data_CSV.csv` units are `author_test` and forced into Test by `allocate_project_split`.
- Linked legacy snapshots (`storage_mode == "linked_legacy"`) inherit the published legacy split (XJTU 9/3/3, HSE author protocol).
- **Threshold rule actually used by the project runtime** is `signal_inference._resolved_thresholds(schema, prefix)`:
  - `absolute`: fixed `yellow`/`red`, `direction` above|below (all owned imports; legacy HSE 300/600 Pa).
  - `initial_baseline_multiple` (linked legacy bearings only): baseline = median of the unit's first `baseline_n` (5) rows; yellow = max(median + 3·pop-sd, 1.25·median); red = 2·median; unavailable until `baseline_n` rows exist.
  - Red comparison is inclusive: `value >= red` (above) / `value <= red` (below) — `forecast_prefix` crossing logic.
- `health_zones.signal_zone_reference` is a **different** rule (legacy bearing zone lab: 3-sample persistence for yellow, red only after baseline, reads `horizontal_rms`/`vertical_rms`). Project snapshots carry no `onset_persist` and no per-axis RMS. It must not be reused for the project overlay, otherwise the overlay would disagree with Results/crossing.
- **What launched project models train on:** `signal_training._windows` builds `x = last history_length signal values` (single segment, no gap crossing) and `y = signal value at t + horizon` (masked when not exactly present). GRU / LSTM / quantile boosting are trained on **future numeric signal values**, never on zone class IDs. Thresholds are applied only after prediction (Results red-crossing). Class-label models (`health_zones.ZoneGRU`, `future_red_*`) exist only in the legacy research lab, not in this project flow.
- Streamlit 1.63: `st.tabs(..., key=, on_change=)` exists → "View" can open the matching Data Quality tab through the `quality_tab` session key (no `default=`).
- `ProjectStore._file_lock` is non-re-entrant and shared by `launch_lock`, `get`, `update`, `project_path`, `snapshot_path` (hence `load_snapshot`). `cli.spawn_worker` only uses `store._load()` inside the lock.
- `train_signal_run` persists `scaler = {mean, std, fit_units: sorted(train ids), output_domain}` in the run manifest and `training_contract.json` → post-move train-only fitting is testable on the real key `scaler["fit_units"]`.

## Locked product decisions

### D1 — Label source and overlay honesty
- One shared module `src/pdm/project_zones.py` owns the project threshold rule. `signal_inference._resolved_thresholds` moves there as `resolve_thresholds` (keep `_resolved_thresholds` as an import alias in `signal_inference.py`), and `forecast_prefix` uses the shared comparator for "already red"/crossing. Quality overlay, card/tab counts and Results all use this module. No second rule.
- Zone per admitted row: `red` if beyond red (inclusive), else `yellow` if beyond yellow (inclusive), else `green`; `unknown` when thresholds are unavailable (baseline mode before `baseline_n` rows, or invalid rule). Gaps do not change labels; the line still breaks via `gap_safe_trace`.
- Caption (`project_zones.describe_rule(schema)`) is **built from `schema["thresholds"]`**, never hardcoded:
  - Absolute: "Zones use the limits saved with this data: yellow at {≥|≤} {yellow} {unit}, red at {≥|≤} {red} {unit} (instantaneous, per measurement)."
  - Baseline multiple: "Zones use each unit's initial baseline: median of its first {baseline_n} measurements; yellow = max(median + {onset_sigma} × SD, {onset_ratio} × median), red = {red_ratio} × median. The first {baseline_n − 1} measurements are not zoned." Unzoned count is exactly `baseline_n − 1`.
  - `thresholds["note"]` appended when present (legacy HSE "Provisional laboratory bands; not confirmed industrial fault limits").
  - Always: "Signal models in this project learn to forecast future {signal label} values, not zone classes. Zones are derived from those values with the same limits, and Results uses them to report the expected red entry."
- Zone/crossing parity is tested only on prefixes with ≥ `history_length` rows since the last gap, using the real `forecast_prefix` via the `contract` fixture in `tests/test_signal_models.py`.

### D2 — Snapshot-move semantics
- Move unit = whole physical `unit_id`. Never rows or windows.
- API: `move_units(project_id, unit_ids, destination, *, expected_snapshot_id, store=None) -> dict` and pure `preview_move(snapshot, unit_ids, destination)` in `data/project_prepare.py`. Runs synchronously in the Streamlit process (no model compute; parquet copy + JSON). ⚠ Speculative: a multi-GB `features.parquet` copy could take seconds; optional note: `os.link` with `shutil.copyfile` fallback (not required).
- **Job predicate (single source):** new `pdm.worker.heavy_job_active(project_id=None) -> bool` = `worker_alive()` OR `read_status()["status"]` ∈ {queued, starting, running, training, preparing, stopping}. Global like `spawn_worker` (one worker blocks all projects). Used by `move_units`, `render_quality`, and tests.
- **Lock discipline (registry lock is not re-entrant):** `ProjectStore.launch_lock()` is the same `_file_lock(.registry.lock)` as `get`/`update`/`project_path`/`snapshot_path`/`load_snapshot`; a nested acquire in one process hangs (`fcntl.flock` / `msvcrt.locking` on a new handle). Therefore:
  1. Outside any lock: `store.get`, `load_snapshot(pid, expected_snapshot_id)`, `preview_move` re-validation against real `units`, fail-fast checks, path resolution, build the entire staging dir (copies + `split.json` + `data_report.json` + `processed_fingerprint.json`).
  2. One critical section `with store.launch_lock():` doing only: re-check `heavy_job_active()`; `registry = store._load()`; compare `registry` entry `active_snapshot_id == expected_snapshot_id` and `storage_mode == "owned"`; `staging.rename(dest)`; `store._activate_snapshot_locked(registry, pid, new_sid)` (sets `active_snapshot_id`, `selected_run_id=None`, `state="ready"`, saves; caller must hold the lock). On failure `rmtree(dest)` inside the section.
  3. No `store.get`/`update`/`load_snapshot`/`project_path`/`snapshot_path` while the lock is held.
  4. `_publish_snapshot`/`_stage_snapshot` never assume a lock; `prepare_project` and `_maybe_wrap_legacy` keep calling them unlocked with `store.update` activation as today.
  - Acceptance: `move_units` finishes < 5 s in a subprocess test and a concurrent `store.get` returns while parquet is being copied.
- Publishes a new snapshot dir: `features.parquet`, `units.parquet`, `feature_schema.json` byte-identical; new `split.json` (same seed/desired_weights, recomputed `realized_counts`, `protocol="whole_unit_project_v1_manual"`, `parent_snapshot_id`, append-only `manual_moves` `{unit_ids, from, to, at}`); recomputed `data_report.json`; new `processed_fingerprint.json` (new `split_hash`, same `source_digest`/`source_manifest_id`, `parent_snapshot_id`). Failure at any step leaves no staging dir, no new snapshot dir, and an unchanged project record (rollback test + stale-snapshot race test).
- Old snapshot dir is never modified or deleted. Old runs remain loadable (`load_signal_run` → old snapshot) but are **not listed** in Training/Results for the new snapshot. Training page shows: "{n} earlier model run(s) were trained on a previous data snapshot and are not shown. Train again on this data." Results nav stays disabled until a run on the new snapshot exists (already true via `selected_run_id=None`).
- Refusals (ValueError/RuntimeError, shown as `st.warning`, never a crash): stale `expected_snapshot_id`; unknown unit; unit already in destination; any `author_test` unit leaving Test ("Official HSE test units stay in Testing Data."); `linked_legacy` project ("This project uses the published split of its source dataset. Create a new project to change the split."); a set left with zero units ("{Set} would have no units. Keep at least one unit in each set."); `heavy_job_active()`. The UI previews counts via `preview_move` and disables the button with the same warning. `preview_move` treats a missing `units` frame as "no official test units"; `move_units` re-validates on the real snapshot.
- Official HSE test units are excluded from the Testing tab's movable list, with a caption giving their count.
- Moving units into or out of Testing shows: "Changing Testing units after reviewing results makes later Test scores optimistic."
- Scalers are not touched; the next `train_signal_run` fits on the new Train units — asserted via the real persisted `run["scaler"]["fit_units"]` (`test_train_after_move_fits_scaler_on_new_train_units`). Re-import replaces manual moves with a fresh split (import caption says so). Undo/restore is out of scope.

### D3 — Import cards
- Cards are layout over the unchanged `ImportSpec` (`validation_mode`/`test_mode` ∈ auto|folder, `weights`, `seed`). No backend change for import.
- Auto weights/seed live in a "Split settings" `st.popover` (fixed label and `key="import_split_settings"`, so it doesn't close on value change) directly under the cards. Inputs keyed `import_weight_train`, `import_weight_validation`, `import_weight_test`, `import_seed`; cards read `st.session_state.get(key, default)` before the popover renders. Each auto card shows its effective renormalized share, e.g. Testing manual ⇒ Validation "18% of the training pool".
- Card counts come from a `st.cache_data` helper keyed on `(project_id, snapshot_id)` that runs `load_snapshot` + `part_summary` once per immutable snapshot (no full hash check on every rerun); displayed counts equal `part_summary`. Counts are captioned "Saved data" so they aren't mistaken for the folder just picked.
- "View" `on_click` sets `quality_tab` to the tab label and `project_step="Data Quality"`; Data Quality uses `st.tabs(labels, key="quality_tab", on_change="rerun")` with no `default=`; zone labelling runs only for the open tab (`tab.open`). `_reset_project_session` pops `quality_tab`, `import_weight_*`, `import_seed`, `import_split_settings`, `import_view:*`, `quality_move*`.
- `_source_widgets` is replaced by `_import_cards`; `test_failed_worker_launch_cleans_this_attempt_uploads` patches `_import_cards` instead.
- HSE: Testing card in auto mode shows "Official HSE test units (Test_Data_CSV.csv) in the Training folder stay in Testing." Validation card with a separate folder still rejects `Test_Data_CSV.csv` (backend already does).
- Signal & limits box stays on the page below the cards, unchanged.

## Phases

| Phase | Subtasks | Kind |
|---|---|---|
| A. Data/protocol | 01 snapshot move API | backend + invariants |
| B. Shared rule | 02 project zone labels | backend + invariants |
| C. Import chrome | 03 three-card import page | UI + AppTest |
| D. Quality chrome | 04 zone overlay + summary, 05 move control | UI + AppTest |

## Dependencies

- 01, 02: independent (can run in parallel; different files).
- 03: independent of 01/02 at code level (uses `part_summary`); run after 01 so card copy about manual moves is accurate.
- 04: depends on 02 and 03 (03 adds `key="quality_tab", on_change="rerun"` to `st.tabs`, which 04 needs for `tab.open`).
- 05: depends on 01 and 04 (same file `project_quality_ui.py`; avoid parallel edits).

## Execution order

01 → 02 → 03 → 04 → 05. (01 ∥ 02 allowed.)

## Effort

| Subtask | Estimate |
|---|---|
| 01 move API + lock + job predicate + tests | 2 h |
| 02 zone labels + tests | 1–1.5 h |
| 03 import cards + AppTest | 1.5–2 h |
| 04 zone overlay + summary + AppTest | 1.5 h |
| 05 move control + AppTest | 1–1.5 h |
| Total | ~7–8.5 h |

## Leakage / protocol invariants (must hold after every subtask)

- Whole-unit splits: disjoint Train/Validation/Test; `assert_split_coverage` passes on every published snapshot; origin leaks rejected.
- HSE `author_test` units always in Test; official RUL never an input.
- Content-fingerprint duplicate rejection unchanged (`_canonicalize_adapted`).
- Snapshots immutable: no in-place writes to an existing `snapshots/<id>` dir.
- Runs bound to their snapshot; `ProjectStore.update` still rejects cross-snapshot `selected_run_id`.
- Registry lock never nested: nothing that calls `get`/`update`/`load_snapshot`/`project_path`/`snapshot_path` runs inside `launch_lock()`.
- One job predicate: `pdm.worker.heavy_job_active`.
- Train-only scalers fit at train time only; the move writes no scaler.
- Zone labels are display/diagnostic only; not added to `features.parquet`, not a model input.
- No FastAPI/React, no new NN architecture, `filters_full_history` stays disabled.

## UI copy notes (English)

- Card titles: "Training Data", "Validation Data", "Testing Data" (match `PARTS` in `project_quality_ui.py`).
- Auto state text: "Automatically split from training data".
- Empty state: "No data yet" / "Import to see units, rows, and gaps."
- Metric labels exactly "Units", "Admitted rows", "Gaps" with existing `QUALITY_*_HELP`.
- Move control: "Move units", "Units to move", "Move to", "Move selected units", success "Moved {n} unit(s) to {Set}. A new data snapshot is active; earlier model runs stay with the previous data snapshot." Test caption: "Changing Testing units after reviewing results makes later Test scores optimistic." Fixed units: "{n} official HSE test unit(s) are fixed in Testing Data." Move-widget keys are cleared inside the button `on_click`, not after render.
- Zone legend: "Green · {n}", "Yellow · {n}", "Red · {n}", "Not zoned · {n}".
- All new widgets need `help=` (AppTest `test_import_and_quality_controls_expose_help` asserts every radio/number/text/select on Import has help). New strings go in `src/pdm/ui_copy.py`.

## Verification (every subtask)

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check src tests
```

Focused: `tests/test_project_import.py`, `tests/test_projects.py`, `tests/test_spec_invariants.py`, `tests/test_project_ui.py`, `tests/test_worker_and_app.py -k "import or quality or project"`, `tests/test_signal_models.py`. UI subtasks: AppTest plus manual check at `http://127.0.0.1:8501` in light and dark. `--smoke` training is not a quality claim.
