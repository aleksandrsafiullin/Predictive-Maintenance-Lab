# Subtask 1: Whole-unit move API that publishes a new immutable snapshot

## Goal
Add `move_units(...)` in `src/pdm/data/project_prepare.py` that moves whole physical units between Train/Validation/Test by publishing a new snapshot, leaving the old snapshot and its runs untouched, without deadlocking on the registry lock. Add the single job predicate `pdm.worker.heavy_job_active`.

## Context
Data Quality (subtask 05) needs a persisted manual split change. Snapshots are immutable and runs are bound to `snapshot_id` (`ProjectStore.update`, `project_training_ui.py`, `project_results_ui.render_results`, `signal_training.load_signal_run`). `prepare_project` has the staging → rename → `store.update` publish pattern.

**Lock fact (C1):** `ProjectStore.launch_lock()` and `get`/`update`/`project_path`/`snapshot_path` (and therefore `load_snapshot`) all take `_file_lock(self.lock_path)` — `fcntl.flock` on POSIX, `msvcrt.locking` on Windows — on a new file handle. It is **not re-entrant**: `with store.launch_lock(): store.get(pid)` hangs in the same process. `cli.spawn_worker` only calls `store._load()` inside the lock.

## Acceptance Criteria
- [ ] `pdm.worker.heavy_job_active(project_id: str | None = None) -> bool` = `worker_alive()` OR `read_status().get("status") in ACTIVE_JOB_STATES` where `ACTIVE_JOB_STATES = {"queued", "starting", "running", "training", "preparing", "stopping"}` (union of the sets in `cli.spawn_worker` and `project_ui._render_import`). **Global**, like `spawn_worker`: the single worker blocks every project, so `project_id` must not narrow the result (kept only for call-site readability; drop it if unused). This is the only predicate used by `move_units`, `render_quality` (subtask 05) and tests. `cli.spawn_worker` may switch to it only if behavior is identical; not required.
- [ ] `preview_move(snapshot: dict, unit_ids, destination) -> dict` (pure, no I/O) returns `{"counts": {train, validation, test}, "problem": str | None, "fixed_units": [...]}`. A missing `snapshot["units"]` (or no `source_group` column) means "no official HSE test units" — no KeyError (W5). Refusals, exact English: "Choose at least one unit.", "Unknown unit: {id}.", "{id} is already in {Set}.", "Official HSE test units stay in Testing Data.", "{Set} would have no units. Keep at least one unit in each set."
- [ ] `move_units(project_id, unit_ids, destination: Literal["train","validation","test"], *, expected_snapshot_id: str, store: ProjectStore | None = None) -> dict` returns `{project_id, snapshot_id, parent_snapshot_id, dir, split, report, schema, fingerprint}` (same shape as `prepare_project`). It re-validates with `preview_move` against the real `load_snapshot(project_id, expected_snapshot_id)` (which has `units`), and raises `ValueError(problem)`.
- [ ] Extra refusals (ValueError/RuntimeError): project `storage_mode == "linked_legacy"` ("This project uses the published split of its source dataset. Create a new project to change the split."); `heavy_job_active()` ("Wait for the current job to finish before changing sets."); stale snapshot ("The data changed since this page loaded. Reload Data Quality and try again.").
- [ ] **Lock discipline (C1):**
  1. Outside any lock: `store.get`, `load_snapshot(pid, expected_snapshot_id)`, fail-fast `active_snapshot_id` and `heavy_job_active()` checks, `store.snapshot_path(...)` for parent and destination, `tempfile.mkdtemp(dir=snapshots/)`, copy parquet/schema files, write `split.json`, `data_report.json`, `processed_fingerprint.json`, verify hashes of the staged dir.
  2. One critical section `with store.launch_lock():` that only does: re-check `heavy_job_active()`; `registry = store._load()`; `record = store._entry(registry, pid)`; check `record["active_snapshot_id"] == expected_snapshot_id` and `record["storage_mode"] == "owned"`; `staging.rename(destination)`; `store._activate_snapshot_locked(registry, pid, new_sid)` (new private method: sets `active_snapshot_id`, `selected_run_id=None`, `state="ready"`, `_safe_id` checks, `store._save(registry)`; documented "caller must hold `launch_lock()`"). If activation raises → `shutil.rmtree(destination)` inside the same section, re-raise.
  3. Never call `store.get`, `store.update`, `load_snapshot`, `project_path`, or `snapshot_path` while `launch_lock()` is held.
- [ ] `_publish_snapshot` refactor (shared by `prepare_project` and moves) **never assumes a lock is held**; `prepare_project` and `project_ui._maybe_wrap_legacy` keep calling it unlocked and keep using `store.update(...)` for activation exactly as today. Staging/writing is a separate helper `_stage_snapshot(...) -> (staging, snapshot_id, payload)` so `move_units` can stage outside the lock and activate inside it.
- [ ] `move_units` completes in under 5 s in a subprocess test on the contract fixture and does not hold the registry lock while copying parquet files (test proves it: a concurrent `store.get` from another thread during the copy returns promptly).
- [ ] New snapshot dir: `features.parquet`, `units.parquet`, `feature_schema.json` byte-identical to the parent (same sha256); new `split.json` with parent `seed`, `desired_weights`, `manual_modes`, recomputed `realized_counts`, `protocol="whole_unit_project_v1_manual"`, `parent_snapshot_id`, `manual_moves` = parent's list + `{unit_ids (sorted), from: {unit: set}, to, at: ISO UTC}`; each split list sorted.
- [ ] `data_report.json` recomputed (`by_split`, `split_counts`, `snapshot_id`, `created_at`, `parent_snapshot_id`); `quality`, `source_digest`, `outcome_semantics` carried over.
- [ ] `processed_fingerprint.json`: new `snapshot_id`, `split_hash(split)`, `file_hashes`, same `source_digest`/`source_manifest_id`, plus `parent_snapshot_id`.
- [ ] On any failure before or during activation: no `snapshots/.snapshot-*` staging dir left, no new `snapshots/<id>` dir, project record unchanged (W8).
- [ ] Old snapshot dir bytes unchanged; `load_snapshot(pid, old_id)` still verifies; an old completed run's `load_signal_run` still works; `store.update(selected_run_id=old_run)` now raises.
- [ ] After a move, training on the new snapshot fits the scaler on the new Train units only (W7, protocol invariant): the persisted `run["scaler"]["fit_units"]` (real key written by `signal_training.train_signal_run`, line ~333) equals `sorted(new_split["train"])` and differs from the parent's Train set.

## Implementation Notes
- Files: `src/pdm/data/project_prepare.py` (`preview_move`, `move_units`, `_stage_snapshot`, `_by_split`, `_publish_snapshot`), `src/pdm/projects.py` (`_activate_snapshot_locked`), `src/pdm/worker.py` (`heavy_job_active`, `ACTIVE_JOB_STATES`).
- Import `heavy_job_active` lazily inside `move_units` (avoid a `pdm.worker` ↔ data import cycle).
- `_activate_snapshot_locked` alternative allowed by review: `update(..., expect_active_snapshot_id=parent)` with the compare inside `update`'s own lock and **no** outer lock held by the caller. Pick one; the private locked method is preferred because the rename and the registry write must share one critical section with the job re-check.
- Copying: `shutil.copyfile`. Optional note: `os.link` with `shutil.copyfile` fallback saves space/time on large parquet; not required (hardlinks are safe only because nobody writes snapshot files in place).
- `from` map uses the parent split; destination list = parent list + moved ids.
- Do not refit scalers or touch `runs/`. Do not add a new worker job kind (rationale: master plan D2).
- `split_hash` includes `protocol`, so the manual protocol name changes the hash even if lists coincide with an older snapshot — intended.
- Undo/restore is out of scope.

## Dependencies
None.

## Verification
New tests in `tests/test_project_import.py` (reuse `tests/project_contract.make_contract_snapshot`; the HSE fixture used by `test_hse_manual_validation_and_author_test_never_train`):
- `test_move_units_publishes_new_snapshot_and_keeps_old` — old dir hashes unchanged, new split disjoint, counts, `manual_moves`, `parent_snapshot_id`, byte-identical parquet.
- `test_move_units_refuses_empty_set_unknown_and_same_set`.
- `test_move_units_stale_snapshot_race` — two calls with the same `expected_snapshot_id`; first succeeds, second raises ValueError; exactly one new snapshot dir.
- `test_move_units_rollback_on_activation_failure` — monkeypatch `ProjectStore._activate_snapshot_locked` to raise → no new `snapshots/<id>`, no `.snapshot-*` staging dir, `active_snapshot_id` unchanged.
- `test_move_units_refuses_hse_author_test_leaving_test`.
- `test_move_units_refuses_linked_legacy`.
- `test_move_units_refuses_while_heavy_job_active` — monkeypatch `pdm.worker.heavy_job_active` (or `worker_alive`/`read_status`).
- `test_move_units_does_not_deadlock_or_hold_lock_while_copying` — run `move_units` in a `subprocess` with `timeout=5`; separately, monkeypatch the copy step to block on an event while a thread calls `store.get(pid)` and must return within 1 s.
- `test_preview_move_without_units_frame` — snapshot dict without `units` → no KeyError, no fixed units.
- `test_move_units_old_run_not_selectable_on_new_snapshot` — minimal completed run manifest bound to the old snapshot (pattern from `tests/test_projects.py::test_run_selection_requires_completed_bound_signal_manifest`); `store.update(selected_run_id=…)` raises after the move.
- `test_train_after_move_fits_scaler_on_new_train_units` (in `tests/test_signal_models.py`, reusing its `contract` fixture): move one Train unit to Validation, `train_signal_run(pid, new_sid, "gru", {"epochs": 1, "history_length": 2, "hidden_size": 4, ...short horizons})`; assert `run["scaler"]["fit_units"] == sorted(new_split["train"])` and `!= sorted(parent_split["train"])`. Protocol invariant, not a quality claim.
- `tests/test_spec_invariants.py`: a moved snapshot keeps whole-unit disjointness and every unit's rows in exactly one split.
- Unit test for `heavy_job_active` states (queued/starting/running/training/preparing/stopping → True; completed/failed/cancelled + dead worker → False).
```bash
.venv/bin/python -m pytest tests/test_project_import.py tests/test_projects.py tests/test_signal_models.py tests/test_spec_invariants.py -q
.venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check src tests
```
