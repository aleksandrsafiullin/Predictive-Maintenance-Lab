# Subtask 1: Swap two units in one new snapshot

## Goal
Add `preview_swap` and `swap_units` next to `move_units` so two whole units can exchange set membership in a single immutable snapshot, with the same lock, refusals, and parent-snapshot retention as a move.

## Context
`src/pdm/data/project_prepare.py` already publishes manual splits through `move_units`: stage outside `ProjectStore.launch_lock`, then under the lock re-check `heavy_job_active`, compare `active_snapshot_id`, copy `zone_limits.json` if present, rename, and `_activate_snapshot_locked`. The lock is not re-entrant. `snapshot_path` calls `project_path` which calls `get`. Two `move_units` calls would publish two snapshots and can fail the empty-set check in between. Data Quality (subtask 03) needs one publish. No Streamlit in this subtask.

## Acceptance Criteria
- [ ] `preview_swap(snapshot, unit_a, unit_b) -> dict` is pure (no I/O). It returns `counts`, `problem`, `fixed_units`, and `from` / `to` (`None` unless the pair is legal). `fixed_units` matches `_fixed_test_units`: missing `units` or missing `source_group` yields `[]` and does not raise.
- [ ] Problem strings, first match, `unit_a` checked before `unit_b`: `Choose two different units.`; `Unknown unit: {id}.`; `Units must be in different sets.`; `Official HSE test units stay in Testing Data.`; `{SPLIT_LABELS[name]} would have no units. Keep at least one unit in each set.` An `author_test` unit is refused only when its current set is `test` and the swap would send it somewhere else.
- [ ] `swap_units(project_id, unit_a, unit_b, *, expected_snapshot_id, store=None)` returns the same key set as `move_units`. It publishes one new snapshot. `realized_counts` equal the parent's. Split lists stay sorted. `protocol` stays `whole_unit_project_v1_manual`. Parent `seed`, `desired_weights`, and `manual_modes` are kept.
- [ ] `manual_moves` gains one entry: `kind="swap"`, `unit_ids` sorted, `from` / `to` as unit→set maps, `at` ISO UTC. Existing move entries are unchanged (`to` stays a set-name string, no `kind`). A later `move_units` appends a normal move entry after a swap entry.
- [ ] `features.parquet`, `units.parquet`, and `feature_schema.json` are byte-identical to the parent. `data_report.json` and `processed_fingerprint.json` follow the `move_units` fields (`parent_snapshot_id`, recomputed `by_split` / `split_counts`, same `quality` / `source_digest` / `outcome_semantics` / `source_manifest_id`). A saved `zone_limits.json` is copied inside the lock. The parent directory still `load_snapshot`s. No parent run directory is deleted.
- [ ] `swap_units` does not call `move_units`. Swapping the only unit in Training with the only unit in Validation succeeds and leaves both counts at 1 (the case two moves would reject).
- [ ] Refusals: linked legacy (`LINKED_LEGACY_MOVE_ERROR`), stale `expected_snapshot_id` (second caller loses; exactly one new snapshot dir), `heavy_job_active` (`JOB_ACTIVE_MOVE_ERROR`, including a job that becomes active during staging), unknown id, same set, `author_test` leaving Test (`_hse_project` / `Test_1`). Activation failure rolls back: no new `snapshots/<id>`, no `.snapshot-*` staging dir, `active_snapshot_id` unchanged.
- [ ] Lock: destination path is computed before `launch_lock`. The critical section does not call `store.get`, `load_snapshot`, `project_path`, or `snapshot_path`. A concurrent `store.get` during the parquet copy returns within 1 s. A subprocess `swap_units` on the contract fixture finishes in under 5 s.
- [ ] Shared private helper `_publish_manual_snapshot` owns the stage/lock/activate tail. `move_units` uses it. `tests/test_project_import.py` move tests pass without assertion edits.

## Implementation Notes
- Files: `src/pdm/data/project_prepare.py`, `tests/test_project_import.py`.
- Keep the lazy `from pdm.worker import heavy_job_active` inside the publish path so `pdm.worker` monkeypatches still apply.
- Copy files with the existing `_copy_snapshot_file`. Reuse `_stage_snapshot`, `_by_split`, `_publish_snapshot`, `assert_split_coverage`.
- `from` / `to` on a swap are maps. Do not collapse them to one destination string.
- Whole `unit_id` only. No row or window arguments.
- Do not refit scalers, touch `runs/`, or add a worker job kind.

## Dependencies
None.

## Verification
```bash
.venv/bin/python -m pytest tests/test_project_import.py -q -k "swap_units or move_units or preview_swap or zone_limits"
.venv/bin/python -m ruff check src/pdm/data/project_prepare.py tests/test_project_import.py
```

New tests in `tests/test_project_import.py`, reusing `make_contract_snapshot`, `idle_worker`, and `_hse_project`:

- `test_preview_swap_problems_and_missing_units_frame`
- `test_swap_units_publishes_one_snapshot_and_keeps_counts`
- `test_swap_units_one_unit_sets_do_not_empty` — train size 1 and validation size 1, one new dir, counts unchanged
- `test_swap_units_refuses_same_set_unknown_and_author_test`
- `test_swap_units_stale_snapshot_race`
- `test_swap_units_rollback_on_activation_failure`
- `test_swap_units_refuses_linked_legacy_and_active_job`
- `test_swap_units_does_not_deadlock_or_hold_lock_while_copying`
- `test_swap_units_keeps_saved_zone_limits`
- `test_swap_units_does_not_call_move_units` — monkeypatch `move_units` to raise; `swap_units` still returns
- `test_move_after_swap_appends_string_to_entry` — parent log has a swap; the next `move_units` entry has string `to`

Existing `test_move_units_*` stay as they are.
