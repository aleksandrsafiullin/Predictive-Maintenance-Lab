# Subtask 20: Eval/replay integration gate (TZ §9 remainder)

## Goal

Add remaining TZ §9 integration tests not owned elsewhere; extend AppTest for replay auto-advance. Sprint **full-suite** gate only (excluding full real-data training).

## Context

**Test ownership (do not duplicate):**
- Subtask 08: `test_evaluate_rejects_changed_dataset_or_split`
- Subtask 11a/11b: preprocessing, survival, gap, causality tests
- Subtask 15b: `test_insufficient_coverage_not_false_miss`, `test_alert_multiple_episodes_and_time_weighting`

Subtask 20 implements and gates replay/UI integration tests (including the three named below).

## Acceptance Criteria

- [ ] `test_alert_policy_cache_invalidation` — H/K change alters alerts only, not RUL predictions; stale cache not shown.
- [ ] `test_replay_log_has_no_future_rows`.
- [ ] `test_replay_play_advances_without_clicks` — AppTest simulates Play + fragment rerun.
- [ ] `test_filter_endpoint_baseline_comparison`.
- [ ] Full suite green: `.venv/bin/python -m pytest tests -q`.
- [ ] Does **not** re-implement tests owned by 08, 11a, 11b, or 15b.

## Implementation Notes

**Files:** `tests/test_spec_invariants.py`, `tests/test_worker_and_app.py`, `tests/conftest.py`

- Mock run dir with `dataset_fingerprint.json`, `split.json`, `best.pt`, eval subdir.
- For play advance: use Streamlit AppTest `run(timeout=...)` with session state seeds.
- If 15b tests already pass, 20 only re-runs full suite.

## Dependencies

Subtasks 8, 14, 15, 17, 15b, 18, 19. **Soft:** 16.

## Verification

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check src tests
```

**Estimate:** 2–2.5 h (AppTest + integration wiring; alert episode tests owned by 15b).
