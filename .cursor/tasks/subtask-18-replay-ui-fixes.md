# Subtask 18: Replay UI fixes (P1-2)

## Goal

Fix five replay defects: H/K stale alerts, log filtering, auto-refresh fragment, precomputed predictions (no unbounded inline inference), composite state key.

## Context

P1-2: `_auto_refresh` fragment only updates worker caption; replay step/charts outside. H/K change leaves old `alerts.csv`. Heavy inference must not run on Streamlit request thread.

Spec/AppTest gates for replay behavior (`test_alert_policy_cache_invalidation`, `test_replay_log_has_no_future_rows`, `test_replay_play_advances_without_clicks`) are owned by **subtask 20** — same deferral pattern as 04→11b.

## Acceptance Criteria

- [ ] H/K change: re-run alert engine on cached `evaluations/<eval_id>/predictions.csv` OR show stale banner and block inconsistent status.
- [ ] Alert log: filter `unit_id == selected` AND `timestamp_s <= replay_time`.
- [ ] Play auto-advance: replay step + charts inside `@st.fragment` rerun or equivalent; Pause stops timer.
- [ ] **Prefer precomputed** `evaluations/<eval_id>/predictions.csv` for historical replay. If on-demand inference is supported at all: cap steps or enqueue lightweight worker job — never unbounded full-unit replay inline.
- [ ] **Play without eval artifacts shows blocking message**, not silent sensor-only charts and not unbounded inline inference.
- [ ] Session key includes `(dataset_id, run_id, unit_id, eval_id, policy_hash)`.
- [ ] Implementation complete; `test_alert_policy_cache_invalidation`, `test_replay_log_has_no_future_rows`, and `test_replay_play_advances_without_clicks` pass after subtask 20 (not a gate here).

## Implementation Notes

**Files:** `src/pdm/app.py`, `src/pdm/replay.py`, `src/pdm/evaluate.py`, `src/pdm/worker.py`, `tests/test_worker_and_app.py`

- Default path: load predictions from eval dir; slice to current replay step.
- Optional on-demand: worker job `{"kind": "replay_predict", ...}` with progress; UI polls worker status.
- `_live_prefix_charts` without predictions: disable Play, show "Run Evaluate first" (or worker in progress).
- `st.session_state` migration on key change: reset step, clear stale alerts.
- Fragment: `st.fragment(run_every=timedelta(seconds=0.4))` wrapping controls + charts.

## Dependencies

**Required:** Subtasks 4, 14, 17.

**Soft:** Subtask 16 — eval summary tables on replay screen can follow; core replay fixes (H/K cache, log filter, fragment, Play block, state key) must not wait for 16.

## Verification

```bash
.venv/bin/python -m ruff check src/pdm/app.py src/pdm/replay.py
.venv/bin/python -m pytest tests/test_worker_and_app.py -q -k "app_smoke or screen"  # AppTest smoke: Test & Replay screen loads
```

Manual: H/K change updates or blocks stale status; log capped at replay time; Play blocked without eval artifacts. Full gates: subtask 20.
