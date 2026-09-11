# Subtask 19: Demo / historical replay presentation (Section 6)

## Goal

Polish replay demo UX: model + unit selection, split label, sensor/RUL/baseline charts, H display, warning time, end-of-unit summary; honest censored end messaging.

## Context

Section 6 Demo/Historical Replay. GT is evaluator overlay only. Filter file ending before 600 Pa → `End of observed data`, not fake threshold crossing.

## Acceptance Criteria

- [ ] Replay header: saved model run_id, split label, eval_id (if historical), H_trigger.
- [ ] Charts: sensor (RMS or Δp + 600 Pa line), RUL prediction + baseline; GT overlay optional, does not affect prediction.
- [ ] Current position marker; warning episode time when active.
- [ ] End-of-unit card: unit metrics vs overall test table row.
- [ ] Censored / truncated filter: message `End of observed data` when `observation_end_s` reached without event; no invented 600 Pa.
- [ ] Official test RUL shown only as evaluation annotation, not sensor fact.

## Implementation Notes

**Files:** `src/pdm/app.py`, `src/pdm/replay.py`, `src/pdm/evaluate.py`

- Check `event_observed` and `differential_pressure` vs limit for display logic.
- Use Plotly separate y-axes (pressure vs RUL).
- CSV export of visible log slice.

## Dependencies

**Required:** Subtask 18.

**Soft:** Subtask 16 — end-of-unit comparison vs overall test table can use eval UI from 16 when available.

## Verification

```bash
.venv/bin/python -m ruff check src/pdm/app.py src/pdm/replay.py
```

Manual optional: `streamlit run src/pdm/app.py` on 127.0.0.1:8501. Replay behavior gates: subtask 20.
