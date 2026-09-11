# Orchestration

Execute planner–plan review–dev–review for a **complex task or list of tasks** in Predictive Maintenance Lab.

**Task**: Use the text after `/orchestration` as the task. If none provided, ask what to orchestrate.

## Workflow

Read `.cursor/skills/subagent-planner-dev-review-orchestration/SKILL.md` and apply it.

Sequence: planner → plan-reviewer → planner (revise if needed) → for each subtask (developer → code-reviewer → fix loop) → done

## Repo constraints (PDM Lab)

- **Root**: repo root (`src/pdm/`)
- **Stack**: Python · PyTorch GRU/LSTM · Streamlit · pytest · ruff
- Decompose across data/splits, model/loss, train/eval, UI/worker, tests
- Leakage/split/preprocess: plan tests in `tests/test_spec_invariants.py`
- Loss/model: plan NLL/RUL numeric tests + finite grads
- UI/worker: plan AppTest in `tests/test_worker_and_app.py`
- Do not plan FastAPI/React, extra architectures, or unread MAT `filters_full_history`

## Verification baseline

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check src tests
```
