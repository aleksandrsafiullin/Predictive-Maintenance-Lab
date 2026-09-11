# Implementation

Execute dev–review workflow for an **easy task or fix** in Predictive Maintenance Lab.

**Task**: Use the text after `/implementation` as the task. If none provided, ask what to implement.

## Workflow

Read `.cursor/skills/subagent-dev-review-workflow/SKILL.md` and apply it.

Sequence: developer → code-reviewer → developer (fix if needed) → code-reviewer

## Repo constraints (PDM Lab)

- **Product**: local GRU/LSTM lab — XJTU-SY bearings RUL + HSE filter time-to-600 Pa
- **Root**: this repo (`src/pdm/`, not a JS monorepo)
- **Stack**: Python 3.11+ · PyTorch · Streamlit · Plotly · pytest · ruff
- **Data**: `src/pdm/data/`, `configs/*.yaml`
- **Train/eval**: `train.py`, `evaluate.py`, `models.py`, `losses.py`
- **UI**: `src/pdm/app.py` on `127.0.0.1:8501`; heavy jobs in `worker.py`
- **Verify**: `.venv/bin/python -m pytest tests -q` and `ruff check src tests`

## Key references

- `README.md`
- `Cursor_Predictive_Maintenance_MVP_Spec.md`
- `configs/bearings.yaml`, `configs/filters.yaml`
