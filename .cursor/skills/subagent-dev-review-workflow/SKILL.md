---
name: subagent-dev-review-workflow
description: Orchestrates developer and code-reviewer subagents for Predictive Maintenance Lab implementation. Use when the user requests implementation, a feature, a bugfix, or any development task. Developer implements first, code-reviewer validates, then developer applies fixes if needed.
---

# Subagent Dev–Review Workflow (PDM Lab)

Orchestrate subagents: developer implements, code-reviewer validates, developer applies fixes.

## Workflow Steps

### 1. Launch Developer

Use the Cursor Subagent/Task tool with `subagent_type: "developer"`:

- **description**: Short summary (3–5 words)
- **prompt**: Task, concrete paths under `src/pdm/` / `tests/` / `configs/`, and constraints (no leakage, GRU/LSTM only, worker vs UI, `.venv` Python)
- Wait for completion before step 2

### 2. Launch Code Reviewer

After the developer finishes, use `subagent_type: "code-reviewer"`:

- **description**: Short summary (e.g. "Review train leak fix")
- **prompt**: Review the developer's work. Focus on leakage, split integrity, loss/head correctness, checkpoint/dataset isolation, worker/UI, and tests in `tests/test_spec_invariants.py`
- Wait for completion

### 3. Conditional: Re-run Developer

If the reviewer reports **Critical** or **Warnings**:

- Re-launch `developer` with the fix list (optionally `resume` the previous agent ID)
- Re-launch `code-reviewer`
- Repeat once more only if critical/warnings remain

Suggestions are optional. Escalate if the same Criticals persist after 2 cycles.

## Checklist

```
- [ ] Step 1: Launch developer
- [ ] Step 2: Wait for developer
- [ ] Step 3: Launch code-reviewer
- [ ] Step 4: Parse critical / warnings / suggestions
- [ ] Step 5: If critical/warnings → developer with fix list
- [ ] Step 6: Re-run code-reviewer
- [ ] Step 7: Clean review → done
```

## Example Flow

**User**: "Future rows change the prediction at t"

1. **developer**: "Trace `Predictor` / `ReplaySource.prefix`; assert prefix-only features; add invariant test."
2. **code-reviewer**: "Confirm no test RUL in `predict.py`; scaler still train-only."
3. **developer** (if needed): "Apply fixes; `.venv/bin/python -m pytest tests -q`."

## Repo Rules

- Package: `src/pdm/` — not a JS monorepo
- Leakage tests belong in `tests/test_spec_invariants.py`
- UI: `src/pdm/app.py` + AppTest in `tests/test_worker_and_app.py`
- Commands: `.venv/bin/python -m pytest tests -q` and `ruff check src tests`
- Do not run full 30-epoch training unless the user asks; `--smoke` is not a quality claim

## Agent References

- Developer: `.cursor/agents/developer.md`
- Code Reviewer: `.cursor/agents/code-reviewer.md`
