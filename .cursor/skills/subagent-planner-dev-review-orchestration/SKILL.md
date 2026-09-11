---
name: subagent-planner-dev-review-orchestration
description: Orchestrates planner, plan-reviewer, developer, and code-reviewer for large Predictive Maintenance Lab tasks. Use for multi-step epics, data-pipeline changes, loss/model refactors, Streamlit/worker changes, or leakage-protocol work. Sequence: planner → plan-reviewer → per subtask developer → code-reviewer.
---

# Subagent Planner–Plan Review–Dev–Review (PDM Lab)

Orchestrate large tasks: planner decomposes into subtasks, plan-reviewer validates, then per subtask developer implements and code-reviewer validates.

## Workflow Overview

```
Planner → Plan Reviewer → (Planner revise if needed) → [Per subtask: Developer → Code-reviewer → fix loop] → Done
```

## Phase 1: Planner

`subagent_type: "planner"`

- **prompt**: Full scope, paths under `src/pdm/`, leakage/architecture constraints. Create `master-plan.md` and subtask files in `.cursor/tasks/`
- Wait for completion

Output: `.cursor/tasks/master-plan.md`, `.cursor/tasks/subtask-01-*.md`, …

## Phase 2: Plan Review

`subagent_type: "plan-reviewer"`

- **prompt**: Review `.cursor/tasks/` against PDM architecture, leakage rules, GRU/LSTM-only, worker/UI split, verification gates. Return Critical, Warnings, Suggestions, Verdict (`APPROVED` | `REVISE`).

### 2b. Revise until approved

If `REVISE` or Critical/Warnings: re-run planner with feedback, then plan-reviewer.

**Stop / escalate:** `APPROVED` or only Suggestions; same Criticals after 2 cycles; or >10 review cycles → ask the user.

## Phase 3: Discover Subtasks

List `.cursor/tasks/subtask-*.md` (exclude `master-plan.md`), sort by number, read each before handing to developer.

## Phase 4: Per-Subtask Loop

### 4a. Developer — full subtask (Goal, AC, notes, verification)

### 4b. Code-reviewer — AC + leakage, loss math, worker/UI, tests

### 4c. Fix loop until no Critical/Warnings

Escalate if stuck (same Criticals twice) or >10 fix cycles on one subtask.

## Phase 5: Completion

Summarize what shipped, completed subtasks, leftover suggestions. Do not treat `--smoke` metrics as model quality.

## Checklist

```
- [ ] Phase 1: planner
- [ ] Phase 2: plan-reviewer
- [ ] Phase 2b: revise until APPROVED
- [ ] Phase 3: sort subtasks
- [ ] For each subtask: developer → reviewer → fix loop
- [ ] Summarize
```

## Example Flow

**User**: "Add spectral-band ablation flag for bearings"

1. **planner**: config + `features.py` + prepare cache invalidation + invariant test + UI advanced option
2. **plan-reviewer**: flags missing train-only scaler note; planner adds it
3. **developer** 01: feature flag in `configs/bearings.yaml` / `features.py`
4. **code-reviewer**: bands still fixed before test; no test-set tuning
5. Repeat for train/UI/tests

## Agent References

- Planner: `.cursor/agents/planner.md`
- Plan Reviewer: `.cursor/agents/plan-reviewer.md`
- Developer: `.cursor/agents/developer.md`
- Code Reviewer: `.cursor/agents/code-reviewer.md`

## Notes

- All commands use `.venv/bin/python`
- Data/split/preprocess subtasks: `pytest tests/test_spec_invariants.py`
- Loss/model subtasks: NLL/RUL unit tests + finite grads
- UI/worker subtasks: `tests/test_worker_and_app.py` (AppTest)
- Follow planner dependency order after approval
