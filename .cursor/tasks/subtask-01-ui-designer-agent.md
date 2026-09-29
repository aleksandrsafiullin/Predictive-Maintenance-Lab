# Subtask 01: Create the `ui-designer` agent

## Goal
Create `.cursor/agents/ui-designer.md`, a design-expert agent that owns the visual language of this Streamlit lab and produces/maintains `.cursor/tasks/design-spec.md`.

## Context
Alex asked for a separate design expert whose job is to make the UI look like a premium, Apple-grade product. Existing agents (`developer`, `code-reviewer`, `planner`, `plan-reviewer`) are engineering-focused. This file is the only deliverable; no source changes.

## Acceptance Criteria
- [ ] File `.cursor/agents/ui-designer.md` exists with frontmatter in the same shape as `.cursor/agents/developer.md`:
  ```yaml
  ---
  name: ui-designer
  description: Visual/UX design expert for the Predictive Maintenance Lab Streamlit app. Use for Streamlit visual design of this lab — dual light/dark theme tokens, typography, spacing, component styling, Plotly chart styling, and plain-English UI copy. Produces and maintains .cursor/tasks/design-spec.md.
  ---
  ```
  (`model:` line optional; if present copy the style used by `developer.md`.)
- [ ] Body encodes, as hard rules:
  - Apple-like restraint: one type scale, generous whitespace, one accent color, 1px hairline borders, no rainbow palettes, no emoji or decorative glyphs in chrome, no glassmorphism/blur/glow/gradient gimmicks.
  - Light and Dark are both first-class palettes; a screen must contain zero surface/text colors from the other theme; every token has a value for both.
  - Streamlit + Plotly only (CSS via `st.markdown(unsafe_allow_html=True)`, `st.container`, `help=`, config.toml). No React, FastAPI, custom SPA, or new JS frameworks. Existing WebGL component internals are out of scope.
  - UI copy in English; plain-language help rules (what it is + how it changes the result, ≤220 chars, jargon translated once).
  - Never invent data fields, metrics, or columns; never touch model, loss, splits, scalers, leakage rules, worker protocol, or defaults.
  - Accessibility: text contrast ≥ 4.5:1 against its background, focus-visible ring on all interactive elements.
- [ ] Body defines the deliverable: `.cursor/tasks/design-spec.md` with token tables (both themes), type scale, spacing/radius, component rules (sidebar, page header, cards/metrics, forms, buttons, tables, tabs, alerts, charts, empty states, expanders), config.toml guidance, and copy rules. "TBD" is not allowed in a signed-off spec.
- [ ] Body defines the review workflow: read `src/pdm/ui_theme.py` (once it exists), `src/pdm/visualization/explorer.css`, product screens in `src/pdm/project_ui.py`, `project_quality_ui.py`, `project_training_ui.py`, `project_results_ui.py`; take screenshots of `http://127.0.0.1:8501` in both themes; output a punch list keyed by file.
- [ ] Body lists the files it may edit (`.cursor/tasks/design-spec.md`, and only on explicit request `src/pdm/visualization/explorer.css`, `src/pdm/ui_theme.py` token values, `src/pdm/ui_copy.py` strings) and files it must never edit (`train.py`, `losses.py`, `splits.py`, `signal_training.py`, `worker.py`, `data/**`).
- [ ] No other file changes.

## Implementation Notes
- Keep it under ~120 lines; mirror tone/structure of `.cursor/agents/developer.md` (role line, When invoked, Rules, Deliverable, Output format).
- Reference `Cursor_Predictive_Maintenance_MVP_Spec.md` for product constraints and `.cursor/rules/project-core.mdc` for stack limits.
- Writing under `.cursor/` may need sandbox-escalated permissions in this environment.

## Dependencies
None.

## Verification
- `test -f ".cursor/agents/ui-designer.md" && head -5 .cursor/agents/ui-designer.md` shows `name: ui-designer`.
- `rg -n "glassmorphism|rainbow|React|Light|Dark|help=" .cursor/agents/ui-designer.md` returns hits for each rule.
- No pytest node required (docs-only); `.venv/bin/python -m ruff check src tests` still clean.
