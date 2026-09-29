# Subtask 02: Design spec sign-off (ui-designer)

## Goal
Run the `ui-designer` agent against the planner draft `.cursor/tasks/design-spec.md` and leave a signed-off, fully concrete spec (both theme token tables, type, spacing, component rules, copy rules) that subtasks 03–09 implement without further design decisions.

## Context
The planner already drafted the spec with real values (no TBD) and WCAG-checked ratios so developers are never blocked. Downstream: 03 (theme module), 06 (CSS), 07 (charts), 08 (CSS polish), 09 (layout) implement it; 04/05 use §7 copy rules. The designer's job here is to audit and tighten it against the actual screens, not to start over. Current UI evidence: `src/pdm/visualization/explorer.css` (cyan `#55c9dc` accent, 8–10 px micro-labels, gradient metric cards), `src/pdm/project_ui.py` sidebar (nav buttons, duplicate-prone theme selectbox), Plotly helpers in `project_chart_style.py`, `lab_ui.style_figure`, `visualization/overlay.py`.

## Acceptance Criteria
- [ ] Header `Status:` changed from `DRAFT` to `SIGNED OFF by ui-designer, <date>`; a short "Changes from draft" list at the bottom (may be empty).
- [ ] Every token in sections 2.1–2.3 has a Light and a Dark value; no `TBD`, no "similar to", no ranges.
- [ ] Contrast table §2.4 (already filled by planner with computed ratios, including `muted`/`sidebar`, `accent_text`/`accent_soft`, `success`/`surface`, `text`/`control`, `accent_ink`/`accent`) is re-verified; if the designer changes any value, the ratio is recomputed and every text pair stays ≥ 4.5:1. `zone_yellow` stays lines/fills only.
- [ ] Section 2.5 exclusivity still holds after any edits: no value of a Dark surface/text token equals any Light token value and vice versa (no exceptions).
- [ ] §5 Tables decision (`st.table` on product screens) is kept or explicitly replaced with a single alternative; no open choice left for the developer.
- [ ] Component rules cover: sidebar, page header, cards/metrics, forms/inputs, buttons, tables, tabs, alerts, charts, empty states, expanders, `help=` tooltip bubble.
- [ ] Native-surface rules present (file uploader, number steppers, radio/checkbox, progress, toast/status, header toolbar, scrollbars via `color-scheme`, tooltip, popover, select, code).
- [ ] Section 7 copy rules present and unchanged in intent (what + effect, ≤220 chars, jargon translated).
- [ ] Explicit per-screen notes for the five product screens (Projects, Import data, Data Quality, Training, Results): what the primary action is, which elements are cards, which chart height.
- [ ] No source code changes in this subtask.

## Implementation Notes
- Invoke via Task tool with `subagent_type` matching `.cursor/agents/ui-designer.md` (after subtask 01); if the harness cannot resolve custom agent types, run the same prompt through `generalPurpose` and paste the agent body as instructions.
- Optional but recommended: start `.venv/bin/python -m pdm app`, capture current Light and Dark screenshots of the five product screens to `runs/_ui_review/before/` (gitignored) and reference them in the per-screen notes.
- Keep the spec a single file; do not create additional markdown files.

## Dependencies
Subtask 01.

## Verification
- `rg -n "TBD|DRAFT" .cursor/tasks/design-spec.md` returns nothing.
- `rg -c "\| \`" .cursor/tasks/design-spec.md` shows the token tables intact.
- Manual: plan-reviewer or Alex reads the per-screen notes and confirms they describe the intended look.
- No pytest node (docs-only). `.venv/bin/python -m ruff check src tests` still clean.
