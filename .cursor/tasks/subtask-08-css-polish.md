# Subtask 08: CSS polish — typography, sidebar, buttons, cards, alerts, tabs

## Goal
Apply design-spec §3–§5 visual rules that are pure CSS (no Python layout change): type scale, sidebar nav rows, buttons, focus ring, cards/metrics, forms/inputs, expanders, alerts, tabs and chart containers, in both themes.

## Context
After 06/07 colors are correct but the look is still a dev tool: 8–10 px uppercase micro-labels, gradient `.lab-metric` cards, cyan glow on `.lab-status.is-running`, amber focus outline (`explorer.css` L49), stock button/tab/alert styling. File: `src/pdm/visualization/explorer.css` only (plus `theme_css` if a new var is needed).

## Acceptance Criteria
- [ ] Font stack and the 9-role scale from §3 applied to `h1` (st.title), `h2/h3` (st.header/subheader), body, widget labels, captions, metric label/value. No `font-size` below 11 px anywhere in `explorer.css` (including `.lab-eyebrow`, `.lab-metric-label`, `.lab-panel-badge`, `.lab-status`, `.lab-metric-detail`, `.lab-panel-number`, `.lab-panel-subtitle`, `.lab-panel-foot`, mobile media queries).
- [ ] Sidebar nav buttons styled as §5 rows (left-aligned, 36 px, radius 8; selected `accent_soft` + `accent_text`; disabled 45%; hover `control_hover`) using `.st-key-project_nav\:*` / `[data-testid="stSidebar"] [data-testid="stButton"]` selectors. Group-label captions styled per §3.
- [ ] Buttons: primary/secondary/disabled per §5; focus-visible ring = `var(--pdm-focus-ring)` 3 px (replace amber outline).
- [ ] Cards: `stMetric`, `stForm`, `stVerticalBlockBorderWrapper`, `stExpander` per §5; `.lab-metric` gradient removed (flat `surface`); tone accents via 3 px left border only; no glow `box-shadow` on `.lab-status.is-running`.
- [ ] Inputs: height 36, radius 8, hover/focus states per §5.
- [ ] Alerts per §5 (`stAlert`: `surface_subtle`, `text`, 3 px left border by kind); tabs underline style; chart containers (`stPlotlyChart`) radius 12 + 1 px `border`.
- [ ] Main container padding/max-width per §4.
- [ ] Still zero color literals in `explorer.css` (06 test keeps passing).

## Implementation Notes
- Streamlit alert kinds are distinguishable via `[data-testid="stAlertContentInfo|Success|Warning|Error"]` or `[data-baseweb="notification"][kind=…]`; confirm in browser devtools and record the selectors used in a one-line CSS comment only if non-obvious.
- Keep legacy explorer classes (`.lab-hero`, `.lab-metrics`, `.st-key-lab_*`) working; restyle, don't delete.
- No Python edits in this subtask; layout changes belong to 09.

## Dependencies
Subtasks 06 (tokens/CSS pipeline), 07 (chart styling), 04 and 05 (help icons present to style).

## Verification
Add to `tests/test_worker_and_app.py`:
- `test_explorer_css_min_font_size_11px`: `sizes = [float(s) for s in re.findall(r"font-size:\s*(\d+(?:\.\d+)?)px", css)]`; assert `min(sizes) >= 11`; also assert no `font-size` in `rem`/`em` below `0.6875rem` (`r"font-size:\s*(0?\.\d+)rem"` → value × 16 ≥ 11).
- `test_explorer_css_has_no_glow_or_gradient`: no `linear-gradient(` except the slider progress track rule, no `box-shadow:0 0 8px`.
- 06 CSS tests still pass.
- `.venv/bin/python -m pytest tests -q`, `.venv/bin/python -m ruff check src tests`.
- Manual (required for UI claims): `http://127.0.0.1:8501`, all five product screens in Light and Dark; screenshots to `runs/_ui_review/after-css/<screen>-<theme>.png` (gitignored).
