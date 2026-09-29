# Design Spec: Predictive Maintenance Lab (Streamlit)

**Status:** SIGNED OFF by ui-designer, 2026-09-29. Planner version audited against `project_ui.py`, `project_quality_ui.py`, `project_training_ui.py`, `project_results_ui.py`, `visualization/explorer.css`, `project_chart_style.py`, `.streamlit/config.toml` and Streamlit 1.63.0; changes listed at the bottom.
**Implemented by:** subtasks 03, 06, 07, 08, 09. Tokens live in code in exactly one place: `src/pdm/ui_theme.py` (`TOKENS["light"]`, `TOKENS["dark"]`). CSS and Plotly read from there. No color literal anywhere else in UI code.

## 1. Principles

- Restraint: one type family, one type scale, one accent, hairline borders, generous whitespace.
- No rainbow, no gradients on surfaces, no glow, no glassmorphism, no emoji or decorative glyphs in chrome (nav markers like `✓ 01` go away).
- Both themes first-class. A screen in Light must contain **zero** surface/text colors from Dark and vice versa.
- Hierarchy through size, weight and space, not color. Color means state (accent = interactive/selected, success/warning/danger = status).
- One primary button per view (main area). The sidebar has no primary buttons; the selected nav row is a selection state, not a primary action.
- Every user-set control has plain-English `help=`. Copy rules in section 7.

## 2. Color tokens

Token names map 1:1 to CSS custom properties `--pdm-<name>` (underscores → hyphens) and to keys in `TOKENS[theme]`. Hex values are written uppercase in `ui_theme.py`; tests compare case-insensitively.

### 2.1 Surfaces, text, lines

| Token | Light | Dark | Use |
|-------|-------|------|-----|
| `bg` | `#F5F5F7` | `#0F0F11` | App background (main area) |
| `sidebar` | `#EDEDF0` | `#161618` | Sidebar background |
| `surface` | `#FFFFFF` | `#1C1C1E` | Cards, forms, charts, tables, metrics, tooltip, popover |
| `surface_subtle` | `#FAFAFC` | `#232326` | Alerts, expanders, file drop zone, table header, code |
| `input` | `#FFFFFF` | `#1C1C1E` | Text/number/select fields |
| `control` | `#FFFFFF` | `#2C2C2E` | Secondary button fill, number-input steppers, segmented-control track |
| `control_hover` | `#F0F0F3` | `#3A3A3C` | Secondary button hover, nav row hover, hovered option |
| `text` | `#1D1D1F` | `#F2F2F4` | Primary text, values |
| `text_secondary` | `#424245` | `#D1D1D6` | Body copy under titles, code text |
| `muted` | `#636366` | `#98989D` | Captions, labels, axis text, sidebar group labels, disabled nav |
| `border` | `#D2D2D7` | `#38383A` | Hairlines on inputs, cards |
| `border_soft` | `#E5E5EA` | `#2C2C2E` | Dividers, chart grid, table row separators, progress/slider track |
| `border_strong` | `#C7C7CC` | `#48484A` | Hover border on inputs, table header rule, radio/checkbox ring, dropzone dash |

### 2.2 Accent and status

| Token | Light | Dark | Use |
|-------|-------|------|-----|
| `accent` | `#0071E3` | `#0A84FF` | Primary button fill, focus outline, forecast series, radio/checkbox checked, progress/slider fill. **Never text.** |
| `accent_hover` | `#0062C4` | `#409CFF` | Primary hover fill |
| `accent_ink` | `#FFFFFF` | `#000000` | Text/icons on `accent` / `accent_hover` fills |
| `accent_soft` | `#E8F1FC` | `#10263D` | Selected nav row bg, selected option bg, selected segment bg |
| `accent_text` | `#0058B0` | `#64B5FF` | Text on `accent_soft` (selected nav, selected option, selected segment), links everywhere |
| `success` | `#1E7B34` | `#30D158` | Success text/border, green zone |
| `warning` | `#B35C00` | `#FF9F0A` | Warning text/border |
| `danger` | `#D70015` | `#FF453A` | Error text/border, red zone, red limit |
| `focus_ring` | `rgba(0,113,227,0.28)` | `rgba(10,132,255,0.35)` | 3px soft halo on focused **inputs only**, in addition to the solid `accent` border. Never the sole focus indicator. |

### 2.3 Chart tokens

| Token | Light | Dark | Use |
|-------|-------|------|-----|
| `chart_bg` | `#FFFFFF` | `#1C1C1E` | `paper_bgcolor` and `plot_bgcolor` (= `surface`) |
| `chart_text` | `#636366` | `#98989D` | Tick labels, axis titles, legend, all annotations incl. "Now" and limit labels (= `muted`) |
| `chart_grid` | `#E5E5EA` | `#2C2C2E` | Y grid only; X grid off |
| `chart_axis` | `#D2D2D7` | `#38383A` | Axis lines, zero line |
| `series_observed` | `#6E6E73` | `#A1A1A6` | Measured signal / actual (line only). Dark is `#A1A1A6`, not `#AEAEB2`, so it does not collide with Light `zone_unknown`. |
| `series_forecast` | `#0071E3` | `#0A84FF` | Model forecast / prediction |
| `series_band` | `rgba(0,113,227,0.10)` | `rgba(10,132,255,0.16)` | Forecast interval fill |
| `series_band_line` | `rgba(0,113,227,0.45)` | `rgba(10,132,255,0.45)` | Interval edges |
| `series_reference` | `#1D1D1F` | `#F2F2F4` | "Now" cursor / as-of line (dotted, 1px) |
| `zone_green` | `#1E7B34` | `#30D158` | Health zone green |
| `zone_yellow` | `#C28800` | `#FFD60A` | Health zone yellow — **lines, markers and fills only, never text** (Light 3.08:1 fails as text) |
| `zone_red` | `#D70015` | `#FF453A` | Health zone red, red limit line, expected-red-entry marker |
| `zone_unknown` | `#AEAEB2` | `#7C7C80` | Unknown / gray zone — **area fills and markers with a text label only, never a standalone line** (Light 2.21:1 vs `chart_bg`) |
| `zone_yellow_fill` | `rgba(194,136,0,0.08)` | `rgba(255,214,10,0.08)` | Yellow band fill |
| `zone_red_fill` | `rgba(215,0,21,0.06)` | `rgba(255,69,58,0.09)` | Red band fill |
| `series_cycle` | `["#0071E3", "#6E6E73", "#B35C00", "#1E7B34", "#8944AB", "#D70015"]` | `["#0A84FF", "#A1A1A6", "#FF9F0A", "#30D158", "#BF5AF2", "#FF453A"]` | Multi-model comparisons only, in this order. Dark gray entry is `#A1A1A6` (= Dark `series_observed`). |
| `heatmap_scale` | `[[0,"#FFFFFF"],[0.5,"#9CC3F0"],[1,"#0071E3"]]` | `[[0,"#1C1C1E"],[0.5,"#1F4E80"],[1,"#0A84FF"]]` | Recurrent trace heatmap (sequential, single hue) |

### 2.4 Contrast (WCAG 2.x relative luminance, re-verified 2026-09-29 by script)

**Text pairs — all ≥ 4.5:1.**

| Pair | Light | Dark |
|------|-------|------|
| `text` / `bg` | 15.46 | 17.13 |
| `text` / `surface` | 16.83 | 15.22 |
| `text` / `surface_subtle` | 16.14 | 14.02 |
| `text` / `sidebar` | 14.40 | 16.16 |
| `text` / `control` | 16.83 | 12.47 |
| `text` / `control_hover` | 14.80 | 10.15 |
| `text_secondary` / `bg` | 9.20 | 12.58 |
| `text_secondary` / `surface` | 10.01 | 11.18 |
| `muted` / `bg` | 5.50 | 6.67 |
| `muted` / `surface` | 5.99 | 5.93 |
| `muted` / `sidebar` | 5.12 | 6.29 |
| `muted` / `surface_subtle` | 5.74 | 5.46 |
| `accent_text` / `surface` (links) | 6.95 | 7.77 |
| `accent_text` / `bg` (links) | 6.38 | 8.75 |
| `accent_text` / `accent_soft` (selected nav/option/segment) | 6.09 | 7.02 |
| `accent_ink` / `accent` (primary button) | 4.70 | 5.76 |
| `accent_ink` / `accent_hover` | 5.93 | 7.42 |
| `success` / `surface` | 5.33 | 8.42 |
| `success` / `surface_subtle` | 5.12 | 7.75 |
| `warning` / `surface` | 4.72 | 8.28 |
| `warning` / `surface_subtle` | 4.53 | 7.62 |
| `danger` / `surface` | 5.38 | 4.99 |
| `danger` / `surface_subtle` | 5.16 | 4.60 |

**Forbidden text pairs (fail 4.5:1) — never render these:**

| Pair | Light | Dark | Use instead |
|------|-------|------|-------------|
| `accent` as text on `accent_soft` | 4.12 | 4.21 | `accent_text` |
| `accent` as text on `bg` | 4.31 | 5.25 | `accent_text` (rule: `accent` is never text in either theme) |
| `muted` on `control_hover` | 5.26 | 3.95 | Hover states switch text to `text` |
| `zone_yellow` on `surface` | 3.08 | 12.05 | Label in `chart_text`; yellow only as line/fill |
| `warning` as text on `bg` | 4.34 | 9.31 | Status text (`success` / `warning` / `danger`) only on `surface` or `surface_subtle`, never on `bg`. Native `:orange[...]` and colored captions are not used. |

**Non-text (WCAG 1.4.11, ≥ 3:1 required where the graphic carries meaning):**

| Pair | Light | Dark | Verdict |
|------|-------|------|---------|
| `accent` / `bg` (focus outline) | 4.31 | 5.25 | pass |
| `accent` / `surface` (focus outline, forecast line) | 4.70 | 4.66 | pass |
| `accent` / `sidebar` (focus outline in sidebar) | 4.02 | 4.95 | pass |
| `series_observed` / `chart_bg` | 5.07 | 6.61 | pass |
| `zone_yellow` / `chart_bg` (limit line) | 3.08 | 12.05 | pass |
| `zone_red` / `chart_bg` | 5.38 | 4.99 | pass |
| `zone_unknown` / `chart_bg` | 2.21 | 4.09 | fill + label only (see §2.3) |
| `focus_ring` halo / `surface` | 1.49 | 1.61 | decorative only; the solid `accent` outline/border carries focus |

Disabled controls are exempt from contrast (WCAG 1.4.3) but must stay legible; see sidebar disabled rule.

### 2.5 Cross-theme exclusivity (tested)

- Verified 2026-09-29: the **full** Light token value set and the **full** Dark token value set are disjoint (every hex and rgba value, all keys in §2.1–2.3). Any future edit must keep that true.
- `SURFACE_AND_TEXT_KEYS = (bg, sidebar, surface, surface_subtle, input, control, control_hover, text, text_secondary, muted, border, border_soft, border_strong)`.
- `DARK_EXCLUSIVE = {TOKENS["dark"][k] for k in SURFACE_AND_TEXT_KEYS}` must not appear (case-insensitive) in `theme_css("light")` or Light Plotly JSON; `LIGHT_EXCLUSIVE` likewise for Dark. No exceptions. Dark `accent_ink` is `#000000`, so `#FFFFFF` never appears in Dark output.
- Each theme's CSS block emits **only** that theme's values; `theme_css("light")` never contains a Dark token value of any key, and vice versa.
- Legacy literals that must disappear from all product-screen code paths (case-insensitive match):
  - `explorer.css` dark set: `#0b0f14 #10151c #141a22 #19212b #111820 #151d26 #1a232d #222e3a #2a3542 #202a35 #384656 #354351 #253a4b #f3f5f7 #d8e0e8 #aab7c5 #8493a3 #55c9dc #83dce8 #071317 #e7b866 #f0d39f #4ca982 #d36d6d`
  - `explorer.css` light set: `#f4f5f7 #eceef1 #f7f8fa #f8f9fb #f4f6f8 #e9edf1 #d9dee5 #e6e9ee #c6ced8 #d9e0e7 #deeff2 #20252b #303943 #5b6672 #737f8c #087f91 #086c7b #96620a #805507 #237a53 #b34444 #34414d #4c5661 #697480 #f0f2f5 #e8ebef #e2e7eb`
  - Chart helpers: `#26313b #e3e8ee #586675 #8998aa #9aa7b5 #1769b4 #4c9bff #946600 #f2c44d #b43e3e #f36c68 #4f5e6d #dce5ef` and the `rgba(23,105,180,…)`, `rgba(76,155,255,…)`, `rgba(180,62,62,…)`, `rgba(243,108,104,…)`, `rgba(148,102,0,…)`, `rgba(242,196,77,…)` fills
  - Older palette: `#182536 #15202c #61d8ee #5bd3f5 #2ecc71 #f1c40f #e74c3c`
  - Plotly templates `plotly_dark` / `plotly_white` (use `template="none"` + explicit tokens)
  - The light-only `--gdg-*` block in `explorer.css` (no Glide variable overrides anywhere)

## 3. Typography

Font stack (Alex is on Windows → Segoe UI; macOS → SF):
`-apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", "Inter", Roboto, "Helvetica Neue", Arial, sans-serif`.
Numbers in metrics, tables, slider values and chart ticks: `font-variant-numeric: tabular-nums`.

Size ramp (the only sizes allowed, px): **11, 12, 13, 14, 15, 17, 28**. Weights allowed: **400, 500, 600**.

| Role | Size / line-height | Weight | Tracking | Color |
|------|-------------------|--------|----------|-------|
| Page title (`st.title`) | 28 / 34 | 600 | −0.02em | `text` |
| Page description (first line under title) | 15 / 22 | 400 | 0 | `text_secondary`, max-width 720px |
| Section (`st.subheader`, `st.header`) | 17 / 24 | 600 | −0.01em | `text` |
| Empty-state title | 17 / 24 | 600 | −0.01em | `text` |
| Body, alert text | 14 / 21 | 400 | 0 | `text` |
| Button | 14 / 20 | 600 primary, 500 secondary | 0 | per button rule |
| Tab label | 14 / 20 | 500 (selected 600) | 0 | per tab rule |
| Expander summary | 14 / 20 | 500 | 0 | `text` |
| Nav row | 14 / 20 | 500 (selected 600) | 0 | per sidebar rule |
| Label (widget) | 13 / 18 | 500 | 0 | `text` |
| Table cell | 13 / 18 | 400 | 0 | `text` |
| Tooltip / popover body | 13 / 18 | 400 | 0 | `text` |
| Sidebar wordmark | 13 / 18 | 600 | 0 | `text` |
| Caption, chart text | 12 / 17 | 400 | 0 | `muted` |
| Metric label | 12 / 16 | 500 | 0, sentence case | `muted` |
| Table header | 12 / 16 | 600 | 0 | `muted` |
| Sidebar group label (`PROJECT`, `WORKFLOW`, `APPEARANCE`) | 11 / 14 | 600 | 0.06em, uppercase | `muted` |
| Metric value | 28 / 32 | 600 | −0.02em | `text` |

No other sizes or weights. No all-caps outside sidebar group labels. Minimum 11 px anywhere (kills the 8–10 px `.lab-metric-label`, `.lab-panel-badge`, `.lab-status`, `.lab-eyebrow`, `.lab-specimen-label`, `.lab-panel-number`, `.lab-panel-subtitle`, `.lab-panel-foot`, `.lab-metric-detail`). The caption override `font-size: .78rem` in `explorer.css` is replaced by the 12 px caption role.

## 4. Spacing, radius, elevation, motion

- Spacing scale (px): **4 8 12 16 24 32 48 64**. No other gap, margin or padding values in CSS.
- Main container padding `32px 48px 64px`; max content width 1200 px on Projects, Import data, Data Quality, Training; 1600 px on Results (replay chart benefits from width).
- Title → description 8; description → first content 24; section → section 32; card padding 24; metric padding 16; gap between cards/metrics 12; gap between form fields 16.
- Radius: controls (buttons, inputs, selects, nav rows, tooltip, popover, segmented control) 8; cards, forms, charts, tables, alerts, toasts, expanders, dropzone 12; pills 999. No other radii (the 9, 10, 14 px values in `explorer.css` go away).
- Borders: 1px `border`; no double borders (card inside form, table inside scroll container → drop the inner border).
- Elevation: Light cards `0 1px 2px rgba(0,0,0,0.04)`; Dark none. Popover/tooltip: Light `0 4px 16px rgba(0,0,0,0.08)`; Dark none (border carries the edge).
- Motion: 150 ms ease-out on background, border and color only. No transforms, no pulsing, no `filter: brightness`.

## 5. Component rules

States for every interactive element: default, hover, focus-visible, disabled. Focus-visible everywhere is a **2px solid `accent` outline, 2px offset** (≥ 3:1 on `bg`, `surface`, `sidebar` per §2.4). Inputs additionally show the `focus_ring` halo. The amber focus outline in `explorer.css` is removed.

### Sidebar
- Background `sidebar`, right hairline `border_soft`. Padding 24 16.
- Top: wordmark "Predictive Maintenance Lab" per type table; 24 below.
- Group labels (`PROJECT`, `WORKFLOW`, `APPEARANCE`) per type table; 24 above each group except the first, 8 below.
- Project selector: standard select (see Popovers, selects), full width.
- Nav = full-width rows, **left-aligned** text, height 36, radius 8, padding 0 12, transparent bg, no border, `text`. Gap between rows 4.
  - Hover: `control_hover` bg, `text`.
  - Selected: `accent_soft` bg + `accent_text` text, weight 600. Never `accent` fill, never `accent` text.
  - Disabled (step locked): `muted` text at full opacity, transparent bg, no hover, `not-allowed` cursor. (A 45% opacity step measured 1.87:1 Light / 2.25:1 Dark — too faint to read the workflow.)
  - Focus-visible: global outline rule.
  - No numbers, no check glyphs, no icons.
  - Implementation hook: the code keeps `type="primary"` on the selected nav button only as a selector; CSS scoped to `[data-testid="stSidebar"]` restyles `button[kind="primary"]` to the selected-row rule and `button[kind="secondary"]` to the default-row rule. Nothing in the sidebar renders as a primary button.
- Appearance control: `st.segmented_control` with `Light` / `Dark`, exactly one instance, full width. Track `control` bg, 1px `border`, radius 8; unselected segment `text` 13/500 on `control`, hover `control_hover`; selected segment `accent_soft` bg + `accent_text` 600. Behavior in subtask 03.
- Worker/job details in a collapsed expander at the bottom of the sidebar.

### Page header
- `st.title` + one-sentence description (`st.write`, styled as page description). No eyebrow labels, no decorative dots (drop `.lab-title-dot`), no hero banners, no `.lab-specimen` block on product screens.
- Heading anchor links (Streamlit "Link to heading" icon): hidden on product screens (`[data-testid="stHeaderActionElements"] { display: none }`).
- Job/import status alerts render directly below the description, above all inputs.

### Cards and metrics
- `st.container(border=True)` → `surface`, 1px `border`, radius 12, padding 24, Light elevation per §4.
- `st.metric` → card style with padding 16; label/value per type table; equal columns, gap 12. Metric value uses `text` in every tone (no accent/amber tinted values, drop `.lab-tone-*` value colors).
- Status tones only via a 3px left border in `success`/`warning`/`danger`; never tinted card backgrounds, never gradients (drop the `.lab-metric` gradient and `:before` bar).

### Forms and inputs
- `st.form` → `surface` card, padding 24, 1px `border`, radius 12; nested containers have no border.
- Inputs (text, number, select, text area): height 36, radius 8, bg `input`, 1px `border`, text `text` 14/400, placeholder `muted`.
  - Hover: border `border_strong`.
  - Focus: border `accent` + 3px `focus_ring` halo.
  - Disabled: 40% opacity, `not-allowed` cursor.
- Number-input steppers: `control` bg, `text` icon, left hairline `border`; hover `control_hover`; focus-visible outline rule.
- Radio / checkbox: unchecked ring 1px `border_strong` on `input`; checked fill `accent` with dot/check `accent_ink`; label `text` 14/400; focus-visible outline on the control; disabled 40% opacity.
- Slider (Results "Observation time"): track 4px `border_soft`, filled part `accent`, thumb 16px `surface` with 1px `border_strong` and a 2px `accent` ring; value label `muted` 12 tabular; tick bar `muted`. Drop the `color-mix` glow and the `linear-gradient` track trick; use the track/fill colors directly.
- `help=` tooltip icon: `muted`, hover `text`. Tooltip bubble: `surface` bg, `text` 13/18/400, 1px `border`, radius 8, padding 8 12, max-width 320, popover elevation per §4.
- Max 3 fields per row. Primary submit directly after the last field, left-aligned, auto width.

### File uploader
- Dropzone `surface_subtle`, 1px dashed `border_strong`, radius 12, padding 24; hover border `accent`; instructions `muted` 12; file-name rows `text` 13 with `muted` size.
- "Browse files" button = secondary button (no `!important` override stacks beyond what Streamlit needs).
- Uploaded file chips: `surface` bg, 1px `border_soft`, radius 8; delete icon `muted`, hover `danger`.

### Buttons
- Primary: `accent` bg, `accent_ink` text 14/600, no border, radius 8, height 36, padding 0 16; hover `accent_hover`; focus-visible outline rule; disabled 40% opacity + `not-allowed`.
- Secondary: `control` bg, `text` 14/500, 1px `border`; hover `control_hover` bg + `border_strong` border (not `accent`); focus-visible outline rule; disabled 40% opacity.
- Destructive actions (Delete project) use the secondary style; the confirming checkbox and the label carry the meaning. No red-filled buttons.
- Buttons inside `st.form` follow the same rules (`[data-testid="stFormSubmitButton"]` selectors in addition to `[data-testid="stButton"]`).

### Tables
- **Decision:** product screens use `st.table` (HTML, CSS-stylable). No `st.dataframe` on product screens.
  - Data Quality measurement table (`project_quality_ui.py`): `st.table` inside `st.container(height=240, border=True)`; the container is the card (1px `border`, radius 12, `surface`); the table inside has no outer border; header row is `position: sticky; top: 0` with `surface_subtle` bg so it stays visible while scrolling.
  - Results per-horizon table (`project_results_ui.py`): plain `st.table`, no height container (one row per horizon); the table wrapper gets 1px `border`, radius 12, `overflow: hidden`.
  - Same columns as today in both; no sorting, selection or editing.
- Style: header bg `surface_subtle`, header text `muted` 12/600, header bottom rule 1px `border_strong`; row separators 1px `border_soft`; cell text `text` 13/400, cell padding 8 12; numeric columns tabular and right-aligned, text columns left-aligned; no zebra striping; no index column.
- Call form on both screens (Streamlit 1.63 signature): `st.table(frame, hide_index=True, border="horizontal")`. `border="horizontal"` gives row separators only; the outer 1px `border` + radius 12 comes from the Data Quality scroll container or, on Results, from CSS on the `stTable` wrapper.
- `st.dataframe` (Glide canvas) remains only on legacy research screens and keeps the CSS `filter: invert(1) hue-rotate(180deg)` rule, scoped to "app theme ≠ `config.toml` base" (base is `dark`, so it applies when the app theme is Light). No `--gdg-*` requirement.

### Tabs
- Underline style: label `muted` 14/500; hover `text`; selected `text` 600 with 2px `accent` underline; hairline 1px `border_soft` under the bar; 24 between tab bar and panel content; focus-visible outline rule on the tab.

### Alerts, toasts, status, progress, spinner
- `st.info/success/warning/error`: bg `surface_subtle`, text `text` 14/400, radius 12, padding 12 16, 1px `border_soft`, 3px left border in `accent`/`success`/`warning`/`danger`. No tinted fills. Icon (if Streamlit renders one) in the same tone color as the left border.
- `st.toast` / `st.status`: `surface` bg, `text`, 1px `border`, radius 12, popover elevation per §4.
- `st.progress`: track 4px `border_soft`, bar `accent`, radius 999; label (if any) `muted` 12.
- `st.spinner`: text `muted` 13, arc `accent`.

### Header toolbar and scrollbars
- `stHeader` transparent over `bg`; toolbar icons `muted`, hover `text` on `control_hover`, radius 8; main-menu popover per popover rule. `toolbarMode = "minimal"` stays.
- Sidebar collapse control: `control` bg, 1px `border`, radius 8, icon `text_secondary`.
- `color-scheme: light|dark` on the themed root (`body`) so native scrollbars, date pickers and form controls follow the app theme. No custom `::-webkit-scrollbar` styling.

### Popovers, selects, code
- Select field per input rule; chevron `muted`.
- Select dropdown / `[data-baseweb="popover"]` / `[role="listbox"]` / menus: `surface`, `text` 14/400, 1px `border`, radius 8, popover elevation per §4, padding 4; option height 32, radius 8; hovered/focused option `control_hover` + `text`; selected option `accent_soft` + `accent_text`.
- `st.code` / inline `code`: `surface_subtle` bg, `text_secondary` text, 1px `border_soft`, radius 8 (blocks) / 4 (inline). The Light-only hard-coded inline-code rule in `explorer.css` is replaced by this token rule for both themes.

### Charts (Plotly)
- `theme=None` in `st.plotly_chart`, styled by one function `ui_theme.style_figure(fig, theme, height=...)`, which replaces `project_chart_style.style_signal_chart`.
- `template="none"`, `paper_bgcolor=plot_bgcolor=chart_bg`, font 12 `chart_text` with the font stack.
- Margins: `l=56 r=16 b=48`; `t=48` when the legend is shown, `t=16` when `showlegend=False`.
- Legend horizontal, top-left, `x=0, y=1.02, yanchor="bottom"`, font 12 `chart_text`, no border, transparent bg.
- `hovermode="x unified"`; hoverlabel bg `surface`, font 12 `text`, border `border`.
- Y grid `chart_grid`; X grid off; axis lines 1px `chart_axis`; zero line `chart_axis`; tick labels `chart_text` tabular. No chart titles (use `st.subheader`). All annotations (limit labels, "Now") in `chart_text` 12, never in the zone color.
- Lines 2px everywhere (forecast included; no 3px). Markers only when a trace has < 200 points, 4px; forecast horizon points always 6px markers (few points). Expected-red-entry marker: `x` symbol, 12px, `zone_red`.
- Observed = `series_observed`; forecast = `series_forecast`; interval edges 1px `series_band_line`, fill `series_band`; limits dashed 1px in `zone_yellow` / `zone_red`; limit bands `zone_yellow_fill` / `zone_red_fill`; "Now" line dotted 1px `series_reference`.
- Chart container (`stPlotlyChart`): 1px `border`, radius 12, `overflow: hidden`, `surface` bg, so the chart reads as a card without an extra `st.container`.
- Heights: **280** (inline/secondary chart), **360** (primary chart of a screen). No other heights on product screens.

### Empty states
- Left-aligned block inside a `surface` card (padding 24): title 17/600 `text`, one sentence 14/400 `text_secondary`, one primary action button only if an action exists on that screen and no other primary is visible. No emoji, no illustrations, no `st.info` for empty states.

### Expanders
- `surface_subtle` bg, 1px `border`, radius 12; summary 14/500 `text`, padding 12 16; chevron `muted`; hover summary bg `control_hover`; focus-visible outline on the summary; content padding 16.

## 6. Streamlit config

`.streamlit/config.toml` has one `[theme]` table: `base = "dark"` plus Dark token values (keys verified against Streamlit 1.63.0 `config._config_options`):

```toml
[theme]
base = "dark"
primaryColor = "#0A84FF"
backgroundColor = "#0F0F11"
secondaryBackgroundColor = "#1C1C1E"
textColor = "#F2F2F4"
font = "sans-serif"
borderColor = "#38383A"
linkColor = "#64B5FF"
codeBackgroundColor = "#232326"
codeTextColor = "#D1D1D6"
baseRadius = "8px"
showWidgetBorder = true
redColor = "#FF453A"
orangeColor = "#FF9F0A"
greenColor = "#30D158"
blueColor = "#0A84FF"
```

- **Never write `[theme.light]` / `[theme.dark]` (or their `.sidebar` tables) into `.streamlit/config.toml`.** With those sections Streamlit 1.63 follows the OS `prefers-color-scheme` and shows its own Light/Dark/System switch in the header menu, which fights the in-app Appearance control (the only theme switch, subtask 03) and repaints Glide dataframe canvases from the native theme instead of our `data-theme`.
- Light surfaces are CSS only: `ui_theme.theme_css(theme)` + `explorer.css`. The native base stays dark, so legacy `st.dataframe` canvases are always dark and the light-app-only `filter: invert(1) hue-rotate(180deg)` rule flips them.
- `base = "dark"` matches `DEFAULT_THEME`. `font = "sans-serif"` (hyphenated; `"sans serif"` is not a valid 1.63 value). `codeTextColor` must be set: it otherwise defaults to Streamlit's green text color.
- `[server]`, `[browser]`, `[client]` untouched.
- Token → config key mapping (documents which CSS variables cover what config cannot switch at runtime; config carries only the Dark column): `primaryColor`=`accent`, `backgroundColor`=`bg`, `secondaryBackgroundColor`=`surface`, `textColor`=`text`, `borderColor`=`border`, `linkColor`=`accent_text`, `codeBackgroundColor`=`surface_subtle`, `codeTextColor`=`text_secondary`, sidebar background=`sidebar`, `redColor`=`danger`, `orangeColor`=`warning`, `greenColor`=`success`, `blueColor`=`accent`.
- Handled in CSS: every Light color, plus font stack, type scale, spacing, card/form/table/alert styling, sidebar nav states, segmented control, focus-visible outline, `focus_ring` halo, chart container border, `color-scheme`, legacy `st.dataframe` invert, anchor-link hiding.

## 7. Help-text copy rules (used by subtasks 04 and 05)

- Format: `What it is. How it changes the result.` Optional third sentence: `Typical: …` or `Leave the default if unsure.`
- ≤ 220 characters per `help=`. Second person, present tense, no exclamation marks, ends with a period.
- First use of any technical term gets a one-line translation in the same string (e.g. "epoch (one full pass over the training data)").
- State units (seconds, samples, Pa, g).
- Never promise accuracy; say direction of effect ("usually", "can"). Never call an uncalibrated range "likely".
- Captions only where the control has no `help=` slot (tabs, metric groups, charts); one sentence.
- All help strings live in `src/pdm/ui_copy.py` as module constants (including the Appearance hint), so they are reviewable in one place and testable.
- Sentence case for labels and buttons ("Train model", "Import and check data"); no Title Case except proper nouns (GRU, LSTM, XJTU-SY, HSE).

Reference examples (real controls in `project_training_ui.py`; all ≤ 220 chars):

- History samples: `How many past measurements the model reads before each forecast. More samples give longer context but need longer unbroken histories. Typical: 8.`
- Training epochs (GRU/LSTM only): `An epoch is one full pass over the Train units. More epochs can fit the signal better but take longer; Validation keeps the best epoch. Typical: 12.`
- Forecast horizons (seconds): `How far ahead to forecast, in seconds, separated by commas. Longer horizons warn earlier but are usually less accurate. Use multiples of your sampling interval.`

## 8. Per-screen notes

Before-screenshot (Dark, Projects) is saved at `runs/_ui_review/before/projects-dark.png` (gitignored). It shows the issues fixed by this spec: centered cyan primary nav row, low-contrast ink on the cyan primary button, 11–12 px labels, Open buttons floating far right. Other screens were read from code only (see Changes from draft).

### Projects (`project_ui._render_projects`)
- **Primary action:** "Create project" (form submit). The only primary on the screen.
- **Cards:** the create form (`st.form` card: Project name + Source format side by side in 2 columns, submit below). "Open a project" list: one `st.container(border=True)` card; each row = project name 14/600 `text` + source kind 12 `muted` on the left, "Open" secondary button on the right (`st.columns([6, 1], vertical_alignment="center")`, button `width="stretch"`); 1px `border_soft` separator between rows, none after the last.
- **Empty state:** when there are no projects, the list card is replaced by an empty-state card: title "No projects yet", sentence "Create a project above to import sensor data." No button (the form above is the action).
- **Delete:** expander "Delete {name}" below the list; secondary "Delete project" button, disabled until the checkbox is ticked; archive receipt as `st.success`.
- **Chart:** none.

### Import data (`project_ui._render_import`)
- **Primary action:** "Import and check data", left-aligned below the last card, disabled while a job is active (the job `st.info` sits directly above it).
- **Cards:** "Training source" card (source radio, uploader or path input, then Validation / Testing radios in 2 columns with their own source inputs). "Automatic split weights · 70 / 15 / 15" expander between the two cards (3 weight inputs in one row, seed below). "Signal and limits" card (source caption, signal fields for Sensor CSV in one row of 3, Red condition radio, Yellow / Red limits in 2 columns, limits caption).
- Import status alert (`_import_status`) directly under the page description.
- **Chart:** none.

### Data Quality (`project_quality_ui.render_quality`)
- **Primary action:** "Continue to Training", rendered after the admission alert, only when training is admitted; disabled while a replacement import runs.
- **Cards:** per tab (Training Data / Validation Data / Testing Data): 3 metric cards in one row (Units, Admitted rows, Gaps; no tones); captions under them; chart card = the styled Plotly container (unit selectbox directly above it, outside the card); measurement table card = `st.container(height=240, border=True)` wrapping `st.table` with sticky header; the "Admitted measurements for the selected unit" caption sits above the table card.
- Admission result as `st.success` / `st.warning` below the tabs.
- Not-ready state ("Import data to inspect Train, Validation, and Test.") becomes an empty-state card with no button.
- **Chart:** per-unit signal, height **280**, `series_observed` line, markers only when < 200 points, no legend (`t=16`), axis titles "Time (s)" and "{label} ({unit})".

### Training (`project_training_ui.render_training`)
- **Primary action:** "Train model" (form submit), disabled while any job is active. "Stop job" is secondary.
- **Model selector:** "Model" select sits above the form (it changes which fields the form shows), not in a card.
- **Cards:** the settings form card. Common fields: History samples, Forecast horizons (seconds), Random seed. **GRU / LSTM only:** Training epochs, Hidden units, Batch size (epochs exist only for the recurrent models). **Quantile boosting** is an engine already on this screen (`ENGINE_LABELS`); this spec styles that existing control and does not add an architecture. It shows Boosting iterations instead of epochs — no epoch field, no epoch wording in its help, caption or progress. Its worker progress reads "Fitting {h} s, quantile {q}, {n} iterations"; GRU/LSTM progress reads "Epoch {i}/{n} · …". Max 3 fields per row. Do not add any further model types.
- Job block directly under the description: `st.info` message, `st.progress` bar, "Stop job" secondary button.
- Latest-run block: 2 metric cards in one row (Validation mean absolute error, Held-out Test mean absolute error) with the unit caption above and known-measurements caption below.
- **Chart:** none.

### Results (`project_results_ui.render_results`)
- **Primary action:** "Play" / "Pause" (the replay toggle). "Reset" is secondary. No other primary on the screen.
- Selectors: "Saved model run" and "Test unit" side by side in 2 columns above the replay controls; model caption below them.
- Replay controls row: Play/Pause, Reset, then the "Observation time" slider (`st.columns([1, 1, 5])` stays); cursor caption below.
- **Cards:** replay chart = styled Plotly container; "Held-out Test summary" section: 2 metric cards in one row (Known future measurements, Mean absolute error); per-horizon `st.table` card below them.
- Forecast context (`_show_forecast_context`): captions and at most one alert (`st.warning` for red entry / already red, `st.info` for no crossing) directly under the chart.
- Empty state (no saved run): empty-state card, title "No saved model yet", sentence "Train a signal model to see its forecast here." No button (sidebar Training is the path).
- **Chart:** replay figure, height **360** (primary chart), legend shown (`t=48`), series per §5 Charts.

## Changes from draft

1. Status set to signed off; all token values kept unchanged (every ratio re-computed by script and matches the planner table).
2. §2.4 split into text pairs, forbidden text pairs and non-text pairs; added `text` on `surface_subtle`/`sidebar`/`control_hover`, `text_secondary`/`surface`, `accent_text`/`surface`/`bg`, and status colors on `surface_subtle`.
3. `accent` is never text: Light `accent`/`bg` is 4.31:1, so links move to `accent_text` everywhere (draft listed `accent` for links).
4. Focus: translucent `focus_ring` measures 1.49 / 1.61:1 against `surface` (fails 3:1). Focus-visible is now a 2px solid `accent` outline, 2px offset; `focus_ring` is an extra halo on inputs only.
5. New forbidden pair: `muted` on `control_hover` (Dark 3.95:1); hover states switch text to `text`.
6. `zone_unknown` restricted to fills/markers with a label (Light 2.21:1 as a line).
7. §2.5 strengthened: full Light and Dark value sets verified disjoint; `SURFACE_AND_TEXT_KEYS` now also covers `muted` and the three border tokens; legacy-literal list completed from `explorer.css` and the chart helpers; case-insensitive matching.
8. Type scale made closed: explicit size ramp and weights, plus rows for button, tab, expander, nav, table header/cell, tooltip, wordmark, empty-state title (the draft used 13/19 tooltip and 14/500 tabs outside the scale).
9. Spacing scale gains 64 (used by container padding); card padding 20 → 24 and metric padding 16 (20 was off-scale); alert/toast radius 10 → 12 (10 was off-scale).
10. Sidebar: nav rows left-aligned with explicit padding/gap; disabled nav uses full-opacity `muted` instead of 45% opacity (1.87 / 2.25:1 was unreadable); CSS hook for the existing `type="primary"` selected button; segmented-control styling defined.
11. Tables: Results per-horizon table is a plain `st.table` (no 240 px container; only Data Quality scrolls); sticky header in the Data Quality scroll container; both use `hide_index=True, border="horizontal"` (verified in the 1.63 signature) so there is no inner double border.
12. Charts: top margin depends on legend (draft `t=16` would clip the `y=1.08` legend); legend anchored at `y=1.02, yanchor="bottom"`; `l=56 b=48`; forecast line 2px (not 3); forecast points 6px; all annotations in `chart_text`; chart container border replaces an extra card.
13. Added rules for slider, spinner, uploaded file chips, sidebar collapse control, heading anchor links, form-submit button selectors, destructive button.
14. §6: concrete config.toml with verified 1.63 keys; `font` fixed to `"sans-serif"`; added `borderColor`, `linkColor`, `codeBackgroundColor`, `codeTextColor` (defaults to green otherwise), `baseRadius`, `showWidgetBorder`, status colors, sidebar backgrounds.
15. §7: added sentence-case rule and three real `help=` reference examples.
16. §8 per-screen notes added (primary action, cards, chart height for all five screens; Training states that epochs exist only for GRU/LSTM and Quantile boosting has no epoch count).
17. Screenshots: app was serving; one Dark capture of Projects saved. The embedded browser stopped repainting after navigation (captures kept returning the Projects frame while the page DOM showed Data Quality), so the remaining screens and all Light captures were reviewed from code only.
18. Review fix: `warning` text on `bg` is forbidden (Light 4.34:1). Status text sits only on `surface` or `surface_subtle`.
19. Review fix: Quantile boosting is documented as an existing Training control, not a new architecture. No further model types.
