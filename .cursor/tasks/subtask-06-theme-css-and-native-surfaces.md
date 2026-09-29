# Subtask 06: Theme CSS from tokens, native surfaces, config, product tables

## Goal
Generate each theme's CSS variable block from `pdm.ui_theme.TOKENS`, make `explorer.css` color-literal-free, restyle every native Streamlit surface from tokens in both themes, align `.streamlit/config.toml`, and move the two product tables to `st.table`, so no Light screen shows Dark CSS colors and vice versa.

## Context
Master plan R5/R6 (CSS side). Inventory:
- `src/pdm/visualization/explorer.css`: L3–22 two hardcoded palettes (cyan `#55c9dc`/`#087f91`); L99–111 light `--gdg-*` literals; L112–115 canvas `invert` hack; L116–119 light `code` literals; gradient `.lab-metric` L145.
- `src/pdm/visualization/presentation.py::apply_explorer_style` reads `explorer.css` verbatim.
- `.streamlit/config.toml` `[theme] base="dark"` + dark literals.
- Product `st.dataframe`: `src/pdm/project_quality_ui.py` L110 (Time, signal, Record position; scroll, height 220) and `src/pdm/project_results_ui.py` L277 (per-horizon rows, a handful).
- Spike notes at the bottom of `subtask-03-theme-source-of-truth.md` list which tooltip/popover selectors work.

### Table decision (fixed, not a developer choice)
Both product tables switch to `st.table(frame)` inside `st.container(height=240)` (Data Quality) / no height (Results, few rows). They show the same columns, need no sorting/selection/editing, and become DOM tables the theme CSS can style. The legacy research `st.dataframe` calls stay; the existing canvas `invert` rule is kept but scoped to "app theme ≠ `config.toml` base" (`body:has(.brain-lab-shell[data-theme="light"])` while base is dark). No `--gdg-*` variables.

## Acceptance Criteria
- [ ] `TOKENS` in `src/pdm/ui_theme.py` hold the signed-off values of `design-spec.md` §2.
- [ ] `ui_theme.theme_css(theme) -> str` returns one rule `body:has(.brain-lab-shell[data-theme="<theme>"]) { --pdm-*: …; --lab-*: var(--pdm-*); color-scheme: <theme>; }` built only from `TOKENS[theme]`. **Each theme's block contains only that theme's values**; `theme_css("light")` emits no Dark token value, `theme_css("dark")` no Light token value.
- [ ] `apply_explorer_style(theme)` injects `theme_css(theme)` followed by static `explorer.css`. `explorer.css` has no color literals: no `#hex` (3–8 digits), no `rgb(`/`rgba(`/`hsl(`, no CSS color keywords (`white`, `black`, `red`, …); only `var(--pdm-*)` / `var(--lab-*)`, `transparent`, `inherit`, `currentColor`, and `color-mix()` over vars. Elevation shadow uses `color-mix(in srgb, var(--pdm-text) 4%, transparent)`.
- [ ] Legacy aliases kept so existing selectors work: `--lab-bg`→bg, `--lab-sidebar`→sidebar, `--lab-panel`→surface, `--lab-panel-raised`/`--lab-surface`→surface_subtle, `--lab-input`→input, `--lab-control(-hover)`→control(_hover), `--lab-line`→border, `--lab-line-soft`→border_soft, `--lab-line-strong`/`--lab-track`→border_strong, `--lab-selected`→accent_soft, `--lab-text`→text, `--lab-text-soft`/`--lab-muted-strong`→text_secondary, `--lab-muted`→muted, `--lab-accent`→accent, `--lab-accent-soft`→accent_text, `--lab-accent-ink`→accent_ink, `--lab-amber(-soft)`→warning, `--lab-success`→success, `--lab-danger`→danger.
- [ ] Native surfaces styled from tokens in both themes (design-spec §5):
  - `help=` tooltip bubble, select dropdown/listbox/options, `[data-baseweb="popover"]` menus, main-menu popover — using the selectors confirmed in the 03 spike;
  - `st.code` and inline `code`;
  - file-uploader dropzone and its "Browse files" button;
  - number-input stepper buttons;
  - radio and checkbox indicators (unchecked ring, checked fill, dot/check);
  - `st.progress` track and bar;
  - `st.toast` and `st.status` containers;
  - header toolbar (`stHeader`, `stToolbar`) icons and background;
  - scrollbars and native controls via `color-scheme` on the themed root.
- [ ] Canvas `invert` rule scoped as in the table decision; Glide `--gdg-*` literal block (L99–111) deleted.
- [ ] Product tables: `project_quality_ui.py` and `project_results_ui.py` use `st.table` as decided; same columns and row order; `st.table` styled per §5 Tables.
- [ ] `.streamlit/config.toml`: `[theme]` base dark with Dark token values; new `[theme.light]` and `[theme.dark]` sections from tokens (keys `primaryColor`, `backgroundColor`, `secondaryBackgroundColor`, `textColor`); `[server]`/`[browser]`/`[client]` untouched.
- [ ] No Python behavior change beyond the two table widgets.

## Implementation Notes
- Token key → CSS var: `bg`→`--pdm-bg`, `surface_subtle`→`--pdm-surface-subtle`, etc. (underscore → hyphen). Chart and list tokens (`series_cycle`, `heatmap_scale`) are not emitted as CSS.
- `st.table` on a frame with a few thousand rows is acceptable in a height-240 scroll container; keep `hide_index` semantics by resetting the index / using `frame.style.hide(axis="index")` if the plain index is visible.
- AppTest: tests that look for `at.dataframe` on these two screens (check `rg -n "at.dataframe" tests`) must switch to `at.table`.
- Do not touch `src/pdm/visualization/component/frontend/**`.

## Dependencies
Subtask 02 (values), subtask 03 (`ui_theme` module, spike selectors).

## Verification
Add to `tests/test_worker_and_app.py`:
- `test_light_theme_css_contains_no_dark_colors`: `css = theme_css("light").lower()`; assert no value from `DARK_EXCLUSIVE` and no value of any `TOKENS["dark"]` string token that differs from Light appears; assert none of the design-spec §2.5 legacy literals appear in `css + explorer_css_text`.
- `test_dark_theme_css_contains_no_light_colors`: mirror with `LIGHT_EXCLUSIVE` (no exceptions).
- `test_explorer_css_has_no_color_literals`: on `Path("src/pdm/visualization/explorer.css").read_text()` assert no match for `r"#[0-9a-fA-F]{3,8}\b"`, `r"\b(?:rgba?|hsla?)\("`, or `r"(?<![-\w])(?:white|black|red|green|blue|yellow|orange|gray|grey|cyan|magenta)(?![-\w])"`.
- `test_theme_css_covers_native_surfaces`: `theme_css + explorer.css` contains selectors for `stTooltipContent` (or the spike-confirmed tooltip selector), `data-baseweb="popover"`, `stFileUploaderDropzone`, `stNumberInput` button, `stCheckbox`, `stRadio`, `stProgress`, `stToast`, `stHeader`, and `color-scheme`.
- `test_product_tables_use_st_table`: Data Quality AppTest (setup from `tests/test_project_ui.py::test_quality_three_sets_and_training_gate`) has `len(at.table) >= 1` and `len(at.dataframe) == 0`; static check that `st.dataframe` does not appear in `project_quality_ui.py` or `project_results_ui.py`.
- `test_config_toml_has_light_and_dark_sections`: `tomllib.load` on `.streamlit/config.toml` has `theme.light.backgroundColor == TOKENS["light"]["bg"]` and `theme.dark.backgroundColor == TOKENS["dark"]["bg"]`, and `server.address == "127.0.0.1"`.
- `.venv/bin/python -m pytest tests -q`, `.venv/bin/python -m ruff check src tests`.
- Manual at `http://127.0.0.1:8501`, both themes: tooltip on a `?` icon, select dropdown, file uploader, number steppers, radio, progress (start a training job), toast, header menu, page scrollbar — none from the other palette.

## Review fix
`[theme.light]`/`[theme.dark]` were removed from `.streamlit/config.toml` because Streamlit 1.63 turns them into a second, OS-following theme system with its own header switch that fights the Appearance control and repaints Glide canvases, so config keeps a single `[theme]` with `base = "dark"` and Dark tokens while Light comes only from `theme_css` + `explorer.css` plus the light-app-only dataframe canvas invert.
