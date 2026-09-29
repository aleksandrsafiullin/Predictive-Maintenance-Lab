# Subtask 07: One Plotly styler and a color-literal sweep of UI modules

## Goal
Route all Plotly styling through `ui_theme.style_figure`, replace every hardcoded color in UI Python modules with theme tokens, and lock it in with tests that catch 6-digit hex, 3-digit hex, `rgb(a)`, and quoted CSS color names.

## Context
Master plan R6 (Python side). Inventory (verified):
- `src/pdm/project_chart_style.py` L9–11, `plotly_white`/`plotly_dark`.
- `src/pdm/project_quality_ui.py` L98.
- `src/pdm/project_results_ui.py` L36–39, L61–81, L100.
- `src/pdm/lab_ui.py` L21 (`COLORS` rainbow), L34 (`style_figure`), L401, L405.
- `src/pdm/health_zones_ui.py` L22 (`ZONE_COLORS`), L99–120.
- `src/pdm/filter_health_zones_ui.py` L17–21, L99–123 (marker outline `color="white"`).
- `src/pdm/visualization/overlay.py` L77–80 (theme-independent cyan/amber/coral), L111–240.
- `src/pdm/visualization/recurrent_trace.py` L62 (heatmap mid `#15202c`), L65.
- `src/pdm/monitoring/ui.py` L21, L239, L247.
- `src/pdm/app.py` L1798, L1807.

## Acceptance Criteria
- [ ] `ui_theme.style_figure(fig, theme, *, height=None) -> go.Figure` implements design-spec §5 Charts (`template="none"`, chart tokens, y-grid only, font stack, unified hover, hoverlabel from `surface`/`text`/`border`).
- [ ] `project_chart_style.style_signal_chart(fig, theme)` stays (callers/tests import it) as a thin wrapper over `style_figure`. `lab_ui.style_figure(fig, height)` delegates with `current_theme()`.
- [ ] Every trace/shape/annotation color in the listed modules comes from `TOKENS[theme]`: observed → `series_observed`; forecast → `series_forecast`; bands → `series_band`/`series_band_line`; as-of/now → `series_reference`; limits/zones → `zone_*` / `zone_*_fill`; comparisons → `series_cycle`; recurrent heatmap → `heatmap_scale`; marker outlines → `surface`. `ZONE_COLORS` / `COLORS` become `zone_colors(theme)` / `series_cycle(theme)` in `ui_theme` or are removed.
- [ ] No `template="plotly_dark"` / `"plotly_white"` anywhere in `src/pdm`.
- [ ] No behavior change: same traces, data, thresholds, axes, hover content; health-zone semantics (green/yellow/red/unknown) unchanged.

## Implementation Notes
- Keep `st.plotly_chart(..., theme=None)` everywhere.
- Where a module had no `theme` argument, call `current_theme()` at render time (not import time).
- `ui_theme.py` is the only file allowed to contain color literals.
- If existing tests pin old literals (`tests/test_health_zones.py`, `tests/test_filter_health_zones_ui.py`, `tests/test_work_overlay_design.py`, `tests/test_operational_explorer.py`, `tests/test_neural_explorer.py`), update them to compare against `TOKENS[...]`; do not weaken semantic assertions.

## Dependencies
Subtask 03 (`current_theme`, `TOKENS`), subtask 06 (final token values).

## Verification
Add to `tests/test_worker_and_app.py`:
- `test_ui_modules_have_no_hardcoded_color_literals`: for each of `app.py, project_chart_style.py, project_quality_ui.py, project_results_ui.py, project_training_ui.py, project_ui.py, lab_ui.py, health_zones_ui.py, filter_health_zones_ui.py, visualization/overlay.py, visualization/recurrent_trace.py, visualization/presentation.py, monitoring/ui.py`, assert no match of
  - `r"#[0-9a-fA-F]{6}\b"` and `r"#[0-9a-fA-F]{3}\b"` inside string literals (use `r"[\"']#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?[\"']"` plus a check inside f-strings/CSS strings for `#[0-9a-fA-F]{3,8}\b`),
  - `r"\brgba?\("`,
  - quoted CSS color names used as values: `r"(?:color|bgcolor|fillcolor|line_color|bordercolor)\s*=\s*[\"'](?:white|black|red|green|blue|yellow|orange|gray|grey|cyan|magenta)[\"']"` and dict form `r"[\"'](?:color|bgcolor)[\"']\s*:\s*[\"'](?:white|black|…)[\"']"`.
  `ui_theme.py` is excluded.
- `test_no_builtin_plotly_bw_templates`: `rg`-style scan of `src/pdm/**/*.py` finds neither `plotly_dark` nor `plotly_white`.
- `test_signal_chart_light_has_no_dark_colors`: one-scatter `go.Figure`, `style_signal_chart(fig, "light")`, `payload = json.dumps(fig.to_plotly_json()).lower()`; assert no `DARK_EXCLUSIVE` value; mirror for dark with `LIGHT_EXCLUSIVE`.
- `test_replay_figure_uses_theme_tokens`: `project_results_ui.replay_figure(result, schema, "light")` with a minimal result dict shaped like `tests/project_results_harness.py::_forecast`; payload contains `TOKENS["light"]["series_forecast"].lower()` and no `DARK_EXCLUSIVE` value.
- Existing suites listed above green.
- `.venv/bin/python -m pytest tests -q`, `.venv/bin/python -m ruff check src tests`.
- Manual: Light Data Quality chart, Results replay chart, Health zones — no dark plot backgrounds or neon series; Dark — no white plot backgrounds.
