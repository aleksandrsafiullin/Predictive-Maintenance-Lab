# Master Plan: Dual theme that sticks, plain-language controls, Apple-grade Streamlit UI

**Repo:** Predictive Maintenance Lab · package `src/pdm/` · **Stack:** Python ≥3.11 · Streamlit 1.63 · Plotly 7 · pytest · ruff
**Date:** 2026-09-29 · **Revised:** 2026-09-29 after plan-review REVISE (all Criticals, Warnings and Suggestions applied; old 04 and 07 split; renumbered 01–10).
**Replaces:** the MaleCNS anatomy epic (old `subtask-01…11`, already implemented; deleted from `.cursor/tasks/`, recoverable from git). `.cursor/tasks/create-ml-workflow-20260929/` is a separate dated archive, not part of this plan.

**Presentation only.** No change to training math, losses, splits, scalers, windowing, horizon semantics, defaults, worker protocol, or saved artifacts. GRU/LSTM only. No React/FastAPI/Docker/MLflow. UI English. Bind `127.0.0.1:8501`. Real data only.

## Overview

Three complaints from Alex (product owner):

1. **Theme broken.** Light shows dark artifacts; switching screens or pressing buttons snaps Light back to Dark.
2. **Data Quality and Training are unreadable** for a normal user. Every control the user sets needs a plain-English hint: what it is and how it changes the result.
3. **Visual quality far below Apple-grade.** Create a dedicated `ui-designer` agent; it owns a concrete design spec the developer implements.

### Root causes (verified in code 2026-09-29)

| # | Finding | Where |
|---|---------|-------|
| R1 | **Confirmed.** Product entry is `app.main()` → `project_ui.main()`. The theme `st.selectbox(..., key="ui_theme")` renders **after** the sidebar nav buttons and project picker, which call `st.rerun()` first. Streamlit's `RerunException` path runs `_remove_stale_widgets`, discarding the un-rendered widget's state; the next run defaults to `"dark"`. | `src/pdm/project_ui.py` ~L319–359 |
| R2 | Second theme selectbox with the same key in `legacy_main()`. | `src/pdm/app.py` L380 |
| R3 | Five modules read `st.session_state.get("ui_theme", "dark")` directly. | `lab_ui.py` L33, `health_zones_ui.py` L98, `filter_health_zones_ui.py` L98, `visualization/overlay.py` L76, `visualization/recurrent_trace.py` L46 |
| R4 | Session state dies on browser reload → dark again. | — |
| R5 | `.streamlit/config.toml` pins `[theme] base="dark"`; native surfaces (dataframe canvas, tooltips, popovers, menus, uploader, steppers, scrollbars) follow it. `explorer.css` papers over the dataframe with `filter:invert(1) hue-rotate(180deg)`. | `.streamlit/config.toml`, `visualization/explorer.css` L112–115 |
| R6 | Hardcoded colors: 23 in `explorer.css`; Python literals in `project_chart_style.py`, `project_quality_ui.py`, `project_results_ui.py`, `health_zones_ui.py`, `filter_health_zones_ui.py`, `lab_ui.py`, `visualization/overlay.py`, `visualization/recurrent_trace.py`, `monitoring/ui.py`, `app.py` L1798/1807; `color="white"` in `filter_health_zones_ui.py`. | listed files |
| R7 | Training form: only `Forecast horizons` has `help=`. Data Quality / Import controls mostly unexplained. | `project_training_ui.py`, `project_quality_ui.py`, `project_ui.py` |

### Key decisions (fixed in the plan, not left to developers)

- **Theme persistence:** manual non-widget `st.session_state["_pdm_theme"]` is the source of truth, written by the widget's `on_change` callback (before any `st.rerun()` in the next run) together with `st.query_params["theme"]`. Not `bind="query-params"`: widget-owned state is exactly what `_remove_stale_widgets` drops, and six render functions plus a timer fragment need the theme without the widget. `st.context.theme.type` is only a first-run seed when there is no `_pdm_theme` and no `?theme=`; `DEFAULT_THEME="dark"` is the fallback. Widget: `st.segmented_control(..., required=True)`, no `default=`/`index=`, seeded via session state; `set_theme(None)` is ignored.
- **Tables:** the two product tables (Data Quality measurements, Results per-horizon) become `st.table` inside a scroll container; legacy `st.dataframe` keeps the canvas `invert` rule scoped to "app theme ≠ config base". No Glide `--gdg-*` work.
- **Config:** `[theme]` base dark + `[theme.light]` / `[theme.dark]` from tokens.
- **WCAG:** token tables in `design-spec.md` already fixed and ratios recorded (dark `accent_ink #000000`, new `accent_text` for selected nav, light `success #1E7B34`, light `muted #636366`, `zone_yellow` lines only).
- **Stop job copy:** "Asks the job to stop after its current safe step. No model is saved from a stopped run; earlier saved runs stay available." (stop deletes the run's `manifest.json`, `signal_training.py` L387).

## Phases

| Phase | # | File | Theme |
|-------|---|------|-------|
| A. Design authority | 01 | `subtask-01-ui-designer-agent.md` | Create `.cursor/agents/ui-designer.md` |
| A. Design authority | 02 | `subtask-02-design-spec.md` | ui-designer signs off `.cursor/tasks/design-spec.md` |
| B. Theme engine | 03 | `subtask-03-theme-source-of-truth.md` | `pdm.ui_theme`, persistent theme, one control, `ui_copy.py`, tooltip/popover CSS spike |
| C. Plain language | 04 | `subtask-04-data-quality-help.md` | Help on Projects, Import, Data Quality |
| C. Plain language | 05 | `subtask-05-training-help.md` | Help on every Training parameter (incl. Stop job) |
| B. Theme engine | 06 | `subtask-06-theme-css-and-native-surfaces.md` | `theme_css`, literal-free `explorer.css`, native surfaces, `config.toml`, `st.table` |
| B. Theme engine | 07 | `subtask-07-chart-tokens-and-literal-sweep.md` | `style_figure`, module color sweep, chart tests |
| D. Visual polish | 08 | `subtask-08-css-polish.md` | CSS-only: type, sidebar, buttons, cards, alerts, tabs |
| D. Visual polish | 09 | `subtask-09-layout-headers-forms-empty-states.md` | `page_header`, form grouping, empty states, one primary |
| E. Gate | 10 | `subtask-10-apptest-gate.md` | AppTest consolidation, pytest/ruff, browser walk |

## Dependencies

```text
01 ─► 02 ─► 03 ─┬─► 04 ─────────────┐
                ├─► 05 ─────────────┤
                └─► 06 ─► 07 ─► 08 ─┴─► 09 ─► 10
```

| # | Depends on |
|---|-----------|
| 01 | — |
| 02 | 01 |
| 03 | 02 |
| 04 | 02, 03 (`ui_copy.py`) |
| 05 | 02, 03 (`ui_copy.py`) |
| 06 | 02, 03 (module + spike selectors) |
| 07 | 03, 06 (final tokens) |
| 08 | 04, 05, 06, 07 |
| 09 | 04, 05, 08 |
| 10 | 03–09 |

04 and 05 may run in parallel with each other (both append to `ui_copy.py`; 04 also edits `project_quality_ui.py` and Import/Projects in `project_ui.py`). 06 also edits `project_quality_ui.py` (`st.table`), so run 04 before 06. 03 touches only the sidebar block of `project_ui.main()`.

## Execution order

1. 01 ui-designer agent
2. 02 design spec sign-off
3. 03 theme source of truth + persistence
4. 04 Data Quality help ‖ 05 Training help (parallel OK)
5. 06 theme CSS (after 04; both edit `project_quality_ui.py`)
6. 07 chart tokens + literal sweep (after 06)
7. 08 CSS polish
8. 09 layout
9. 10 gate

## Estimated effort

| # | Effort |
|---|--------|
| 01 | 0.5 h |
| 02 | 1 h |
| 03 | 1.5 h |
| 04 | 1.5 h |
| 05 | 1 h |
| 06 | 1.5 h |
| 07 | 1.5 h |
| 08 | 1.5 h |
| 09 | 1.5 h |
| 10 | 1 h |
| **Total** | **~12.5 h** |

## Planned file changes

- New: `.cursor/agents/ui-designer.md`, `src/pdm/ui_theme.py`, `src/pdm/ui_copy.py` (`.cursor/tasks/design-spec.md` already drafted)
- Changed: `src/pdm/project_ui.py`, `src/pdm/app.py` (`legacy_main` sidebar, L1798/1807 colors, `screen_data`/`_render_import` help), `src/pdm/visualization/presentation.py`, `src/pdm/visualization/explorer.css`, `src/pdm/project_chart_style.py`, `src/pdm/project_quality_ui.py`, `src/pdm/project_training_ui.py`, `src/pdm/project_results_ui.py`, `src/pdm/future_red_ui.py`, `src/pdm/lab_ui.py`, `src/pdm/health_zones_ui.py`, `src/pdm/filter_health_zones_ui.py`, `src/pdm/visualization/overlay.py`, `src/pdm/visualization/recurrent_trace.py`, `src/pdm/monitoring/ui.py`, `.streamlit/config.toml`, `tests/project_results_harness.py`
- Tests: all new nodes in `tests/test_worker_and_app.py`; color-pinning assertions in existing test files updated to compare against `TOKENS`
- Must not change: `src/pdm/{train,losses,splits,windows,preprocessing,signal_training,signal_inference,worker,training_engine}.py`, `src/pdm/data/**` (incl. `data/project_import.py`, `data/project_prepare.py`)
- Out of scope: `src/pdm/visualization/component/frontend/**` (WebGL internals; the 3D viewport may stay dark like a video canvas, its chrome follows the theme)

## Verification

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check src tests
```

Named nodes are listed per subtask and consolidated in `subtask-10-apptest-gate.md`. Manual: `.venv/bin/python -m pdm app` → `http://127.0.0.1:8501`, full walk in Light then Dark including native surfaces (tooltip, popover, select, code, file uploader, number steppers, radio/checkbox, progress, toast/status, header toolbar, scrollbars) and a browser reload; screenshots under `runs/_ui_review/` (gitignored).

## Remaining risks

1. **Tooltip/popover CSS reach.** Whether 1.63 tooltips and popovers accept injected CSS is the only open spike (subtask 03). If a surface cannot be styled, it follows the config base (dark); record it and surface to Alex rather than hacking JS.
2. **AppTest accessors.** `segmented_control` drive-ability and `.help` on every widget type: fallbacks are radio and `widget.proto.help`.
3. **`st.table` size.** Very long unit histories render every row in DOM inside a 240 px scroll box; acceptable for XJTU/HSE unit sizes, may be slow for unusually long generic CSV units (speculative).
4. **"Compare" screen** exists only in `legacy_main()`; covered for theme persistence, not prioritized in the visual pass.
