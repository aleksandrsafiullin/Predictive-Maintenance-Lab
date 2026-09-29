# Subtask 03: Single theme source of truth that survives reruns, navigation and reloads

## Goal
Introduce `src/pdm/ui_theme.py` as the only owner of the current theme, render exactly one Appearance control, persist the choice across reruns, screen changes, widget actions and browser reloads, and remove every direct `st.session_state["ui_theme"]` read.

## Context
Root causes (master plan R1–R4), confirmed:
- **R1 (confirmed in Streamlit):** `src/pdm/project_ui.py` ~L319–359 — sidebar nav buttons and the project picker call `st.rerun()` **before** `st.selectbox("Appearance", ..., key="ui_theme")` is rendered. The `RerunException` path runs `_remove_stale_widgets`, which discards state of widgets not instantiated in that run, so the next run falls back to `"dark"`.
- **R2:** `src/pdm/app.py` L380 — second selectbox with the same key in `legacy_main()`.
- **R3:** direct reads with dark default: `lab_ui.py` L33, `health_zones_ui.py` L98, `filter_health_zones_ui.py` L98, `visualization/overlay.py` L76, `visualization/recurrent_trace.py` L46.
- **R4:** session state dies on browser reload.
- `render_quality(snapshot, theme)` / `render_results(..., theme)` / `_play_fragment(..., theme)` keep their `theme` parameter; callers pass `current_theme()`.

### Persistence decision: manual store, not `bind="query-params"`
Streamlit 1.63 offers `bind="query-params"` / `persist_state="session"` on radio and segmented control. We use a **manual non-widget store** instead:
1. The theme must survive a run that aborts via `st.rerun()` **before** the widget is drawn. Widget-owned state (bound or not) is exactly what `_remove_stale_widgets` cleans on such runs; relying on bind re-reading the URL on the next instantiation depends on undocumented ordering between stale-widget cleanup and query-param sync.
2. Six render functions and a timer fragment (`project_results_ui._play_fragment`) need the theme without instantiating the widget. They need a plain value, not a widget.
3. A plain `st.session_state["_pdm_theme"]` key is never garbage-collected, and the `on_change` callback writes both `_pdm_theme` and `st.query_params["theme"]` **before** the script body re-executes, i.e. before any `st.rerun()` in that run.
So `_pdm_theme` is the source of truth; the widget is input only; the URL is the reload backup.

## Acceptance Criteria
- [ ] New module `src/pdm/ui_theme.py` with:
  - `THEMES = ("light", "dark")`, `DEFAULT_THEME = "dark"` (fallback; existing test asserts default dark).
  - `TOKENS: dict[str, dict[str, str | list]]` with all keys from `design-spec.md` §2.1–2.3 (draft values OK; subtask 06 finalizes), plus `SURFACE_AND_TEXT_KEYS`, `LIGHT_EXCLUSIVE`, `DARK_EXCLUSIVE` per §2.5.
  - `current_theme() -> str`, resolution order:
    1. `st.session_state["_pdm_theme"]` if valid;
    2. `st.query_params.get("theme")` if valid;
    3. **first run only** (no `_pdm_theme`, no `?theme=`): `st.context.theme.type` if it is in `THEMES` — used only as a one-time seed, never re-read afterwards (it is inferred from the background and can be wrong on first load);
    4. `DEFAULT_THEME`.
    Writes the resolved value to `_pdm_theme`. Never reads `"ui_theme"`.
  - `set_theme(theme: str | None) -> None`: if `theme` is `None` or not in `THEMES`, do nothing to the store and re-seed the widget key from `_pdm_theme`; otherwise set `_pdm_theme` and `st.query_params["theme"] = theme`.
  - `render_theme_control(container=st.sidebar) -> str`:
    - Before rendering: `if st.session_state.get("_pdm_theme_widget") != current_theme(): st.session_state["_pdm_theme_widget"] = current_theme()`.
    - Renders `container.segmented_control("Appearance", options=["light", "dark"], format_func=str.title, key="_pdm_theme_widget", required=True, on_change=_on_theme_change, help=ui_copy.APPEARANCE_HELP)`.
    - **No `default=` / `index=`** argument (value is seeded only via session state; passing both triggers Streamlit's "value set via Session State API" warning).
    - `_on_theme_change()` calls `set_theme(st.session_state.get("_pdm_theme_widget"))`.
    - Returns `current_theme()`.
- [ ] `src/pdm/ui_copy.py` created (or appended) with `APPEARANCE_HELP = "Switch between a light and a dark look. Your choice is kept while you move between screens and after a page reload."`.
- [ ] `project_ui.main()`: the Appearance control is rendered **first** inside `with st.sidebar:` under an `APPEARANCE` caption (before the project selectbox and nav buttons; subtask 09 may move it visually with CSS order, never in code order). `apply_explorer_style(theme)` is called immediately after, before any main-area rendering.
- [ ] `app.legacy_main()` uses the same `render_theme_control()`; `rg -n 'key="ui_theme"' src` returns nothing.
- [ ] All five direct readers use `from pdm.ui_theme import current_theme`. `rg -n 'session_state.get\("ui_theme"' src` returns nothing.
- [ ] `tests/project_results_harness.py`: replace `st.session_state.setdefault("ui_theme", "light")` with `st.session_state["_pdm_theme"] = "light"`.
- [ ] `_reset_project_session()` (project_ui L29) does not touch `_pdm_theme*` keys (it currently doesn't; the navigation test asserts it).
- [ ] Spike (≤15 min, notes appended at the bottom of this file, no code): can the `help=` tooltip bubble, select popover/listbox and main-menu popover be restyled with CSS injected via `st.markdown` in 1.63 (selectors `[data-testid="stTooltipContent"]`, `[data-baseweb="popover"]`, `[role="listbox"]`, `[role="tooltip"]`)? Record which selectors work. Subtask 06 consumes this list. (`[theme.light]`/`[theme.dark]` support is already known; `st.context.theme` is not a source of truth — no spike needed on those.)

## Implementation Notes
- If AppTest in 1.63 cannot drive `segmented_control` (no accessor or no `set_value`), fall back to `st.radio("Appearance", ["light", "dark"], format_func=str.title, horizontal=True, key="_pdm_theme_widget", on_change=..., help=...)` — still no `index=`, seeded via session state. Record the choice here.
- Update `tests/test_worker_and_app.py::test_import_navigation_and_theme_in_empty_workspace` (L251–256) to find the control by label across widget types; keep asserting default `"dark"` then `"light"`.
- `st.query_params["theme"] = ...` does not rerun; do not clobber other params (`view=brain` in legacy).
- Presentation only: no changes to training, worker, or data modules.

## Dependencies
Subtask 02 (token names). Can start with draft token values.

## Verification
Add to `tests/test_worker_and_app.py` (product app: `AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)`; `root = tmp_path / "projects"`; `monkeypatch.setenv("PDM_PROJECTS_ROOT", str(root))`):

- `test_theme_light_survives_project_navigation_and_actions` — **must fail on current code and pass after the fix** (this is the R1 test):
  1. `_, project, _ = make_contract_snapshot(root)` (from `tests.project_contract`) **before** the first `at.run()`; `contract_pid = project["project_id"]`.
  2. `at.run()`; set Appearance to `"light"`; `at.run()`.
  3. Fill `Project name`, click `Create project` (main-area `st.rerun()`); `at.run()`.
  4. Click sidebar button key `project_nav:Projects` (sidebar `st.rerun()` before the old control position); `at.run()`.
  5. Click button key `f"open:{contract_pid}"`; `at.run()`.
  6. Click `project_nav:Data Quality`, `at.run()`; click `project_nav:Training`, `at.run()`.
  After **every** `at.run()` assert: `at.session_state["_pdm_theme"] == "light"`, the Appearance control value is `"light"`, and `any('data-theme="light"' in str(m.value) for m in at.markdown)`.
- `test_theme_light_survives_legacy_workflow_steps` — **regression coverage** for `legacy_main` (not proof of R1): legacy harness `tests/legacy_app_harness.py` with `tiny_bearing_tables` setup copied from `test_prepared_project_can_advance_from_quality_to_training`; set light; click `Continue to Training`; assert `_pdm_theme == "light"` and light marker.
- `test_theme_restored_from_query_param_on_fresh_session`: fresh product `AppTest`, `at.query_params["theme"] = "light"`, `at.run()`; assert Appearance `"light"` and light marker.
- `test_theme_single_control_in_sidebar`: on product app and legacy harness, exactly one widget labelled `"Appearance"` across all widget collections.
- `test_theme_set_none_is_ignored`: unit test with a stub `st.session_state` (or AppTest run of a tiny script) — `set_theme(None)` leaves `_pdm_theme` unchanged and re-seeds `_pdm_theme_widget`.
- `rg -n '"ui_theme"' src tests` returns nothing.
- `.venv/bin/python -m pytest tests -q` and `.venv/bin/python -m ruff check src tests` green.

## Spike notes (fill in during implementation)
_Tooltip / popover / listbox / menu CSS selectors that work in 1.63:_

Checked 2026-09-29 against Streamlit 1.63.0: bundle in `.venv/.../streamlit/static/static/js/` plus a live run (`PYTHONPATH=src streamlit run src/pdm/app.py --server.port 8502`, 127.0.0.1). A probe `<style>` was appended inside `[data-testid="stMain"]`, the same place `st.markdown(..., unsafe_allow_html=True)` injects CSS, and computed styles were read back.

- **BaseWeb is gone in 1.63.** The bundle has no `data-baseweb` attribute and the live DOM has 0 `[data-baseweb]` nodes. Overlays use react-aria/floating-ui. **`[data-baseweb="popover"]` matches nothing. Do not use it.**
- **Overlays are portaled to `<body>`, outside `.stApp`.** Tooltip chain: `div[stTooltipContent] > div > div > body`. Listbox chain: `div > div[stSelectboxVirtualDropdown] > div > body`. Global CSS injected via `st.markdown` still reaches them, but selectors scoped under `.stApp` / `[data-testid="stAppViewContainer"]` do **not**. Use unscoped selectors or `body:has(.brain-lab-shell[data-theme="light"]) …`.
- `[data-testid="stTooltipContent"]`: **works (live)**. It holds the `help=` bubble and has class `stTooltipContent`. Error tooltips use `stTooltipErrorContent`.
- `[role="tooltip"]`: **works (live)**. It is the tooltip's floating wrapper. DataFrame cell tooltips also use `role=tooltip` (with inner `stDataFrameTooltipContent`), so a bare `[role="tooltip"]` rule also restyles grid tooltips.
- `[data-testid="stTooltipHoverTarget"]` / `stTooltipIcon`: the trigger icon. Present in the bundle, and 1 found live.
- `[data-testid="stSelectboxVirtualDropdown"]`: **works (live)**. It is the selectbox dropdown panel.
- `[role="listbox"]`: **works (live)**. It sits inside the selectbox dropdown. Options are `[role="option"]` with no `data-testid`, and the selected option carries `aria-selected="true"`. DataFrame's internal react-select editor also uses `role=listbox`.
- `[data-testid="stMainMenuPopover"]` (role `menu`, list `stMainMenuList`): exists in the bundle (`index.*.js`). **Not verified live** because the main menu did not render with `client.toolbarMode = "minimal"` in this app (no `stMainMenu` / `stToolbar` in the DOM). By construction it is also a body-level portal, so the same rule applies.
- The segmented control renders as `[data-testid="stButtonGroup"]` with segments exposed as `role="radio"` (`aria-checked`).

_Appearance widget type used (segmented_control or radio fallback):_ **`st.segmented_control`**. AppTest 1.63 drives it via `at.button_group` / `at.segmented_control` (`ButtonGroup.set_value`), so no radio fallback was needed. It has no `default=` / `index=` and is seeded only through `st.session_state["_pdm_theme_widget"]`. Checked live: `?theme=light` on a fresh load renders Light, and clicking Dark rewrites the URL to `?theme=dark` and flips `data-theme`.

_AppTest caveat:_ an ad-hoc run of the **pre-fix** `project_ui` under AppTest kept the light marker after Create project and a sidebar `project_nav:Projects` click. AppTest re-submits every widget state from its last tree on each `run()`, which masks R1. On old code, `test_theme_light_survives_project_navigation_and_actions` fails because the `_pdm_theme` store is missing, not because the marker flips. It is a store/contract test, not a faithful browser reproduction of R1.
