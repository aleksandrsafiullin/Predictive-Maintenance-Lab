# Subtask 10: AppTest consolidation and final gate

## Goal
Make sure every promised behavior — theme persistence, per-theme color exclusivity, native-surface coverage, help text on Import/Data Quality/Training, single primary action — is covered by named nodes in `tests/test_worker_and_app.py`, the full suite and ruff are green, and a real browser walk in both themes confirms it.

## Context
Subtasks 03–09 each add tests. This subtask deduplicates setup, fills gaps and runs the gate. Constraint from Alex: UI verification lives in `tests/test_worker_and_app.py` (Streamlit AppTest).

## Acceptance Criteria
- [ ] Shared helpers at the top of the new block in `tests/test_worker_and_app.py`: `_product_app()`, `_theme_control(at)` (finds `Appearance` across segmented control / radio / selectbox), `_help_of(widget)`, `_light_marker(at)`.
- [ ] All nodes exist and pass:
  - 03: `test_theme_light_survives_project_navigation_and_actions`, `test_theme_light_survives_legacy_workflow_steps`, `test_theme_restored_from_query_param_on_fresh_session`, `test_theme_single_control_in_sidebar`, `test_theme_set_none_is_ignored`
  - 04: `test_import_and_quality_controls_expose_help`, `test_ui_copy_help_strings_follow_rules`
  - 05: `test_training_controls_expose_help`, `test_training_stop_help_copy`, `test_training_mae_help_mentions_equal_units`, `test_future_red_training_controls_expose_help`
  - 06: `test_light_theme_css_contains_no_dark_colors`, `test_dark_theme_css_contains_no_light_colors`, `test_explorer_css_has_no_color_literals`, `test_theme_css_covers_native_surfaces`, `test_product_tables_use_st_table`, `test_config_toml_locks_streamlit_base_dark`
  - 07: `test_ui_modules_have_no_hardcoded_color_literals`, `test_no_builtin_plotly_bw_templates`, `test_signal_chart_light_has_no_dark_colors`, `test_replay_figure_uses_theme_tokens`
  - 08: `test_explorer_css_min_font_size_11px`, `test_explorer_css_has_no_glow_or_gradient`
  - 09: `test_product_screens_single_primary_button`, `test_product_titles_unchanged`, `test_training_form_boosting_has_no_epochs`
- [ ] Gap node: `test_theme_light_survives_results_replay_fragment` using `tests/project_results_harness.py` (sets `_pdm_theme="light"`), asserting `data-theme="light"` is re-injected after a second `at.run()`.
- [ ] `test_import_navigation_and_theme_in_empty_workspace` uses `_theme_control(at)`; still asserts default `"dark"`, then `"light"` after `set_value`.
- [ ] `rg -n '"ui_theme"|plotly_dark|plotly_white' src tests` → no hits.
- [ ] No diffs in model/data/worker code (git guard below).
- [ ] Full gate green.

## Implementation Notes
- AppTest cannot evaluate CSS rendering; visual claims rely on the browser walk. `--smoke` training is not a quality claim and not needed.
- Keep 15 s timeouts; monkeypatch `pdm.project_ui.list_project_runs` / `available_signal_engines` like existing tests if slow.

## Dependencies
Subtasks 03–09.

## Verification
```bash
.venv/bin/python -m pytest tests/test_worker_and_app.py -q -k "theme or help or css or color or primary or titles or table or config or template or epochs"
.venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check src tests
git diff --stat -- src/pdm/train.py src/pdm/losses.py src/pdm/splits.py src/pdm/windows.py src/pdm/preprocessing.py src/pdm/signal_training.py src/pdm/signal_inference.py src/pdm/worker.py src/pdm/training_engine.py src/pdm/data/project_import.py src/pdm/data/project_prepare.py src/pdm/data
```
(`git diff --stat` for these paths must be empty.)

Manual browser walk (`.venv/bin/python -m pdm app`, `http://127.0.0.1:8501`):
1. Set Light. Projects → create project → Import data (hover `?` icons, open a select dropdown, use number steppers, pick a radio, open file uploader dropzone) → open an existing project → Data Quality (switch tabs, change unit, scroll the table) → Continue to Training (switch Model to Quantile boosting and back, hover every `?`, start a job to see `st.progress` and any toast/status, press Stop job) → Results (play/pause replay). Theme stays Light at every step; header toolbar and main menu, scrollbars, tooltips, popovers, selects, code, uploader, steppers, radio/checkbox, progress, toast/status all Light.
2. Reload the browser tab: still Light (`?theme=light` in the URL).
3. Repeat 1–2 in Dark: no light surfaces anywhere.
4. ui-designer final PASS on `runs/_ui_review/after/` screenshots.
