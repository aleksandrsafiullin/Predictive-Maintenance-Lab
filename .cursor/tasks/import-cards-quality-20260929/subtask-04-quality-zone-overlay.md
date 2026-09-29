# Subtask 4: Traffic-light overlay and zone summary on Data Quality

## Goal
On the Data Quality selected-unit chart, paint the green/yellow/red zone of every admitted measurement from the snapshot's threshold rule, draw the yellow/red limits exactly like Results, and show zone counts per set and per unit with an honest caption.

## Context
`project_quality_ui.render_quality` draws one `series_observed` line via `gap_safe_trace` with no zones. `project_results_ui.replay_figure` already draws yellow/red bands + dashed limit lines from `thresholds`. Subtask 02 provides `pdm.project_zones` (single rule). Project models forecast numeric signal; zones are derived — the caption must say both.

## Acceptance Criteria
- [ ] Extract the band/limit drawing from `project_results_ui.replay_figure` (lines ~67–90) into `project_chart_style.add_threshold_layers(fig, thresholds: dict, values: list[float], theme: str) -> None`; `replay_figure` calls it (Results output unchanged — existing results tests pass). `yellow_trace`/`red_trace` lists handled exactly as today.
- [ ] Quality chart for the selected unit:
  - base line: `gap_safe_trace(frame)` in `tokens(theme)["series_observed"]`, lines only, `showlegend=False`;
  - zone markers: one `go.Scatter(mode="markers")` per zone present in `project_zones.label_unit(frame, schema)`, color `zone_colors(theme)[zone]` (`unknown` → `zone_colors(theme)["unknown"]`), names "Green · {n}", "Yellow · {n}", "Red · {n}", "Not zoned · {n}", horizontal legend;
  - limits: `add_threshold_layers(fig, thresholds, values, theme)` with thresholds from `project_zones.resolve_thresholds(schema, frame.iloc[:baseline_n])` in baseline mode, else the whole frame; drawn only when `status == "available"`;
  - hover shows time, value, zone.
- [ ] Gaps still break the line (no connection across `gap_before`).
- [ ] Caption under the chart: `project_zones.describe_rule(schema)` (built from `schema["thresholds"]` incl. `baseline_n`, `onset_sigma`, `onset_ratio`, `red_ratio`, and `thresholds["note"]` when present — W10) + "Signal models in this project learn to forecast future {label} values, not zone classes. Zones are derived from those values with the same limits, and Results uses them to report the expected red entry."
- [ ] No usable rule: plain line + "No valid yellow/red limits are saved with this data, so measurements are not zoned." — no exception.
- [ ] Zone summary per tab, under Units/Admitted rows/Gaps: "Zones: Green {g} · Yellow {y} · Red {r} · Not zoned {u} rows" from `project_zones.zone_counts(features, split[part], schema)`. Per selected unit: legend counts.
- [ ] Measurements table gains a "Zone" column (Green/Yellow/Red/Not zoned).
- [ ] Zone labelling (split counts + unit chart) runs **only for the open tab**: check `tab.open` (Streamlit 1.63 `st.tabs(..., key=, on_change="rerun")` from subtask 03) before computing (W2). Split counts cached with `st.cache_data` keyed on `(project_id, snapshot_id, part)` (immutable snapshot); pass the DataFrame as a hash-excluded `_features` argument.
- [ ] Light and dark consistent (only `tokens`/`zone_colors`; no hex literals).

## Implementation Notes
- Files: `src/pdm/project_quality_ui.py`, `src/pdm/project_chart_style.py`, `src/pdm/project_results_ui.py` (refactor only), `src/pdm/ui_copy.py` (`QUALITY_ZONE_MODEL_CAPTION`, `QUALITY_ZONE_SUMMARY_HELP`, `QUALITY_NO_ZONES`).
- Do not import `health_zones` here.
- AppTest fixtures that monkeypatch `load_snapshot` with a minimal schema (`{"signal_label": "Vibration", "signal_unit": "g"}`, no thresholds — `test_quality_three_sets_and_training_gate`) must still render → exercises the "not zoned" fallback. Their fake `snapshot_id` ("snapshot1") keeps the cache key valid.
- If `tab.open` is unavailable when the tabs have no `on_change`, subtask 03 already sets `on_change="rerun"`; don't remove it.

## Dependencies
Subtask 02 (rule), subtask 03 (`quality_tab` key + `on_change="rerun"` on `st.tabs`).

## Verification
`tests/test_project_ui.py`:
- `test_quality_chart_paints_zones_from_snapshot_limits` — `make_contract_snapshot` (absolute, above, yellow 0.4, red 0.8): zone trace names in the Plotly figure carry counts equal to `label_unit(...)`; caption contains "forecast future" and "not zone classes"; zone summary text present.
- `test_quality_chart_without_limits_is_not_zoned` — minimal schema fixture renders and shows `QUALITY_NO_ZONES`.
- `test_quality_baseline_rule_caption_uses_schema_values` — monkeypatched snapshot with `initial_baseline_multiple`, `baseline_n=4`, `red_ratio=2.5`, `note="Provisional"`: caption contains "first 4", "2.5 × median", "Provisional"; first 3 rows "Not zoned".
- `test_quality_zone_labels_only_for_open_tab` — spy on `project_zones.zone_counts`: with `quality_tab="Training Data"` only the train part is labelled.
- Existing Results tests (`tests/project_results_harness.py`, `tests/test_project_replay.py`, `tests/test_signal_models.py`) unchanged and green.
- Manual at `http://127.0.0.1:8501`: Data Quality in light/dark on a unit that crosses red.
```bash
.venv/bin/python -m pytest tests/test_project_ui.py tests/test_project_replay.py tests/test_project_zones.py -q
.venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check src tests
```
