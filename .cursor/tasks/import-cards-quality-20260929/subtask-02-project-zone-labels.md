# Subtask 2: Shared project zone-label rule (`src/pdm/project_zones.py`)

## Goal
One function family that turns a snapshot's saved threshold rule into per-row green/yellow/red/unknown labels, shared by Data Quality, card counts and Results inference.

## Context
Today the project threshold rule lives privately in `signal_inference._resolved_thresholds` and the red comparison is inlined in `forecast_prefix`. `health_zones.signal_zone_reference` is a different legacy-lab rule (persistence, per-axis RMS) and must not be used for project snapshots (see master plan findings). The overlay must match what Results reports.

## Acceptance Criteria
- [ ] New `src/pdm/project_zones.py` with:
  - `ZONES = ("green", "yellow", "red", "unknown")`.
  - `resolve_thresholds(schema: Mapping, prefix: pd.DataFrame) -> dict` — body moved verbatim from `signal_inference._resolved_thresholds` (same keys: `mode, direction, yellow, red, status, reason`, plus `baseline, baseline_n` in baseline mode).
  - `is_beyond(values, limit, direction) -> np.ndarray[bool]` inclusive (`>=` above, `<=` below). NaN → False.
  - `zone_labels(values, thresholds) -> np.ndarray[str]`: red if beyond red; elif yellow is not None and beyond yellow → yellow; else green; `unknown` everywhere when `status != "available"`.
  - `label_unit(frame: pd.DataFrame, schema: Mapping) -> pd.DataFrame` for one unit: sorts by `timestamp_s`, returns `unit_id, timestamp_s, signal, gap_before, zone` (+ `yellow_limit`, `red_limit`, NaN where unknown). Absolute: one resolve on the whole unit. Baseline mode: `n = int(schema["thresholds"].get("baseline_n", 5))`; thresholds from `resolve_thresholds(schema, unit.iloc[:n])`; rows at position `< n - 1` → `unknown` (exactly `n - 1` unzoned rows; causal, identical to `forecast_prefix` availability).
  - `zone_counts(features: pd.DataFrame, unit_ids: Iterable[str], schema) -> dict[str, int]` summing `label_unit` over units (keys = `ZONES`).
  - `describe_rule(schema) -> str` — English caption **built from `schema["thresholds"]`**, never hardcoded numbers (W10):
    - absolute: "Zones use the limits saved with this data: yellow at {op} {yellow:g} {unit}, red at {op} {red:g} {unit} (instantaneous, per measurement)." with `op` = ≥ / ≤ from `direction`; yellow clause omitted when `yellow is None`.
    - `initial_baseline_multiple`: "Zones use each unit's initial baseline: median of its first {baseline_n} measurements; yellow = max(median + {onset_sigma:g} × SD, {onset_ratio:g} × median), red = {red_ratio:g} × median. The first {baseline_n - 1} measurements are not zoned." (defaults 5 / 3.0 / 1.25 / 2.0 only when the key is absent, same as `resolve_thresholds`).
    - If `thresholds["note"]` exists (legacy HSE "Provisional laboratory bands; not confirmed industrial fault limits"), append it as a sentence.
    - Unsupported/invalid rule → "No valid yellow/red limits are saved with this data, so measurements are not zoned."
- [ ] `signal_inference.py` imports `resolve_thresholds, is_beyond` from `project_zones`; `_resolved_thresholds = resolve_thresholds` alias kept; `already_red` and crossing `hit` use `is_beyond`. Behavior unchanged (existing `tests/test_signal_models.py` and `tests/test_project_replay.py` pass untouched).
- [ ] Gaps do not change labels; labels never read `rul`, `event_*`, `split`, or future rows beyond the rule's own definition (baseline uses only the first `baseline_n` rows).
- [ ] No labels are written to snapshots or used as model inputs.

## Implementation Notes
- Keep `project_zones.py` free of Streamlit/Plotly imports (pure numpy/pandas).
- Invalid direction/mode → `resolve_thresholds` already returns `status="unavailable"` → all `unknown`.
- `yellow` may be `None` in absolute mode → only green/red.
- Schema argument is the full snapshot `schema` dict (`schema["thresholds"]`), matching `run["schema"]` usage.

## Dependencies
None (parallel with 01).

## Verification
New `tests/test_project_zones.py`:
- absolute above: exactly yellow → yellow, exactly red → red, below yellow → green.
- absolute below (signed generic): mirrored.
- baseline multiple with a non-default `baseline_n` (e.g. 3) and non-default `onset_sigma`/`onset_ratio`/`red_ratio`: first `baseline_n - 1` rows unknown; thresholds equal `resolve_thresholds(schema, first n rows)`; `describe_rule` text contains those exact numbers.
- `describe_rule` includes `thresholds["note"]` when present.
- gaps: toggling `gap_before` leaves labels unchanged.
- invalid rule → all unknown + "not zoned" description.
- `zone_counts` sums match per-unit labels.
- Parity with Results (W6), in `tests/test_signal_models.py` using its existing `contract` fixture (real `forecast_prefix`; `tests/project_results_harness.py` stubs it, and `tests/test_project_replay.py` does not call it): train a tiny run, and for Test-unit prefixes that have **at least `history_length` rows since the last gap** (so `forecast_prefix` issues points), assert `crossing["status"] == "already_red"` iff `label_unit(prefix)["zone"].iloc[-1] == "red"`. Skip prefixes shorter than that.
- `tests/test_spec_invariants.py`: `label_unit` output depends only on `unit_id`, `timestamp_s`, `signal`, `gap_before` (dropping other columns gives the same result).
```bash
.venv/bin/python -m pytest tests/test_project_zones.py tests/test_signal_models.py tests/test_project_replay.py -q
.venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check src tests
```
