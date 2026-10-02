# Subtask 2: Early-life absolute limit proposal

## Goal
Add a pure function that proposes absolute yellow and red numbers from admitted Training Data only, using the early-life percentile rule locked in the master plan, and prove Validation and Testing rows cannot change it.

## Context
Data Quality already edits absolute limits in session state and persists them with `save_zone_limits`. Import defaults (HSE 300/600, XJTU `initial_baseline_multiple`, generic `DEFAULT_LIMITS` `(1.0, 2.0)`) stay the widget seed when nothing is committed. They are not this proposal. Training histories include degradation, so a full-life p99 sits on the failure peak. Subtask 04 only calls this function. This subtask does not touch Streamlit, `train_signal_run`, or `project_zones.resolve_thresholds`.

The formula below is the master-plan lock. If a comment here ever disagrees with `master-plan.md`, the master plan wins.

## Acceptance Criteria
- [ ] `src/pdm/zone_limit_proposal.py` defines `propose_absolute_limits(features, train_unit_ids, direction) -> dict`. Imports are `math`, numpy, and pandas only (no torch, no `pdm.train`, no `pdm.worker`).
- [ ] Constants: `EARLY_FRACTION = 0.20`, `EARLY_MIN_POINTS = 5`, `MIN_EARLY_VALUES = 8`. Per train unit, finite `timestamp_s` and `signal` only, sort `timestamp_s` with `mergesort`, `k_i = min(n_i, max(5, math.ceil(0.20 * n_i)))`, first `k_i` rows. `gap_before` does not drop or end the prefix. Unit ids are processed in sorted order. Rows whose `unit_id` is not in `train_unit_ids` are ignored.
- [ ] `n < 8` returns `ok=False`, no yellow/red, and `reason` exactly `Need at least 8 finite Training Data values in the early-life window. This snapshot has {n}.` Direction outside `above`/`below` raises `ValueError("Threshold direction must be above or below")`.
- [ ] Otherwise `ok=True` and the master-plan formulas. `method="linear"`. `robust_sigma = 1.4826 * mad`. `min_sep = max(0.5 * robust_sigma, 1e-4 * max(abs(m), 1.0), 1e-6)`. Above: `yellow = max(q90, m)`, `red = max(q99, yellow + min_sep)`. Below: `yellow = min(q10, m)`, `red = min(q01, yellow - min_sep)`. No rounding. Direction argument is echoed and not inferred.
- [ ] Worked example, absolute tolerance `1e-12`. Pool `[0, 1, 2, 3, 4]` from a 20-row unit whose tail is `1000`. Above `(3.6, 4.3413)`. Below `(0.4, -0.3413)`. Replacing the tail with `1e6` keeps both pairs. Replacing `signal[4]` changes them. Eight constant `3.0` values: above `(3.0, 3.0003)`, below `(3.0, 2.9997)`.
- [ ] When the quantile gap already exceeds `min_sep`, red equals that quantile (above: q99; below: q01), not `yellow ± min_sep` stacked on top. Fixture: one train unit with at least 36 finite rows so `k_i = 8` (`math.ceil(0.20 * 36) = 8`). The first 8 signals are `[-100, 0, 0, 0, 0, 0, 0, 100]`; later rows must not change the pair. Do not use an 8-row unit. On that 8-value pool, `min_sep = 0.0001`. Above, yellow is 30 within `1e-12` and red is 93 within `1e-12`, and `red == q99`. Below, yellow is -30 within `1e-12` and red is -93, and `red == q01`. Do not use a uniform ramp. Seven `0`s and one `10` makes above red equal q99 (`9.3`) but below q01 is `0`, so red becomes `yellow - min_sep` (`-0.0001`); that pool does not lock both sides.
- [ ] Leakage: a features frame that also contains validation and test rows. Setting those signals, timestamps, and any extra columns (`rul`, `event`, `official_rul`) to extreme values does not change the returned floats (`==` on yellow and red). Removing or editing one in-prefix train row does. A `gap_before=True` row inside the prefix counts. A spike after `k_i` does not, even if `gap_before` is true.
- [ ] The function never reads `split["validation"]`, `split["test"]`, or a schema. Its signature has no schema and no units table. Passing a different train-id list (the unit that would move) changes the proposal when that unit's prefix differs from the remaining train units.
- [ ] Returned above pair always has `yellow < red`, both finite, `yellow >= median(pool)`. Below pair always has `yellow > red`, both finite, `yellow <= median(pool)`. Neither pair equals `(300, 600)` or `(1, 2)` unless the pool itself produces those numbers.

## Implementation Notes
- Files: `src/pdm/zone_limit_proposal.py` (new), `tests/test_zone_limit_proposal.py` (new).
- Do not add this test to `tests/test_spec_invariants.py`. That module is the bearings/filters scaler, window, and RUL gate. This rule never enters training.
- Do not call the function from `signal_training`, `train.py`, or `prepare_project`.
- `math.ceil`, not `numpy.ceil` (float).
- Contract fixture note, not a hardcoded assertion: `tests/fixtures/project_contract/sensor.csv` has 9 units × 16 rows. Seed 19 and weights 0.7/0.15/0.15 allocate 6/2/1, so the train early pool is 6 × 5 = 30. A test may `assert propose_absolute_limits(...)["ok"]` on `make_contract_snapshot` and `assert n == 30` only if it documents that allocation. Prefer synthetic frames for the numeric locks so a split-seed change cannot break the formula tests.

## Dependencies
None. Can land beside subtask 01.

## Verification
```bash
.venv/bin/python -m pytest tests/test_zone_limit_proposal.py -q
.venv/bin/python -m ruff check src/pdm/zone_limit_proposal.py tests/test_zone_limit_proposal.py
```

Test names:

- `test_worked_example_above_and_below_ignore_failure_peak`
- `test_constant_pool_minimum_separation`
- `test_wide_gap_uses_quantile_not_stacked_separation`
- `test_val_and_test_mutations_do_not_change_proposal`
- `test_train_prefix_edit_changes_proposal`
- `test_gap_inside_prefix_counts_and_tail_gap_does_not`
- `test_fewer_than_eight_early_values_is_disabled`
- `test_direction_is_respected_and_invalid_direction_raises`
- `test_different_train_ids_change_proposal`
- `test_nan_train_signal_is_dropped_before_the_prefix`
