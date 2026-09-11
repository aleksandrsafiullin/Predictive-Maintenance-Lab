# Subtask 2: Causal gap flags shared by prepare and inference (R2)

## Goal

Compute filter `gap_before` **causally** at prepare **and** recompute the same causal flags in `build_windows` / `count_window_eligibility` so stored full-file parquet cannot drive train eligibility. Gate **train and eval** against expected `GAP_RULE_VERSION` (`missing ≠ expected`). Align timestamp uniqueness/monotonicity with `valid_history_window`.

## Context

Review R2 (HIGH). `_measurements_from_csv` currently calls `gap_before_from_delta_t(..., causal=False)` (full-file median). `Predictor.predict_from_history` and `ReplaySource.prefix` recompute `causal=True`. Offline-allowed windows can be rejected at inference; future long intervals can flip **offline** flags at past times.

**Plan-review Critical:** this patch does **not** re-prepare real HSE/XJTU data. `fingerprint_mismatches` field-to-field equality treats missing==missing as OK, so old parquet would keep training on full-file flags while replay uses causal. Must not ship that.

Review fixture (must be a named test): timestamps `[0, 6, 12, 42, 142, 242, 342]`, `history_length=3`, `gap_multiplier=3`, `sampling_interval_s=6`. Window ending at 42 (`[6, 12, 42]`) must have **matching** offline vs online eligibility (both reject is fine — full-file currently accepts, causal rejects). Appending future long intervals must **not** change the causal flag or eligibility at t=42.

## Acceptance Criteria

- [ ] `src/pdm/data/filters.py::_measurements_from_csv` writes parquet `gap_before` with `causal=True` (sequential prefix median, same function as inference).
- [ ] `causal=False` remains available for **descriptive diagnostics only** (e.g. data_report counts). It must not feed train / val / eval / replay eligibility.
- [ ] **Belt-and-suspenders:** `build_windows` and `count_window_eligibility` recompute causal `gap_before` from timestamps (filters only) via the **same helper** as inference (`recompute_filter_gap_before` / `gap_before_from_delta_t(..., causal=True)`). Stored parquet flags — including poisoned full-file or flipped bits — must **not** decide eligibility.
- [ ] `build_windows` rejects windows whose timestamps are non-finite, not unique, or not strictly increasing — equivalent to `valid_history_window` (`timestamps_not_strictly_increasing` / `non_finite_timestamps`).
- [ ] `count_window_eligibility` uses the same skip reasons (add a timestamp-exclusion bucket if needed).
- [ ] For every synthetic t before event/censoring: offline eligibility (`build_windows`) equals online (`valid_history_window` / `ReplaySource` / `Predictor` path). Changing future rows does not change input, prediction availability, or gap flag at a past timestamp.
- [ ] `GAP_RULE_VERSION = "causal_v1"` in `src/pdm/windows.py` is stored on `feature_schema.json` and `build_processed_fingerprint` (`src/pdm/data/prepare.py`).
- [ ] Version gate compares against the **expected constant**, not only saved vs live equality. Missing or any value ≠ `GAP_RULE_VERSION` is incompatible. Gate **`run_training` and `bind_evaluation_to_run` / eval** (not evaluate fingerprints alone). `load_processed` may still return frames for the Data screen but must expose the version so train/eval callers fail closed (`require_current_gap_rule` or an explicit check after load).
- [ ] `configs/filters.yaml` gap comments no longer say prepared parquet stores full-file median for offline windows.
- [ ] Do **not** enable `filters_full_history`. Do **not** invent MAT/CSV fields. Do **not** change split protocol. Do **not** run prepare on real XJTU/HSE data.

## Implementation Notes

**Files:** `src/pdm/data/filters.py`, `src/pdm/windows.py` (`GAP_RULE_VERSION`, `gap_before_from_delta_t`, `recompute_filter_gap_before`, `valid_history_window`, `build_windows`, `count_window_eligibility`), `src/pdm/predict.py`, `src/pdm/replay.py`, `src/pdm/data/prepare.py` (`write_processed_version`, `build_processed_fingerprint`, optionally `load_processed`), `src/pdm/train.py` (`run_training` after `load_processed`), `src/pdm/evaluate.py` (`bind_evaluation_to_run`, `fingerprint_mismatches` / `_JSON_COMPARE_FIELDS` — add a **constant compare** helper), `configs/filters.yaml`

- Do **not** implement `fingerprint_mismatches` as “both missing ⇒ match”. Example: `expected = GAP_RULE_VERSION`; if `saved.get("gap_rule_version") != expected` or `current.get("gap_rule_version") != expected` → differing. Same for a dedicated `assert_gap_rule_current(fp)` used by train and eval.
- Keep inference recomputing causal gaps on the prefix. After belt-and-suspenders, `build_windows` also ignores stored flags for filters.
- Extract a small shared helper for the timestamp check used by both `build_windows` and `valid_history_window`.
- `FEATURE_PIPELINE_VERSION` (`v2_raw_first`) is **not** this bump. New field `gap_rule_version`.
- Bearings: do not invent filter-style Δt medians. Recompute path is filters-only (`dataset_id == "filters"`). Bearings keep existing file-index `gap_before` behavior.
- Update `test_filter_causal_gap_uses_prefix_median_not_full_file` only where it assumed prepare=full-file for **eligibility**. Keep the contrast that full-file diagnostics **can** flip when future dense samples arrive; causal must not.

**Tests** in `tests/test_spec_invariants.py`:

- `test_review_r2_window_61242_offline_online_match` — exact timestamps above; `build_windows` include/exclude of end t=42 equals `valid_history_window` on the prefix ending at 42; append extra long intervals and assert causal flags + eligibility at t=42 unchanged; full-file diagnostic **may** flip.
- `test_build_windows_matches_valid_history_even_when_stored_flags_poisoned` — parquet `gap_before` all True / full-file pattern; `build_windows` still matches causal `valid_history_window`.
- `test_stale_or_missing_gap_rule_version_refuses_train_and_eval` — fingerprint / processed dir without `gap_rule_version` or with `"fullfile_v0"`: `run_training` and `bind_evaluation_to_run` (or evaluate) raise incompatible/data error. Do **not** require a real dataset prepare; tmp_path stub is enough (monkeypatch `load_processed` if needed).
- Prefix invariance: future rows do not change `Predictor` / `ReplaySource` gap flag or `valid_history_reason` at a past t.
- Duplicate / non-monotonic timestamps: `build_windows` skips the window; `valid_history_window` returns the same failure reason.
- Synthetic `write_processed_version` (tmp_path) includes `gap_rule_version == "causal_v1"`.

Do not run `.venv/bin/python -m pdm prepare` on real data.

## Dependencies

Subtask 01 first under orchestration (shared `tests/test_spec_invariants.py` only). No production-file dependency.

## Verification

```bash
.venv/bin/python -m pytest tests/test_spec_invariants.py -q -k "gap or causal or window or gap_rule"
.venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check src tests
```
