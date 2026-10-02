# Data Quality: move, replace, and Training-Data limit suggestions

## Overview

Data Quality (`src/pdm/project_quality_ui.py` `render_quality`) already shows Training / Validation / Testing, an inspect selectbox (`quality_unit_{part}`), a zone chart, and a limits column (direction, Yellow, Red, Cancel, Save). There is no move or replace control. Whole-unit moves already publish one new immutable snapshot via `preview_move` / `move_units` in `src/pdm/data/project_prepare.py`. Display limits already persist with `save_zone_limits` on the same snapshot. This epic adds two things and does not retune training.

1. **Move** the inspected unit into one of the other two sets by calling the existing `move_units`. Counts change. Every current refusal stays.
2. **Replace** by swapping set membership of two units in **one** new snapshot. Counts stay the same. A new pure preview and a new publish function sit beside `move_units`. Do not call `move_units` twice.
3. **Suggest limits** with a deterministic statistical rule fit on admitted Training Data rows only. It fills the existing Yellow/Red session inputs and leaves them dirty. The user still presses Save. It is not a neural net, not a new snapshot, and not a change to `train_signal_run`.

UI copy stays English. Zones stay display labels. Signal models still forecast future signal values, not zone classes.

## Locked limit formula

Source of truth for subtask 02. Implement it in `src/pdm/zone_limit_proposal.py` as `propose_absolute_limits`. No PyTorch. No read of Validation rows, Testing rows, lifetimes, official RUL, schema thresholds, `DEFAULT_LIMITS`, HSE 300/600, or the XJTU per-unit baseline.

### Why this rule

`features.parquet` stores admitted rows only: `unit_id`, `timestamp_s`, `signal`, `gap_before`. `_canonicalize_adapted` already drops non-finite signal or time. `gap_before` is a timeline break for windows (`label_unit` says gaps never change zone labels). It is not a rejected sample and not a new life.

Full-life max or p99 is the wrong place for a warning. HSE differential pressure and XJTU RMS rise into the failure peak, so a full-history high quantile sits on that peak and almost never marks an earlier warning. Import defaults are different and are **not** copied:

- HSE absolute seed is yellow 300 / red 600 (`FIRST_IMPORT_THRESHOLDS`, `DEFAULT_LIMITS["hse_filters"]`).
- XJTU import rule is `initial_baseline_multiple` (median of the first 5 samples, yellow = max(median + 3 SD, 1.25 × median), red = 2 × median) applied **per unit** inside `project_zones.resolve_thresholds`. Data Quality's absolute widgets seed `(1.0, 2.0)` when the committed rule is not absolute (`_seed`).

The suggestion must be one absolute pair for the current direction widget, because `save_zone_limits` / `validate_absolute_thresholds` only accept `mode="absolute"`. The early window is the same idea as the XJTU "first samples are the healthy band", pooled into one pair and kept off the degraded tail.

### Sample

Constants:

- `EARLY_FRACTION = 0.20`
- `EARLY_MIN_POINTS = 5`
- `MIN_EARLY_VALUES = 8`
- quantile `method="linear"` (NumPy 2.5.3)

Inputs: a features frame and the **train unit ids only**, plus `direction` in `{"above", "below"}`. Any other direction raises `ValueError("Threshold direction must be above or below")`.

For each train `unit_id` in sorted order:

1. Keep rows with that `unit_id` whose `timestamp_s` and `signal` are both finite. Ignore every other column, including `gap_before`, labels, and thresholds.
2. Sort by `timestamp_s` with `kind="mergesort"`.
3. Let `n_i` be that length. If `n_i == 0`, skip the unit.
4. `k_i = min(n_i, max(EARLY_MIN_POINTS, math.ceil(EARLY_FRACTION * n_i)))`.
5. Take the first `k_i` rows. That is the early-life prefix. Rows after it, including the failure peak, are out. A `gap_before` flag inside the prefix does **not** drop the row and does **not** end the prefix.

`pool` is the concatenation of those prefixes. `n = len(pool)`.

If `n < 8`, return `ok=False` and do not invent numbers. Reason string, exact:

`Need at least 8 finite Training Data values in the early-life window. This snapshot has {n}.`

`math.ceil(0.20 * n_i)` exceeds 5 only when `n_i > 25`. Shorter units, including the contract fixture's 16-row units, contribute their first 5 finite rows. That still drops the later rows.

Weighting: row pool, capped per unit by `k_i`. One long early prefix contributes more rows than a short one. The cap blocks the degraded tail. It does not equal-weight machines.

### Numbers

Let `quantile(q)` mean `float(numpy.quantile(pool, q, method="linear"))`.

```
m = quantile(0.5)
mad = float(numpy.median(numpy.abs(pool - m)))
robust_sigma = 1.4826 * mad
min_sep = max(0.5 * robust_sigma, 1e-4 * max(abs(m), 1.0), 1e-6)
```

`1.4826` is the normal consistency constant for MAD. Copy the literal. Do not round the outputs. `min_sep >= 1e-6`, so yellow and red cannot be equal.

**above** (high side of the early-life pool; do not flip the widget):

```
yellow = max(quantile(0.90), m)
red = max(quantile(0.99), yellow + min_sep)
```

Then `yellow >= m` and `red > yellow`.

**below** (mirror, low side):

```
yellow = min(quantile(0.10), m)
red = min(quantile(0.01), yellow - min_sep)
```

Then `red < yellow` and `yellow <= m`.

Do not change direction. Do not add `min_sep` again when the quantile gap is already larger than `min_sep` (`max` / `min` already encode that).

### Worked example (tests assert this)

One train unit, 20 finite rows, `signal[i] = i` for `i < 5` else `1000`. `k_i = 5`, `pool = [0, 1, 2, 3, 4]`.

- `m = 2`, `mad = 1`, `min_sep = 0.7413`
- above: `yellow = 3.6`, `red = 4.3413`
- below: `yellow = 0.4`, `red = -0.3413`

Changing `signal[19]` (the peak) does not change either pair. Changing `signal[4]` does. Val/test rows of `1e9` do not change it. Both above numbers stay far below the peak at 1000.

Constant pool of eight `3.0` values: `mad = 0`, `min_sep = 0.0003`, above `(3.0, 3.0003)`, below `(3.0, 2.9997)`.

The limits column formats with `%.2f`. A separation of `0.0003` can **display** as the same two decimals. Do not widen `min_sep` to `0.01` to fix that. Small RMS values would be swamped. Stored floats stay exact; Save persists those floats.

### Return value

```python
{"ok": True, "direction": direction, "yellow": float, "red": float, "n": int,
 "early_fraction": 0.20, "early_min_points": 5}
```

or

```python
{"ok": False, "direction": direction, "n": int, "reason": "<exact sentence above>"}
```

The open snapshot's `split["train"]` is the only id list the UI passes. After a move or replace, the next click loads the new snapshot and therefore the new Training Data.

## Locked swap protocol

Names, beside `move_units` in `src/pdm/data/project_prepare.py`:

- `preview_swap(snapshot, unit_a, unit_b) -> dict` — pure, no I/O
- `swap_units(project_id, unit_a, unit_b, *, expected_snapshot_id, store=None) -> dict`

`swap_units` returns the same keys as `move_units`: `project_id`, `snapshot_id`, `parent_snapshot_id`, `dir`, `split`, `report`, `schema`, `fingerprint`.

Semantics: unit A joins B's set and B joins A's set, in **one** new snapshot. `realized_counts` stay equal to the parent. The empty-set rule passes because counts do not change. Do not implement the swap as two `move_units` calls (two snapshots, and the intermediate state can empty a set).

Parent snapshot directory and its runs stay. Do not delete them. `protocol` stays `whole_unit_project_v1_manual`. Copied `features.parquet`, `units.parquet`, and `feature_schema.json` stay byte-identical. `data_report` / fingerprint / `zone_limits.json` sidecar follow `move_units` (copy the sidecar inside the lock only when the parent has one).

Append one `manual_moves` entry. Move entries stay `{unit_ids, from, to: <set name str>, at}` with no `kind` key. Swap entries are:

```python
{"kind": "swap", "unit_ids": sorted([a, b]),
 "from": {a: set_a, b: set_b}, "to": {a: set_b, b: set_a}, "at": <ISO UTC>}
```

`to` is a dict only on swap entries. Nothing in `src/pdm/` reads `manual_moves` today (tests do). `move_units` must keep appending its string-`to` entry even when older entries are swaps.

`preview_swap` returns `counts`, `problem`, `fixed_units` (same as `preview_move`, including when `units` is missing), plus `from` and `to` maps when the pair is legal, else `None`. `fixed_units` comes from `_fixed_test_units`. A missing `units` frame or missing `source_group` column means no fixed units and no KeyError.

Problem strings, first match wins, `unit_a` before `unit_b`:

- missing id or `unit_a == unit_b`: `Choose two different units.`
- unknown: `Unknown unit: {id}.`
- same set: `Units must be in different sets.`
- either unit is in `fixed_units`, its current set is `test`, and its destination is not `test`: `Official HSE test units stay in Testing Data.`
- a projected set would be empty: `{Set label} would have no units. Keep at least one unit in each set.`

Same extra refusals as `move_units`, same sentences: `LINKED_LEGACY_MOVE_ERROR`, `STALE_SNAPSHOT_ERROR`, `JOB_ACTIVE_MOVE_ERROR` (`RuntimeError`).

Lock discipline matches `move_units`. `ProjectStore.launch_lock` is not re-entrant (`snapshot_path` → `project_path` → `get` takes the same lock). Stage outside the lock. Inside the lock: re-check `heavy_job_active()`, `_load` / `_entry`, owned storage, active snapshot id, optional `zone_limits.json` write, rename, `_activate_snapshot_locked`. No `store.get`, `load_snapshot`, `project_path`, or `snapshot_path` while the lock is held. Compute the destination path before acquiring it.

Extract that publish tail into a private helper both functions call (name: `_publish_manual_snapshot`) so the two paths cannot drift. Existing `move_units` tests stay green without edits. `swap_units` must not call `move_units`.

## Locked UI

`render_quality(snapshot, theme="dark", *, storage_mode="owned") -> bool`. `src/pdm/project_ui.py` passes `storage_mode=str(selected.get("storage_mode") or "owned")`. The bool stays the training-admission result.

No expander. No multiselect. The old `.cursor/tasks/import-cards-quality-20260929/subtask-05-quality-move-control.md` expander is not this UI.

**Linked legacy** (`storage_mode != "owned"`): one `st.info` above the tabs, text exactly `LINKED_LEGACY_MOVE_ERROR`. No move or replace widgets. Suggest stays available. Limits save already works for linked legacy.

**Owned, per open tab**, after `quality_unit_{part}` and only when that tab is open (`tab.open is not False`, same `continue` as the chart):

- `Move to` selectbox: the other two `SPLIT_LABELS`, option values are `validation` / `test` / `train`, `format_func` is the label. Key `quality_move_to:{part}`.
- Button `Move unit`, key `quality_move:{part}`, **not** `type="primary"`. `on_click` reads the inspect id and destination from session, and calls `move_units` for that one id with `expected_snapshot_id` set to the snapshot id this page rendered.
- `Replace with` selectbox: unit ids in the other sets for which a swap would be legal (not an `author_test` unit whose set is `test`). Label `"{unit_id} · {set label}"`. Key `quality_replace_with:{part}`.
- Button `Replace unit`, key `quality_replace:{part}`, not primary. `on_click` calls `swap_units(inspected, partner, expected_snapshot_id=<rendered snapshot id>)`.

Official HSE `author_test` units stay in the inspect list. If the inspected unit is one of them and it is in Testing, both buttons stay disabled and the only membership warning is `Official HSE test units stay in Testing Data.` Do not also show `QUALITY_REPLACE_NONE`. If the partner list is empty for any other reason, omit the empty selectbox (Streamlit rejects empty options), show `No unit in another set can take this place.`, and disable Replace.

Preview caption when the preview has no problem:

- Move: `After the move: Training {a} · Validation {b} · Testing {c} units.`
- Replace: `{unit_a} joins {set_a}; {unit_b} joins {set_b}. Training {a} · Validation {b} · Testing {c} units.` `{set_a}` is B's set label and `{set_b}` is A's set label. They are two fields.

When Testing is the current set or the destination (move), or either side (replace): caption `Changing Testing units after reviewing results makes later Test scores optimistic.`

`heavy_job_active()` once per `render_quality` run (already imported in this module). When true, disable move and replace controls and show `JOB_ACTIVE_MOVE_ERROR`. Same predicate `move_units` uses. Suggest is session-only and stays enabled during a job. Save stays disabled (existing).

Success, `st.success`, once, above the tabs:

- `Moved {unit} to {destination}. A new data snapshot is active; earlier model runs stay with the previous data snapshot.` `{destination}` is the set label (`Training Data`, `Validation Data`, or `Testing Data`).
- `Swapped {unit_a} with {unit_b}. A new data snapshot is active; earlier model runs stay with the previous data snapshot.`

Failure: `st.warning` with the exception string. Catch `KeyError`, `ValueError`, `RuntimeError`, `OSError`. No traceback. On success, pop `quality_move_to:*`, `quality_replace_with:*`, and `quality_unit_*` inside the callback before the rerun. Do not pop them on failure. Do not pop limit keys.

**Suggest** sits in `_render_limits`, after the Red input and before the error line. Button label `Suggest from Training Data`, key `quality_limit_suggest:{project_id}:{snapshot_id}`, not primary. Always show:

`From Training Data only, yellow is the early 90th percentile if the signal rises, or the 10th if it falls. Red is the more extreme of the 99th or 1st percentile and a minimum gap from yellow set by that early spread.`

That sentence is `QUALITY_SUGGEST_LIMITS_CAPTION` (216 characters). "Early" is the first 20% with at least 5 points. The minimum gap is `min_sep`. On the worked example, above red is `4.3413` (`yellow + min_sep`), not q99 `3.96`.

Disable the button when `propose_absolute_limits` is not `ok`, and show `reason` as a caption. `on_click` reloads that snapshot, passes `split["train"]` and the current direction key (`above` / `below`, same fallback as `_widget_rule`). Write the Yellow and Red widget keys first, then compare them to the committed rule, so `_widget_rule` sees the new numbers. Set `limits_key` only when the pair differs, same rule as `_on_limits_change`. Pop the limit error key. Do not call `save_zone_limits`. Do not change the direction widget. Success text: `Suggested yellow {yellow:g} and red {red:g} from Training Data. Press Save to keep them.` Show it with `st.success` under `quality_limit_suggest_note:{project_id}:{snapshot_id}`. Do not store it in `quality_limit_status` (`_state_key("status", ...)`); `_render_limits` shows that key with `st.error`. `_on_limits_change`, Save, and Cancel pop the note key. They already pop the error status key.

`_reset_project_session` also drops keys that start with `quality_move_to:`, `quality_move:`, `quality_replace_with:`, `quality_replace:`, `quality_membership_notice`, `quality_limit_suggest`. Do not widen that into a wipe of `quality_limit_yellow` / `quality_limits:`.

### Help strings (`src/pdm/ui_copy.py`)

`tests/test_worker_and_app.py::test_ui_copy_help_strings_follow_rules` requires every public string constant to be ≤ 220 characters, end with `.`, and contain neither `!` nor `likely`.

- `QUALITY_MOVE_TO_HELP` = `The set this unit will join. The unit leaves its current set. Training, Validation, and Test each keep at least one unit.`
- `QUALITY_MOVE_SUBMIT_HELP` = `Publishes a new data snapshot with this unit in the chosen set. Earlier model runs stay on the previous snapshot.`
- `QUALITY_REPLACE_WITH_HELP` = `A unit from another set that trades places with the unit you are inspecting. Set counts stay the same.`
- `QUALITY_REPLACE_SUBMIT_HELP` = `Publishes one new data snapshot in which the two units trade sets. Earlier model runs stay on the previous snapshot.`
- `QUALITY_SUGGEST_LIMITS_HELP` = `Fills yellow and red from Training Data for the selected direction. Zones only label the chart. Nothing is saved until you press Save.`
- `QUALITY_SUGGEST_LIMITS_CAPTION` = the caption sentence locked above.
- `QUALITY_MOVE_TEST_OPTIMISM` = `Changing Testing units after reviewing results makes later Test scores optimistic.`
- `QUALITY_MOVE_DONE` / `QUALITY_REPLACE_DONE` / `QUALITY_SUGGEST_DONE` = the success templates locked above, with `{unit}`, `{destination}`, `{unit_a}`, `{unit_b}`, `{yellow:g}`, `{red:g}`.
- `QUALITY_REPLACE_NONE` = `No unit in another set can take this place.`

`test_import_and_quality_controls_expose_help` currently asserts there is no `Move to` selectbox and no `Move selected units` button. Update it: those negative checks go away; every new widget has help; `assert not at.multiselect` stays. `test_product_screens_single_primary_button` requires the only primary button on Data Quality to be `Continue to Training`. New buttons stay secondary.

## Phases

1. **Snapshot protocol** — `preview_swap`, `swap_units`, shared publish helper. No Streamlit.
2. **Limit proposal** — pure function and leakage tests. No Streamlit. No training loop.
3. **Membership UI** — move and replace on the open quality tab.
4. **Suggest UI** — button in the limits column.

Phases 1 and 2 do not share files. Phases 3 and 4 both edit `project_quality_ui.py`, so 4 follows 3.

## Dependencies

- Phase 3 calls `swap_units` / `preview_swap` and the existing `move_units` / `preview_move`.
- Phase 4 calls `propose_absolute_limits` and the existing limits session helpers (`limits_key`, `_on_limits_change` dirty rule, `_seed` / `committed_rule`).
- Phase 4 does not depend on swap behavior except that a later suggestion reads whatever snapshot is active.
- No model, loss, scaler, or worker-job-kind dependency.

## Execution order

1. `subtask-01-swap-snapshot-api.md`
2. `subtask-02-early-life-limit-proposal.md` (can be implemented beside subtask 01)
3. `subtask-03-quality-move-replace-ui.md` (after 01)
4. `subtask-04-suggest-limits-ui.md` (after 02 and 03)

## Estimated effort

About 6.5 hours. Subtask 01 ~2 h, 02 ~1.5 h, 03 ~2 h, 04 ~1 h. No training run.

## Verification

After the epic:

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check src tests
```

Each subtask lists its own pytest node ids. UI subtasks use Streamlit `AppTest` in `tests/test_project_ui.py` and the existing help/primary-button checks in `tests/test_worker_and_app.py`. `--smoke` is not a gate. Do not add a full `train_signal_run` as a quality claim.

Leakage for the proposal lives in `tests/test_zone_limit_proposal.py`, not `tests/test_spec_invariants.py`. That file guards the legacy bearings/filters scaler, window, and RUL pipeline. This proposal is project-snapshot display math and never enters training.

## Out of scope

- Multiselect, row moves, window moves, undo, deleting parent snapshots.
- A new worker job kind. Publish stays synchronous, as `move_units` is today, and refuses while `heavy_job_active()` is true.
- Changes to `train_signal_run`, architectures, losses, scalers, or `project_zones.resolve_thresholds`.
- Auto-saving a suggestion, auto-changing direction, or publishing a snapshot just to store a suggestion.
- Rewriting import thresholds (300/600 or the XJTU baseline rule) or `feature_schema.json`.
- The Training-page stale-run caption. It already exists (`test_training_page_mentions_runs_from_previous_snapshot`).
- FastAPI, React, extra neural nets, `filters_full_history`.
- Changing Yellow/Red `format="%.2f"` or `step=0.01`.

## Speculative note (not acceptance criteria)

Equal-weight alternative: take p90/p99 inside each unit's prefix, then the median across units, so a long machine cannot outweigh a short one. Change-point alternative: end the prefix at the first sustained rise so a unit that degrades inside the first 20% does not pull the quantile up. Display alternative: force `min_sep >= 0.01` so `%.2f` shows two different numbers. Direction alternative: pick `above` when the late-life train median exceeds the early median. None of these are the locked rule.

## Assumptions locked without a further product question

- One inspected unit per move. Replace is exactly two units, one snapshot.
- Swap log uses `kind: "swap"` and a dict `to`, instead of two fake move entries.
- Early-life pool is the first 20% of each train unit (at least 5 points, at least 8 points in the pool), p90/p99 or p10/p01, MAD separation as written. Direction follows the widget.
- Suggest does not persist. Move and replace buttons are secondary.
- Linked-legacy projects keep limit edit and suggest, and lose only set-membership controls.
- Proposal tests live in a new module, not `test_spec_invariants.py`.
