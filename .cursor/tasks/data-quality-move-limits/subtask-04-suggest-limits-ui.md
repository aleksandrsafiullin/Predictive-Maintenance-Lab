# Subtask 4: Suggest yellow and red from Training Data

## Goal
Add a `Suggest from Training Data` button beside the Yellow and Red inputs that fills those session values from `propose_absolute_limits` and leaves Save for the user.

## Context
Subtask 02 locks the formula in `propose_absolute_limits`. `_render_limits` in `src/pdm/project_quality_ui.py` already owns direction, Yellow, Red, Cancel, and Save. `limits_key` is the dirty session rule. `save_zone_limits` writes the sidecar and does not publish a snapshot. `zone_schema` paints the chart from the session rule before Save. Subtask 03 has already edited this file for move/replace; this subtask only adds the suggest control and its copy. Do not change the formula, `train_signal_run`, or the number-input `format` / `step`.

## Acceptance Criteria
- [ ] Button label `Suggest from Training Data`, key `quality_limit_suggest:{project_id}:{snapshot_id}`, inside the limits container after the Red input and before the error line. Not `type="primary"`. `help=QUALITY_SUGGEST_LIMITS_HELP`.
- [ ] Caption always visible, exact `QUALITY_SUGGEST_LIMITS_CAPTION`: `From Training Data only, yellow is the early 90th percentile if the signal rises, or the 10th if it falls. Red is the more extreme of the 99th or 1st percentile and a minimum gap from yellow set by that early spread.`
- [ ] The click callback loads the snapshot id it was rendered with, calls `propose_absolute_limits(features, split["train"], direction)`, and passes no validation ids, test ids, or schema. Direction is the current radio value (`above` / `below`), including the existing `_widget_rule` fallback for a label that starts with `Signal rises`. The callback does not write the direction key.
- [ ] On `ok`, the callback writes the Yellow and Red widget keys first (same overwrite style as `_on_limits_cancel`), then compares those new numbers with the committed rule. `_widget_rule` must see the suggestion, not the previous inputs. It pops the limit error key. It sets `limits_key` only when the pair differs from the committed rule, using the same comparison as `_on_limits_change`. It does not call `save_zone_limits`. Snapshot id, `feature_schema.json`, and fingerprint bytes stay unchanged. `zone_limits.json` stays absent until Save.
- [ ] Success text, once, via `st.success`: `Suggested yellow {yellow:g} and red {red:g} from Training Data. Press Save to keep them.` Session key `quality_limit_suggest_note:{project_id}:{snapshot_id}`. Do not store it in `quality_limit_status` (`_state_key("status", ...)`); `_render_limits` shows that key with `st.error`. The chart limit lines update to the suggested pair on that rerun (session preview, existing `zone_schema` path).
- [ ] When `ok` is false, the button is disabled and the `reason` caption is the function's `reason` (fewer than 8 early-life values). No traceback.
- [ ] Suggest stays enabled while `heavy_job_active()` is true. Save stays disabled. A suggestion during a job writes no sidecar. Linked legacy still shows the button.
- [ ] Cancel after a suggestion restores the committed rule and writes nothing. A following Save persists the suggestion through the existing save path and still does not change `schema["thresholds"]` or the snapshot id.
- [ ] After a move or replace, a new suggestion matches `propose_absolute_limits` on the **new** snapshot's `split["train"]`, not the parent's. Cover this with a real call, not only a mock. A monkeypatch that records the id list is allowed as an extra assertion that validation and test ids were not passed.
- [ ] `_on_limits_change`, Save, and Cancel pop `quality_limit_suggest_note:{project_id}:{snapshot_id}`. They already pop the error status key; that pop stays. `_reset_project_session` clears keys starting with `quality_limit_suggest` without clearing `quality_limit_yellow` or `quality_limits:`.
- [ ] Help-string test still passes (`≤ 220`, trailing period, no `!`, no `likely`). The suggest button is included in `test_import_and_quality_controls_expose_help`. Data Quality's only primary button remains `Continue to Training`.

## Implementation Notes
- Files: `src/pdm/project_quality_ui.py`, `src/pdm/ui_copy.py`, `src/pdm/project_ui.py` (reset prefix only), `tests/test_project_ui.py`, `tests/test_worker_and_app.py`.
- Do not edit `src/pdm/zone_limit_proposal.py` unless a test exposes a bug in the locked formula. Formula changes belong back in the master plan, not a silent tweak.
- The contract snapshot's train early pool is large enough that the button is enabled (`ok` and `n == 30` under the current 6/2/1 split). Assert against `propose_absolute_limits(...)` rather than pasted floats.
- The tiny snapshot in `test_import_and_quality_controls_expose_help` has two train rows, so the button is disabled and the reason caption is present. Help must still be set on that disabled button. `at.exception` stays empty.
- Catch `KeyError`, `ValueError`, `RuntimeError`, `OSError` from the load/propose path and `st.warning` the message.
- Zones remain whatever `label_unit` already does with the session absolute rule. Do not teach the chart a new mode.

## Dependencies
Subtasks 02 and 03.

## Verification
```bash
.venv/bin/python -m pytest tests/test_zone_limit_proposal.py tests/test_project_ui.py tests/test_worker_and_app.py -q -k "suggest or quality_limit or import_and_quality or product_screens"
.venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check src tests
```

New AppTest names:

- `test_quality_suggest_fills_inputs_without_saving`
- `test_quality_suggest_disabled_when_early_pool_is_short`
- `test_quality_suggest_respects_below_direction`
- `test_quality_suggest_cancel_reverts_and_save_writes_sidecar_only`
- `test_quality_suggest_allowed_during_job_and_on_linked_legacy`
- `test_quality_suggest_after_move_uses_new_training_ids`

No `--smoke` run. No `train_signal_run`.
