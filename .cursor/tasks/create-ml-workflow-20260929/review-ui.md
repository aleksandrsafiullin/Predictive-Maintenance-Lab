# Independent review of subtask C — 2026-09-29

## Verdict

APPROVED after rechecking the UI fixes and legacy AppTest entrypoint wiring. Reviewed read-only: `project_ui.py`, `project_quality_ui.py`, `project_training_ui.py`, `project_results_ui.py`, the `app.py` entrypoint, UI tests, and A/B mapping contracts. Root owns live browser QA.

## Findings sent to C

1. **Fixed**: Browser upload stages are tracked and removed if staging or launch fails. The prelaunch race with another project's worker was rechecked; cleanup uses this attempt's unique job ID instead of global worker liveness. B worker cleans job-owned upload stages after terminal import outcomes.
2. **Fixed**: Data Quality Continue now checks `available_signal_engines` and explains when continuous histories are too short.
3. **Fixed**: The Data Quality unit chart inserts breaks at `gap_before=True`.
4. **Fixed**: `_parse_horizons` rejects non-finite numbers before queueing a worker.
5. **Fixed**: Import and train status fragments present the `stopping` state.

## Positive checks

The active `app.py` entrypoint shows only the five-step project workflow, and legacy research helpers remain callable. Project list, creation, recoverable archive, folder/manual split controls, model selector, run binding, and replay use the agreed contracts. Results derives actual observations through the cursor, draws saved numeric forecast points and resolved thresholds, labels boosting quantiles as unvalidated, and has Play/Pause/Reset and run/unit state separation. Focused C/new-workflow and legacy helper AppTests passed (55 tests); Ruff passed. The five older research UI test files now invoke `tests/legacy_app_harness.py` instead of the new product `app.main`; existing behavior assertions remain. Their 109-test suite passed on independent recheck. Root's live browser QA and full suite rerun are separate integration gates.

Root subsequently verified 9,216-row replay advancing at the 0.6-second UI interval and a 579-test full suite with Ruff. Throughput on much larger prepared snapshots remains unbenchmarked; no speculative caching was added to the runtime.

Light theme follow-up reviewed: uploader button/help CSS is scoped to the project workflow marker; both project charts set light paper/plot/font/grid colors, and the actual, forecast, yellow/red limits, crossing marker, and Now line use darker light-specific colors. The original dark palette remains. Focused chart/UI tests and Ruff passed after the palette change; visual screenshot confirmation remains with root's browser QA.

Final integration gate: Root restarted the current app and verified that Light remains selected through 14 Results Play ticks after allowing the Appearance change to settle. The body stayed light, both theme markers stayed light, and the chart axis titles remained legible. The complete suite passed with 581 tests. Subtask C is frozen for handoff.
