# Independent read-only model/runtime review — 2026-09-29

## Verdict

APPROVED. All four findings below were fixed and independently rechecked; see Follow-up review.

Files reviewed: `src/pdm/signal_training.py`, `src/pdm/signal_inference.py`,
`src/pdm/models/signal_recurrent.py`, `src/pdm/worker.py`, `src/pdm/cli.py`,
`tests/test_signal_models.py`, and `tests/test_project_worker.py`.

## Findings

1. **Run artifact path safety (critical).** `load_signal_run()` validates that
   artifact names are basenames, then hashes and loads files without refusing
   symlinked `manifest.json`, `training_contract.json`, or model artifacts. A
   symlink inside a project run can make `sha256_file` or `joblib.load` follow an
   external file. Reject symlinks and verify each resolved child is inside the
   bound run directory before reading it.
2. **Forecast issue time (warning).** `forecast_prefix()` chooses the last
   measurement at or before caller `as_of_s` and forecasts from its timestamp,
   but returns the caller time unchanged. For an arbitrary time between samples
   or after the recording ends, some forecast points may be at or before the
   returned `as_of_s`, violating the positive-horizon contract. Return the
   actual issue time, or reject/mark unsupported cursors, and test one.
3. **Quantile interval labeling (warning).** Training saves
   `interval_status=unvalidated_pointwise_quantiles`, yet the forecast mapping
   omits calibration status. Results checks `calibration_status` on forecast or
   schema, so users can see lower/upper curves without the saved limitation.
   Propagate status from the verified run manifest to the forecast/UI.
4. **Boosting validation selection (warning).** Recurrent models retain the
   best validation epoch. Quantile boosting fits one configured candidate and
   only reports validation MAE afterward. The approved contract says
   Validation selects model/parameters before Test evaluation. Select a
   documented candidate using Validation or change the stated selection
   contract to match the implemented fixed configuration.

## Verified strengths

The signal windows stay within `gap_before` segments, mask targets unavailable
at a recording end, and use only the declared `signal` input. The scaler is
fit from Training units. Test windows are constructed after fitting and
validation in `train_signal_run()`. Saved runs bind project, snapshot,
parameters, schema, and artifact hashes. Worker jobs carry project/job IDs,
coordinate launch with the registry lock, and use job-scoped stop flags.

This is code inspection; the parent integration run and live browser training
remain separate gates.

## Follow-up review

All four findings were addressed by B: saved run children reject symlinks;
forecast `as_of_s` is the actual observed issue time; forecast results carry the
saved calibration status; and boosting compares two iteration candidates by
unit-equal Validation MAE before Test scoring. The A+B focused tests passed
together (33 tests) after those changes. The browser outcome remains a separate
integration gate owned by the parent.
