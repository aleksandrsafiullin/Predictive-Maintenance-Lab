# Live monitor

Live monitor is one common workflow page after Results. It uses the project's
active snapshot and saved models, including registered sensor snapshots with
Training, Validation, Calibration and Testing roles. No new training is needed
to open an existing compatible model.

## Incoming measurements

Select **Live folder**, choose a **Saved model**, set the **Watched folder** and
refresh interval. The folder is on the computer running the app; it can be a
locally synced SharePoint/OneDrive directory. The page reads its CSV files.

```csv
unit_id,timestamp_s,signal
pump-01,0,0.31
pump-01,60,0.32
```

The signal column can instead use the imported project signal name or `value`.
`timestamp` with UTC date-times can replace `timestamp_s`. Empty machine IDs,
nonfinite times and nonfinite signals are excluded. Duplicate machine/time rows
keep the last value in sorted file order. Optional `gap_before` and
`component_cycle_id` columns preserve declared history breaks.

The table shows the newest received measurement, current zone, forecasted first
center crossing of the red limit, total received measurements and history used.
Machines are ordered by zone and forecasted red time. Click a machine's row to
show its forecast below the table. The selected machine and row highlight stay
selected when table order changes. Red is a saved signal limit, not an automatic
equipment failure diagnosis. A predicted center crossing is not a calibrated
RED event-time interval.

## Forecast and Calibration

Current GRU, LSTM and MaleCNS use `long_forecast_run.forecast_observations`, the
same function used by Results. At least 60 continuous positive observations are
required; all available observations in the current continuous segment are
used. Missing cadence intervals, declared gaps and component cycle changes
break the history. Observations after the issue time are excluded before model
inputs are extracted. A live machine need not belong to the saved training set.

The saved model's complete horizon is preserved. **Use saved Calibration
corridor** applies its verified Calibration result, including the saved minimum
and maximum full widths (defaults 20% and 45%). Calibration changes affect the
next refresh and invalidate cached interval predictions. Without an applied
Calibration result, the saved training corridor is used. Corridor width is not
a confidence guarantee; this page does not establish new forecast quality.

Archived long models retain their original saved history policy and parameters.
Original direct signal models also work through their original model contract.
Original joint-path/learned-distribution corridor protocols are not offered as
live models. Historical RED-only and RUL runs are not live signal models.
Choosing a live model does not change the model selected globally by Training.

## Demo feed

Choose **Demo feed**, select Testing/Validation machines and press **Start demo
feed**. Each refresh appends the selected number of recorded observations to a
new project-owned demo folder. Only the rows already delivered reach the live
forecast. The demo advances while the screen is open; **Stop demo feed** pauses
delivery, and **Clear demo data** removes only the current demo session folder.
The watched sensor folder is separate. Changing the active data snapshot stops
the old demo. Feed writes are serialized across browser sessions.

Monitor settings and demo data live under ignored `data/live/<project_id>/`;
`PDM_LIVE_ROOT` can override that root. This integration supplies the monitor
screen and local demonstration feed. It does not run a Teams notification worker.

## Verification

`tests/test_live_monitor.py` covers mapped and original snapshot navigation,
received-prefix equality with Results, full history, causal future exclusion,
gap recovery, Calibration refresh, stable machine selection and progressive
isolated demo delivery. Existing Calibration and project UI regressions also
run against the shared predictor.
