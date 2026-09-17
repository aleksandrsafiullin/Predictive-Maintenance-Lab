# Sensor forecasting

`signal_forecast.py` implements an independent sensor task. Bearings forecast horizontal
RMS (g); filters forecast differential pressure (Pa). No RUL-to-sensor reconstruction,
forced increasing line, terminal zero target or forecast-to-actual splice exists.

Candidates: persistence, causal local trend, and separate histogram quantile regressors
for each horizon and quantile (0.05, 0.5, 0.95). The initial bounded study uses 300/900
physical seconds for bearings and 30/90 dataset-internal units for filters. Longer horizons
remain deferred, not implicitly extrapolated. The scenario holds current observed conditions.

Targets match actual timestamps within saved tolerance 0.01, require exactly one later
measurement in the same continuous segment, and are masked per horizon. Acquisition gaps,
maintenance resets and segment changes prevent crossing. End-of-record labels remain missing.
Inputs are allowlisted raw signals/current conditions with available causal aggregates and
availability flags; future flow/RPM, official RUL and eventual trial length never enter.

Boosting disables internal random validation and early stopping. Training weights equalize
objects. Output quantiles are sorted, then clipped at zero; this is the physical transform in
both runtime and scoring. The band is **pointwise unvalidated quantile regression**, not a
joint trajectory guarantee. A persistence/trend candidate has no invented interval.

Selection uses an original-train equipment fold. All three methods are compared on exactly
its targets. The final selected sensor source has its own run identity, independently of the
event architecture. Negative results and per-unit/horizon errors are saved. Final validation
reports include coverage, MAE, RMSE, pinball losses and band width/coverage where available.

Threshold crossing means the first saved discrete median forecast point satisfying the
saved instantaneous rule. Sustained crossing is unavailable unless duration is established;
pointwise band crossings are not event-time quantiles. No crossing in the displayed horizon
is not a promise of infinite life.

Implementation uses installed scikit-learn 1.9.0. Primary documentation:
[histogram quantile regression](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html).
