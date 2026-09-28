# Sensor zones: implementation and evaluation

## Scope and interpretation

Both datasets now expose recurring sensor-condition zones based on measured
signals. A zone describes the current measured condition; it is not a temporary
remaining-life window, an event label, or a maintenance deadline. The zone colors
are also separate from prognostic event episodes.

## Bearing vibration zones

The bearing rule uses the larger of horizontal and vertical RMS against a
bearing-specific baseline formed from its first five measurements. Yellow begins
after five consecutive measurements exceed `max(baseline median + 3 × baseline
SD, 1.25 × baseline median)`. Red applies when measured RMS reaches twice the
baseline median. The signal rule follows recovery, and neither zone is defined
from the recording endpoint or RUL. See [bearing zone definition](health_zones.md)
and its implementation in [health_zones.py](../src/pdm/health_zones.py).

The exported weak-label artifact `bearing_signal_zones_v1_54f8f892b58d`
(`runs/_zones/bearings/label_artifacts/.../manifest.json`) is bound to data
version `20260915T153526Z_04154804`. It contains 9,216 rows from 15 bearings:
7,456 green, 570 yellow, and 1,190 red. Four bearings have no yellow rows under
this rule: `Bearing1_4`, `Bearing2_4`, `Bearing3_3`, and `Bearing3_5`.

Trained run `zones_20260928_200845_2f9991`
(`runs/_zones/bearings/.../metrics.json`) selected epoch 4 of 5 on validation.
Against the sensor-derived weak labels,
the smoothed model had validation unit-balanced accuracy 0.973, balanced
accuracy 0.790, and red recall 0.579; it raised red on two of three validation
bearings. On test it had unit-balanced accuracy 0.969, balanced accuracy 0.956,
and red recall 0.959, raising red on all three test bearings. Median first-red
lead relative to the recorded endpoint was 41.5 minutes on validation and 102
minutes on test. These are agreement and timing summaries on a small laboratory
dataset, not independent health-class accuracy or confirmed failure lead time.
The weak labels are derived from the same vibration signals the GRU is trained
to reproduce; no expert zone annotations are available.

## Filter pressure zones

The filter display uses measured differential pressure and row quality. Below
the provisional 300 Pa warning boundary, valid pressure plus positive finite
flow and finite dust-feed context is green; 300 Pa to below 600 Pa is yellow;
600 Pa or higher is red when the pressure channel is usable. Missing required
context or unusable pressure is gray. The 300 Pa boundary is provisional and
pending expert validation. Green means only “below the provisional pressure
band”; it does not mean healthy. The HSE observed-event convention remains
strictly above 600 Pa, distinct from the display's red boundary at or above
600 Pa. Reference residuals and comparable-context trends are supplementary
evidence and do not change the pressure class. See [filter zone definition](filter_sensor_zones.md)
and [filter_zones.py](../src/pdm/monitoring/filter_zones.py).

The current exported label artifact `filter_sensor_zones_v2_662288ba7101`
(`runs/_zones/filters/label_artifacts/.../manifest.json`) is bound to data
version `20260915T153833Z_4182d92b` and policy
`filter_pressure_warning_300pa_v1`. It includes 78,236 row-admitted measurements
from 99 units: 72,662 green, 5,512 yellow, 5 red, and 57 gray. The source has
78,834 measurement records; 598 were excluded before the admitted feature rows.
The artifact records that event/RUL/future labels were not used for zone
classification.

Fresh validation evaluation `20260928T152551Z_91e713`
(`runs/monitoring_bundles/98c47636891c3ac099dd12ea/evaluations/.../evaluation.json`)
completed as `validation_diagnostics` for bundle `98c47636891c3ac099dd12ea`,
processing 5,763 observations across 10 units.
Its saved sensor-zone colors count 5,595 green, 166 yellow, 1 red, and 1 gray;
these match the validation split of the label artifact. This is a validation
diagnostic, not a test result.

## Existing event-model verification

The existing event-model batch `20260915T154129Z_78a7fa`
(`runs/batches/.../manifest.json`) contains the five bearing models and four
filter models. They were verified from saved runs; they were **not retrained**
for this sensor-zone change. The recorded local output is
`output/sensor-zones-existing-event-matrix-verification.json`.
It covers all nine models, checks forecast parity against saved predictions,
trace/plain parity, deterministic replay, and independence from future rows and
truth. All four saved comparison exports (bearing/filter validation/test) match
their saved evaluations in CSV and JSON.

## Limits

The filter cohort has four observed events in training, one in validation, and
none in its 50 test prefixes. Test RUL labels do not make those prefixes
observed failures. The observed event total is five independent filter units;
repeated rows are not additional events. Filter source `Time` and RUL physical
units remain unverified, so filter timing is reported only in dataset-internal
time. The bearing zone labels are weak signal rules, the red threshold is
provisional, and only three bearings are present in each validation and test
split. Neither dataset supports a production protection claim or calibrated
operator probabilities.

## Local artifacts and reproduction

`runs/` and `output/` are ignored local directories, so the artifact files and
verification JSON above are not included in GitHub merges. Their IDs, counts,
and metrics are recorded here from the local manifests and outputs. Recreate
them from the repository root when the corresponding prepared data, saved
bundle, and prior event-model batch are available:

```sh
PYTHONPATH=src .venv/bin/python -m pdm zones-labels --dataset bearings
PYTHONPATH=src .venv/bin/python -m pdm zones-labels --dataset filters
PYTHONPATH=src .venv/bin/python -m pdm zones-train
PYTHONPATH=src .venv/bin/python -m pdm monitor-evaluate --bundle-id 98c47636891c3ac099dd12ea --split validation
PYTHONPATH=src .venv/bin/python scripts/verify_quality_study.py 20260915T154129Z_78a7fa --scope main --output output/sensor-zones-existing-event-matrix-verification.json
```
