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
the smoothed model had validation unit-balanced accuracy 0.973, **unit-balanced
balanced accuracy** 0.790, and red recall 0.579; it raised red on two of three validation
bearings. On test it had unit-balanced accuracy 0.969, unit-balanced balanced accuracy 0.956,
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

The current exported label artifact `filter_sensor_zones_v2_876a032e8d53`
(`runs/_zones/filters/label_artifacts/.../manifest.json`) is bound to data
version `20260928T170138Z_aa9b4ad0` and policy
`filter_pressure_warning_300pa_v1`. It includes 78,236 row-admitted measurements
from 99 units: 72,662 green, 5,512 yellow, 5 red, and 57 gray. The source has
78,834 measurement records; 598 were excluded before the admitted feature rows.
The artifact records that event/RUL/future labels were not used for zone
classification.

The validation split has 5,595 green, 166 yellow, 1 red, and 1 gray rows.
The standalone **Health zones** screen reads the current prepared snapshot and
this exact versioned label artifact. It checks version, fingerprint, policy,
split, count, and label hash before replay. It does not require an event model
or a monitoring bundle.

## Retired RUL training artifacts

The prior event-model runs, batches, comparisons, training and condition studies,
and monitoring bundles were removed from the active local `runs/` directory on
28 September 2026. A recoverable copy is in
`~/.Trash/Predictive-Maintenance-Lab-old-RUL-2026-09-28/`, with
`archive_manifest.json`. These artifacts estimated remaining time or depended on
the old event models. They are not current red-zone training results. The
bearing sensor-zone labels and GRU, and the refreshed filter sensor-zone labels,
remain active. Old published study reports are historical records only.

## Limits

The filter cohort has four observed events in training, one in validation, and
none in its 50 test prefixes. Test RUL labels do not make those prefixes
observed failures. The observed event total is five independent filter units;
repeated rows are not additional events. HSE Figure 6 labels source `Time / s`;
RUL uses the corresponding duration scale by inference from the CSV, since its
unit is not separately restated in the source schema. The bearing zone labels
are weak signal rules, the red threshold is
provisional, and only three bearings are present in each validation and test
split. Neither dataset supports a production protection claim or calibrated
operator probabilities.

## Local artifacts and reproduction

`runs/` and `data/processed/` are ignored local directories, so these artifacts
are not included in GitHub merges. Their IDs, counts, and metrics are recorded
from local manifests. Recreate current labels from the repository root after
preparing the source data:

```sh
.venv/bin/python -m pdm prepare --dataset filters
.venv/bin/python -m pdm zones-labels --dataset filters
.venv/bin/python -m pdm zones-labels --dataset bearings
.venv/bin/python -m pdm zones-train
```

See the [benchmark review](sensor_zone_benchmark_review.md) for threshold
sensitivity and the interpretation of 0.790/0.956.
