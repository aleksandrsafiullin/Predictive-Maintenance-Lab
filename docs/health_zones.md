# Bearing health zones (green / yellow / red)

**Model Report → Health zones** groups measured bearing vibration into three
signal states. The zones describe the current sensor condition; they do not
represent time remaining to an event.

| Zone | Sensor definition | Suggested action |
|---|---|---|
| 🟢 green | No persistent rise above the bearing's initial vibration baseline | Continue normal operation and monitoring |
| 🟡 yellow | Max-axis RMS exceeds `max(baseline median + 3 × baseline SD, 1.25 × baseline median)` for 5 consecutive measurements | Degradation signal detected; plan inspection |
| 🔴 red | Max-axis RMS reaches `red_ratio` × the bearing's initial baseline median (default 2.0) | Urgent condition review; follow site procedures |

The baseline is the median of the first five measurements. Yellow is confirmed
at the fifth consecutive high measurement; the label begins at that confirmation
measurement. Red can be reached directly from green if the measured RMS crosses
its higher threshold. Zones follow the signal back down as vibration falls below
their thresholds. Runtime rule predictions and reference labels use current and
past measurements only.

## What the zones mean

Implemented in `src/pdm/health_zones.py` (`label_unit`, `rule_baseline`). The
signal bands are **weak reference labels derived from vibration**, not
independent expert diagnoses or confirmed failure classes. The 2.0 red ratio is
a provisional signal threshold in configuration (`red_ratio`), chosen as a
simple separation from the 1.25 yellow ratio and requiring domain validation.
It is not fitted to time-to-record-end labels. The green/yellow and yellow/red
transitions therefore have separate, inspectable sensor thresholds.

The experimental GRU learns to reproduce these same sensor-derived labels from
past vibration features. Its probabilities are uncalibrated. Agreement with the
weak labels measures consistency with the stated signal rules, not independent
health-class accuracy. The 15 run-to-failure XJTU-SY bearings are too few to
establish industrial operating limits or a protection function.

## Separate prognosis and evaluation timing

RUL in this dataset is measured to the final recorded sample, which is an
experiment endpoint proxy and not a confirmed industrial failure timestamp.
First-red timing summaries describe the elapsed gap from a red signal state to
that recorded endpoint. They do not assess whether a signal state was early or
late, and they do not assign zones, select the red threshold, or train the
classifier. A red zone is a present condition warning.

## Versioned label artifact

Export the sensor-derived reference table for every bearing with:

```bash
.venv/bin/python -m pdm zones-labels
```

The command writes `labels.csv` and `manifest.json` under
`runs/_zones/bearings/label_artifacts/<artifact_id>/`. The table contains unit,
split, timestamp, measured max-axis RMS, baseline RMS, numeric zone, and zone
name. It contains no RUL or endpoint column. The manifest records the processed
dataset version and fingerprint, label-policy thresholds, label-file SHA-256,
and weak-label provenance. Its artifact ID changes when the dataset fingerprint
or zone policy changes.

For the current prepared snapshot, the reference labels total 7,456 green,
570 yellow, and 1,190 red measurements across 15 bearings. Four bearings
(`Bearing1_4`, `Bearing2_4`, `Bearing3_3`, `Bearing3_5`) have no yellow
measurements under this policy. This is a class-coverage limitation in the
sensor-derived labels, not evidence that those bearings lack gradual degradation.

## Commands

```bash
.venv/bin/python -m pdm zones-train
```

Runs are saved under `runs/_zones/bearings/<run_id>/` (`model.pt`, `meta.json`,
`metrics.json`, `predictions_{validation,test}.csv`).
