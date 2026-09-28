# Bearing health zones (green / yellow / red)

**Model Report → Health zones** shows, minute by minute, whether a bearing is

| Zone | Meaning | Suggested action |
|---|---|---|
| 🟢 green | Normal | Continue operation and monitoring |
| 🟡 yellow | Something is not right — degradation detected | Plan inspection, prepare a replacement |
| 🔴 red | Urgent condition review: the rule crossed its calibrated RMS threshold, or the GRU predicted the retrospective red class | Inspect promptly and follow site procedures for an operating decision |

Each displayed zone uses only measurements up to that moment. Red is a condition warning,
not a 30-minute failure forecast. Laboratory result on the 15 XJTU-SY bearings; not an
equipment-protection system.

## Zone definitions (retrospective reference)

Implemented in `src/pdm/health_zones.py` (`label_unit`).

- **Red:** the last 30 minutes before the recording ends. This is a retrospective
  evaluation label, not a deadline inferred by the live warning. The final recorded
  sample is an experiment-end proxy, not a confirmed industrial failure timestamp.
- **Yellow:** from degradation onset — the first run of 5 consecutive measurements where
  max(horizontal, vertical) RMS stays above both 1.25 × and median + 3σ of the bearing's
  own first 5 minutes (its healthy baseline). A pure 3σ rule was rejected: very steady
  bearings (e.g. Bearing3_1) would turn yellow after a 5 % drift, ~40 hours before recording end.
  A 5-minute baseline is used because Bearing3_5 starts degrading around minute 6.
- **Green:** everything before onset.

## Zone engines

- **Calibrated rule (recommended):** yellow once an onset is confirmed (latched); red once
  RMS ≥ *ratio* × baseline (latched). The ratio (3.25 on the current split) is fitted on
  train bearings only. This threshold is not calibrated to a fixed time horizon.
- **GRU classifier (experimental):** 32-unit GRU over the last 20 measurements. Inputs are
  causal log ratios of all 20 vibration features to the bearing's own baseline, operating
  condition, and trend features (running maximum, 10-minute slope, onset flag, time since
  onset). Train-only scaler, class- and bearing-balanced loss, 5 epochs with the checkpoint
  selected on validation, hysteresis smoothing (escalate after 2, de-escalate after 10
  consistent predictions). Its red class is trained against the retrospective reference,
  but does not provide a calibrated remaining-time estimate.

## How the design was chosen

Leave-bearings-out cross-validation over the 12 train + validation bearings (4 folds; each
holds out one bearing per operating condition). Test bearings were not used for any choice.
Score: bearing-balanced mean of per-zone recall.

| Variant | CV score |
|---|---:|
| Calibrated rule (ratio fitted per fold) | **0.78–0.79** |
| Rule: ratio OR fast 10-minute rise | 0.78 |
| Rule-yellow + logistic-regression red | 0.78 |
| GRU, trend features, 5 epochs | 0.75 (±0.02 across seeds) |
| GRU, base features, 5 epochs | 0.73 |
| GRU trained 20–40 epochs | 0.64–0.70 (overfits) |
| Rule-yellow + gradient-boosted red | 0.73 |

With 12 bearings, no learned model beat the calibrated rule, so the rule is the default engine.

## Results of the shipped run

`pdm zones-train` (seed 42). Score as above; "first red" = minutes before recording end.

| Split | Engine | Score | First red per bearing |
|---|---|---:|---|
| Validation | Calibrated rule | 0.660 | 1_4: 0 · 2_4: 10 · 3_4: 69 |
| Validation | GRU | 0.640 | 1_4: never · 2_4: 7 · 3_4: 68 |
| Test | Calibrated rule | 0.707 | 1_5: 12 · 2_5: 146 · 3_5: 101 |
| Test | GRU | 0.701 | 1_5: 4 · 2_5: 95 · 3_5: 99 |

Neither engine shows red or yellow while the retrospective reference is green in this run.
The warning lead time varies substantially:

- **Abrupt changes near recording end** (Bearing1_4, 2_4, 1_5): vibration barely changes
  until the last minutes, so red is late or absent.
- **Gradual changes** (2_5, 3_4, 3_5): red appears 68–146 minutes before recording end.
  These values do not validate a 30-minute forecast or an automatic replacement decision.

## Commands

```bash
.venv/bin/python -m pdm zones-train            # train + evaluate, ~15 seconds on CPU
.venv/bin/python -m pdm zones-train --red-minutes 60 --seed 7
```

Runs are saved under `runs/_zones/bearings/<run_id>/` (`model.pt`, `meta.json`,
`metrics.json`, `predictions_{validation,test}.csv`).
