# Future-red sensor forecast experiment

## Scope

This laboratory experiment asks whether measurements available at time *t* can
forecast a first entry into the saved sensor rule's red zone within a fixed
horizon. It is distinct from the current-zone classifier discussed in the
[sensor-zone benchmark review](sensor_zone_benchmark_review.md). The saved red
labels are weak, rule-generated signal labels, not verified fault diagnoses.
Results below are research evidence only; they do not support production use.

Source datasets, units, sampling descriptions and split definitions are listed
in the [README](../README.md#predictive-maintenance-lab) and
[data units and endpoints](data_units_and_endpoints.md). Bearing zones derive
from the causal vibration bands described in [health zones](health_zones.md).
Filter red is the provisional display rule at differential pressure ≥600 Pa;
the HSE observed event convention in the source is strictly >600 Pa.

## Target and evaluation protocol

For each saved sensor-zone row at timestamp *t*, the target is 1 when the first
future red label occurs in (*t*, *t* + *H*], where *H* is 1,800 seconds for
bearings and 20 seconds for filters. The origin row is excluded, and a red
label at the inclusive horizon endpoint counts. The comparator is the saved
zone name (`red`); the bearing exporter also verifies `true_zone == 2`.

A target is 0 only when the whole future horizon is observed, every row needed
to establish the outcome has usable zone quality, and no excessive sampling
gap breaks the interval. Rows after a unit's first red are outside the risk set.
Incomplete right-censored horizons, unusable signal rows, and excessive gaps
remain masked/unknown; they are not relabeled as negative. When red occurs in
the horizon, observation and quality checks run through that first red row.
The filters target rule uses pressure plus usable row quality; its saved-zone
policy also records flow and dust-feed context. Targets are frozen from the
current hash-verified zone-label artifact and tied to the prepared data
fingerprint.

Splits are by unit: bearings use the existing 9 train / 3 validation / 3 test
split, and filters use the saved author-test holdout plus train/validation unit
split. Validation selects checkpoints and the classification threshold; test
is evaluated after model freeze. The matrix uses seed 42, 20 history rows,
30-epoch maximum, batch 64, hidden size 64, one layer, patience 5, and 1,000
Fly nodes. It compares GRU, LSTM, Fly connectome reservoir, and degree-matched
Random reservoir. Full MaleCNS is a separate bearing model run.

## Sensor-only matrix results

Values below are from the completed manifests in
`runs/_future_red/matrix/sensor_only_20260928/`. Brier and average precision
(AP) are computed over known test rows. Event recall, mean lead and alert
burden are unit/event-oriented manifest metrics; lead is seconds before first
red, and burden is the fraction of at-risk rows warned.

| Dataset / model | Known test rows (positive) | Brier ↓ | AP | Events detected | Mean lead (s) | Alert burden |
|---|---:|---:|---:|---:|---:|---:|
| Bearings GRU | 225 (67) | 0.151 | 0.728 | 3 / 3 | 1,340 | 100.0% |
| Bearings LSTM | 225 (67) | 0.151 | 0.604 | 2 / 3 | 1,800 | 50.2% |
| Bearings Fly | 225 (67) | 0.378 | 0.631 | 3 / 3 | 1,340 | 87.6% |
| Bearings Random | 225 (67) | 0.387 | 0.627 | 3 / 3 | 1,340 | 87.1% |
| Filters GRU | 29,483 (0) | 0.017 | undefined | not estimable | — | — |
| Filters LSTM | 29,483 (0) | 0.015 | undefined | not estimable | — | — |
| Filters Fly | 29,483 (0) | 0.191 | undefined | not estimable | — | — |
| Filters Random | 29,483 (0) | 0.190 | undefined | not estimable | — | — |

The test baseline results are shared across each dataset's four architectures:

| Dataset / baseline | Brier | AP | Event result |
|---|---:|---:|---|
| Bearings always no entry | 0.298 | 0.298 | 0 / 3; 0% burden |
| Bearings current-red persistence | 0.298 | 0.298 | 0 / 3; 0% burden |
| Bearings trend-to-red | 0.240 | 0.428 | 3 / 3; 420 s mean lead; 10.2% burden |
| Filters always no entry | 0.000 | undefined | Not estimable: no test event |
| Filters current-red persistence | 0.000 | undefined | Not estimable: no test event |
| Filters trend-to-red | 0.026 | undefined | Not estimable: no test event |

On validation, the four filter models each detected the sole observed event
(1/1, about 19.9–20.0 s lead); the trend baseline also detected it (19.8 s).
The four filter model AP values were 1.000, while the trend baseline AP was
0.602. These are one-event results, and validation was used for model and
threshold selection. Bearing validation event detections were GRU 3/3, LSTM
1/3, Fly 3/3 and Random 3/3; the trend baseline was 1/3. See each run
manifest for its full split metrics and per-unit details.

The baselines predict no entry, persist the current-red state, or fit a line
through the last five observations since the latest gap and alert if it
crosses the saved red boundary within the horizon. They are deterministic.

All three observed bearing test event units were detected by GRU, Fly and
Random, but those models warned on roughly 87–100% of at-risk test rows. LSTM
detected two of the three. The trend baseline detected all three with far
fewer warnings, although its row recall was 26.9%. Three test events are too
few to rank these approaches reliably. The filter test split contains zero
observed future-red events and 9,931 masked rows (9,918 censored, 13 quality).
Its 29,483 known negatives therefore cannot estimate event recall, lead time
or future-red discrimination; a low row Brier score is not evidence of useful
forecasting.

## Full MaleCNS bearing run

`runs/_future_red/bearings/models/full_cns_seed42/manifest.json` reports Full
MaleCNS seed 42 against the same three baseline definitions. On validation,
the model detected 3/3 event units at 1,500 s mean lead and 5.3% alert burden
(Brier 0.047; AP 0.607); trend-to-red detected 1/3 at 1,260 s and 1.2%
burden (Brier 0.050; AP 0.179), while both no-entry and current-red
persistence detected 0/3 (Brier 0.057; AP 0.057). On test, MaleCNS detected
3/3 at 1,240 s but warned on 70.7% of at-risk rows (Brier 0.255; AP 0.783).
The test baselines were: always-no-entry and current-red persistence each
0/3, Brier/AP 0.298/0.298; trend-to-red 3/3, 420 s, 10.2% burden,
Brier/AP 0.240/0.428. High event recall alongside this warning burden and
only three test events is not evidence of deployable performance.

## Interpretation and next work

The future targets inherit the saved zone policies. Bearing red is based on a
causal baseline from the first five max-axis RMS measurements, with red at
≥2× its median; the full policy is recorded in the source label manifest.
Filter red is tied to the provisional laboratory pressure rule. These are
weak signal labels, so the experiment measures anticipation of a rule event,
not independently verified degradation or failure. The older **0.790
validation / 0.956 test** values refer to unit-balanced balanced accuracy for
the current-zone GRU against its rule-generated labels. They mean agreement
with that rule (which scores 1.000 by construction), not future-red forecast
quality; see the [benchmark review](sensor_zone_benchmark_review.md).

Next work should prioritize more independently observed bearing event units,
filter test units with observed endpoints, and expert-validated zone rules.
Then evaluate warning lead, missed events and alert burden by unit across
repeated unit-level splits. Keep censored horizons unknown and do not tune on
test data. The results here do not justify adding a production forecast or
maintenance action.

## Reproduction

Prepare both datasets and export their current sensor-zone labels first. Then
run the eight-model sensor-only matrix:

```bash
.venv/bin/python scripts/run_future_red_matrix.py \
  --datasets bearings,filters \
  --architectures gru,lstm,fly,random \
  --seed 42 --history-length 20 --epochs 30 --batch-size 64 \
  --hidden-size 64 --num-layers 1 --patience 5 --fly-nodes 1000 \
  --output-dir runs/_future_red/matrix/sensor_only_20260928_reproduction
```

The runner writes a run manifest, a summary, per-model manifests, predictions,
per-unit metrics and matrix `metrics.csv`. It freezes the target artifacts and
configuration in the run manifest and refuses to overwrite an existing output
directory without `--resume`.
