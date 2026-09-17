# History 20–60 protocol

Existing checkpoints without new history metadata preserve fixed saved history exactly.
New modes: fixed_20, fixed_40, fixed_60, variable_20_60. The latter refuses before 20 valid
contiguous measurements, uses all available measurements up to 60 and then the latest 60.
Actual duration is the difference of first and last timestamps (20 minute-spaced points
span 19 minutes). Feature context and model memory are reported separately.

`history.py` builds windows; variable training selects lengths in a deterministic endpoint
cycle over 20–60, limited by available segment history. This exposes late as well as early
life at short lengths. Validation/runtime use the longest available permitted window.
Recurrent batches are right-padded with explicit lengths and packed sequences; masked
values do not update the selected hidden state. Sampled reservoir caches gather the last
valid state and direct input. Full CNS stays continuous since reset, with saved warmup;
it is never reinterpreted as a 60-point window.

Training runs through training-v2 with its optimizer/scheduler/RNG/stop rules. Preprocessor,
checkpoint metadata, training identity, cache windows and recipe bind the policy. Changing
history invalidates resume and calibration identity. New multiscale recipes add raw-signal
slopes and availability on 5/20/40/60 plus an unknown-regime feature. The no-age variant
removes age from the selected input columns, without editing old recipes.

Two clocks are required: common-clock-60 accuracy and end-to-end availability from the
start of each history. Missing predictions stay missing. Filters use common-clock-60 for
new study selection NLL. Bearing selection is the saved final-30-minute metric; common
clock diagnostics disclose any cohort difference. Startup loss from long windows stays
in availability and event denominators.

```sh
.venv/bin/python -m pdm train --dataset bearings --arch gru --protocol adaptive --history-mode variable_20_60 --device cpu
.venv/bin/python -m pdm train --dataset filters --arch lstm --protocol adaptive --history-mode fixed_40 --feature-recipe multiscale_trend_v2 --device cpu
```

These are working candidate paths, not claims that longer history improves accuracy.
The implementation report records actual fitted configurations and negative results.

The new multiscale v1 trial had an index-alignment defect and is retained only for exact forensic replay. Use v2 for new training; it assigns arrays positionally and resets feature context at maintenance/segment markers. See the implementation report for quarantined and replacement fit IDs.
