# Data units, endpoints and full-history import

Admitted versions retain original membership and origin mapping. HSE physical identities
are namespaced by author train/test source because Data_No is reused for distinct trials.
Existing `assert_split_coverage` rejects origin duplication/cross-split leakage. Full-history
imports need additional content matching; renaming a unit cannot establish independence.

Bearing `last_recorded_sample` remains `bearing_experiment_end_proxy_v1`, not a confirmed
industrial failure. Rejected endpoints remain invalid and are never moved to the final
retained row. Fragment index × 60 is the acquisition clock; each completed 1.28-second
fragment contains 32,768 samples at 25.6 kHz. RMS is acceleration in g, never mm/s.

Filter `filter_lab_600pa_v1` is a laboratory differential-pressure event, retaining the
existing first-measurement-strictly-above-limit convention. The independent monitoring hard limit uses >=600 Pa; the distinction is explicit. The source PDF describes crossing
600 Pa, flow control, the 0–2500 Pa sensor range (p3), and dust feed in mm3/s (pp2/7).
Figure 6 (p5) labels its horizontal axis **Time / s**, and the p7 description gives
Sampling in Hz. Filter CSV Time is therefore seconds; its observed 0.1 increments
match a 10 Hz acquisition interval. The processing conversion is now 1.0, and the
default sampling interval is 0.1 s. RUL is carried on the corresponding remaining
duration scale. The source schema names RUL without repeating its unit, so this is
interpreted from its relationship to Time. Original values remain in the processed
tables for audit.

Source checked locally: `data/raw/filters/Preventive to Predictive Maintenance dataset.pdf`,
version 1.4, 16 June 2021. Figure 6's axis title is the unit evidence for Time;
the p7 schema does not independently spell out RUL's unit. Extracted inspection
evidence is in `output/condition-monitoring/hse-source-text.txt`; this is not a
substitute for MATLAB metadata.
Every study/bundle saves `unit_verification.json` with unresolved issues.

Official filter test RUL stays evaluator-only. It is not an observed failure within a
prefix, and cannot establish detection recall there. Planned preventive replacements
can cause informative censoring; alarm outcomes remain unknown without suitable follow-up.
Probability outputs use the existing remaining-duration Weibull head. Unknown horizon
labels are never negatives. IPCW/reliability certification is not claimed on these few
independent events; see [survival evaluation assumptions](https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html).

## Export pathway

```matlab
addpath('scripts');
export_filter_tables('data/raw/filters/Train_Data_Uncensored.mat', 'data/raw/filters/export');
```

```sh
.venv/bin/python -m pdm validate-filter-export data/raw/filters/export
```

The MATLAB helper inspects actual table/timetable objects and exports original columns,
metadata and numeric roundtrip evidence. Unsupported containers remain explicit. The
Python validator checks schema, path containment and row/column identity. The
`filters_full_history_v1` admission mode stays disabled until actual MATLAB export,
origin/prefix content matching, endpoints and Time/RUL metadata are independently verified.
Missing dependencies return `blocked_external_dependency`, not synthetic measurements.
