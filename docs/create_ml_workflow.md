# Project workflow

The primary Streamlit app follows **Projects → Import data → Data Quality → Training → Results**. A project is an independent workspace, not a fixed dataset name. Existing XJTU-SY Bearings and HSE Filters are registered as linked projects when their local artifacts exist.

## Create and import

Create a named project using **Sensor CSV**, **XJTU-SY bearings**, or **HSE filters**. Choose a folder in the browser, or enter a folder path on the computer running the app. Server paths avoid browser upload limits for large sources. Imported files are copied into the project's owned source area; the original folder is not modified.

Generic CSV requires these columns:

```csv
unit_id,timestamp_s,vibration
pump-01,0,0.31
pump-01,60,0.34
pump-01,120,0.36
```

Choose `vibration` as the signal column and provide a readable name and its physical unit. Set yellow/red limits on **Data Quality after import**. A first Sensor CSV import has no zone rule: measurements are shown as **Not zoned** until limits are edited or saved. Signed numeric signals are supported. Each physical unit needs its own stable `unit_id`; files may contain several units or fragments of one unit. The first version trains on the selected signal history only. Extra outcome, RUL or future-label columns never become model inputs.

Time must be finite seconds without duplicate timestamps within a unit. The generic importer sorts each unit's records by time. Invalid signal values are rejected and break continuous history. Inferred large time gaps also break history. Models cannot train or forecast across a gap. Preparation fails with an actionable message if the source cannot form a valid snapshot.

XJTU-SY imports use the documented raw vibration layout and compute combined max-axis RMS in g. HSE imports use extracted CSV histories and differential pressure in Pa; source `Time` is seconds. HSE's official author Test histories remain held out. Generic CSV is the entry point for other sensor formats after mapping their identifier, clock and selected signal to the schema above.

## Training, Validation and Testing data

**Import data** has three cards: **Training Data**, **Validation Data**, and **Testing Data**. The Training folder initially supplies a pool of whole physical histories. Validation and Testing independently select **Split from training** or **Separate folder**. Each supplied source can use a browser-selected folder or a server folder path.

Before import the cards show **No data yet**. After preparation they show the active snapshot's Units, Gaps and Admitted rows. **View** opens that set's Data Quality tab. These counts describe saved data, not the files currently selected for a replacement import. **Split settings** controls the weights and random seed (defaults: 70 / 15 / 15 and 42); it is disabled when both holdouts use separate folders.

| Validation | Testing | Allocation from the primary pool |
|---|---|---|
| Automatic | Automatic | Desired weights 70:15:15 for Train:Validation:Test |
| Separate folder | Automatic | Train:Test weights 70:15; supplied Validation stays intact |
| Automatic | Separate folder | Train:Validation weights 70:15; supplied Testing stays intact |
| Separate folder | Separate folder | Primary pool stays Training |

Weights are configurable and are normalized over the automatic groups. Actual integer unit counts are shown after import; small datasets cannot exactly match every percentage. Each set must contain at least one usable unit. Rows/windows of one unit never go into different sets. Duplicate unit identities or matching source content in separate groups are rejected, including renamed duplicate files.

For HSE, official `Test_Data_CSV.csv` units in the Training source folder are assigned directly to Testing; the remaining primary histories are divided between the automatic Train/Validation groups. If that file is absent, automatic Testing is drawn from the primary pool too. Alternatively, choose **Separate folder** for Testing and supply its official `Test_Data_CSV.csv` there. Official Test histories retain their protected status regardless of source folder and cannot enter Training or Validation. The selected Testing folder must itself contain admitted histories; official Test units in the Training source do not make an empty or unsupported Testing folder valid.

**Data Quality** has three visible tabs. Each shows unit and row counts, interior gaps, time/signal ranges, rejected rows when present, an equipment selector, a signal chart and a readable measurement table. With a valid rule, the chart and table show Green/Yellow/Red/Not zoned classifications and the tab reports zone counts. These are rule-derived signal zones, not failure diagnoses or the model's training labels. Train fits preprocessing and weights; Validation selects the model; Testing is evaluated after selection. Repeatedly choosing a model based on its Test result would still bias a real study; the app cannot prevent that human decision.

A replacement import becomes active only after preparation succeeds. A failed or stopped replacement leaves the prior active source, snapshot and saved model usable. Previous immutable artifacts remain on disk.

### Move whole units after import

In a Data Quality tab, expand **Move units**, choose **Units to move** and **Move to**, review the resulting counts, then press **Move selected units**. The move is blocked if it would empty a set, if a background job is active, or if the page refers to an outdated snapshot. Official HSE Test units cannot leave Testing. Linked legacy projects retain their published split; create an owned project to use a different split.

A move publishes a new snapshot with a parent reference and move history. Source files and measurement values stay unchanged. Saved limits follow the new snapshot; unsaved previews do not. The selected model is cleared, so **train again for the new split**. Earlier snapshots and models remain on disk, but Results lists only models bound to the active snapshot. Re-importing builds a fresh split from the source settings rather than replaying manual moves. Moving units after inspecting Test results can bias subsequent evaluation.

### Edit yellow and red limits

The panel beside the signal chart applies to all units and all three sets in the current snapshot. On narrow screens it appears below the chart at full width. Choose **Above** for a rising warning signal (`yellow < red`) or **Below** for a falling signal (`yellow > red`). Finite numeric limits use the signal's physical unit.

- A valid edit immediately previews zones on Data Quality in the current session.
- **Save** persists the rule and makes Results use it for limit lines and predicted red entry. **Cancel** restores the last saved rule, or the original import rule if no override was saved.
- Invalid edits are not saved; the chart retains the last valid rule. During a background job, preview remains available but Save is disabled.
- Saving limits does not change the snapshot ID, model weights, saved forecast values or error metrics. It writes `zone_limits.json` beside the immutable snapshot files, outside their fingerprint. Training continues to forecast numeric signal values.

First imports of HSE use provisional 300/600 Pa limits. XJTU-SY uses its initial-baseline rule; opening Data Quality does not replace that rule with the editor's initial 1/2 g values. Editing a value or direction previews an absolute rule, and saving commits that override. On re-import, saved limits are reused for the same signal; changing the Sensor CSV signal column or unit clears the old rule.

## Train one signal model

Choose one engine and its parameters, then **Train model**. Heavy work runs in a background process with project-specific status and a Stop control. One heavy job runs at a time.

- **GRU / LSTM:** recurrent encoder and two learned corridor boundaries; configure history, span, seed, epochs, hidden size and batch size.
- **Quantile boosting:** Train-only quantile tree features followed by a learned bounded corridor readout; configure tree iterations and readout epochs.
- **Fly brain · Full MaleCNS:** all classified neurons and original directed connections, with fixed graph dynamics and a learned bounded corridor readout. Epochs train the readout. Full graph computation can be substantially slower. [Source and computational boundaries](full_cns_signal.md).

The current Training page has one task, **Trend corridor**. Choose GRU, LSTM,
Quantile boosting or Full MaleCNS, then set the history and forecast span. Each
model learns two boundaries at ±10% of its predicted level, expanding up to ±15%
(20–30% total width). Saved first-contract models retain their original ±5–7.5% bounds.
First RED entry follows from their crossings; its upper time bound stays open
when the second boundary does not cross. The percentage is a width constraint,
not a confidence level. Results reports actual containment and flags failed
models. See [the current corridor contract and observed quality](trend_corridor_20261005_ru.md).

Scaling and cadence are derived from Training only. Missing future targets beyond recorded history or across gaps are masked. Model artifacts, parameters, preprocessing, snapshot fingerprints and bindings are checked when a saved run is opened.

The sampled Fly and Random research engines remain in their existing APIs/CLI. Full MaleCNS now has a separate numeric signal adapter for this workflow. An old future-red classifier or RUL checkpoint does not unlock this Results screen; train a signal readout on the current project snapshot.

## Historical saved signal runs

The behavior and starting settings below describe older point/quantile models.
Their predictions remain available with a previous-objective notice. New
corridor runs show the two learned boundaries and derived RED window, with
no separate center-line prediction drawn on the chart.

### Read Results

There is one Results view. Select a saved model and a held-out Test unit.

- **Gray:** measurements received up to the Now cursor.
- **Blue:** direct output heads saved with the model, connected to the last received measurement at Now. The line between heads is a visual connection; only the marked times are evaluated predictions.
- **Show future actual measurements:** off by default. When enabled, a lighter gray line shows the recorded Test suffix after Now for comparison, with a marker at its first recorded RED sample when the zone rule is available. These measurements never enter inference; toggling the overlay preserves the forecast and pauses playback. A retrospective note identifies units on which a warning one third of the mean Train duration before RED is impossible even from the first sample.
- **Yellow and red:** warning/critical signal limits in the same physical unit.
- **Red crossing marker:** the first supported forecast point across the red limit. This is discrete horizon resolution, not an exact continuous failure timestamp.

Play supplies observations in order. Pause holds the cursor; Reset returns to the start. Moving the observation slider pauses playback. A new project, model or unit starts a fresh replay. No later measurement enters an earlier forecast. The app waits for enough continuous history and clearly reports unavailable forecasts, no predicted crossing inside the horizon, and measurements that are already red.

Results uses only the saved direct horizons. A model trained for 1, 2 and 3 minutes cannot forecast a later time without retraining; Results says so for these legacy bearing runs. Training now proposes direct outputs through at least the mean observed Train-unit duration, rounded up to the Training cadence, when both Train and Validation contain known targets. If either split has no target that far ahead, the interface shows the coverage failure instead of claiming a validated forecast. A stable model may never predict red within its trained range. The marker reports the first evaluated forecast point beyond the limit, its timestamp and time remaining from Now; it cannot determine an exact crossing between points.

The editable starting settings are GRU: 24 history samples, 64 hidden units, batch 64, 80 epochs, learning rate 0.0007; LSTM: 32 history samples with the same width, batch, epochs and learning rate; Quantile boosting: 24 history samples and up to 120 iterations; Full MaleCNS: 20 history samples and validation-selected ridge regularization. New GRU/LSTM runs default to learning the change relative to the latest observed signal; the Training switch can disable this residual head. History is shortened if a Train or Validation unit has too little continuous data. Bearings start with candidate horizon multiples of the Training interval `1, 2, 3, 5, 10, 15, 20, 30, 45, 60, 75, 90, 120, 150, 180`; HSE filters start with `1, 2, 3, 5, 10, 20, 30, 60, 100, 150, 200, 300`; generic sensor data starts with `1, 2, 3, 5, 10, 15, 20, 30`. Up to six geometric steps extend each grid to the rounded mean Train-unit duration, staying within the model's 24-head limit. Only horizons with known targets in both Train and Validation are suggested. The Training coverage table shows independent units and targets for each horizon; very long horizons supported by one Validation unit remain exploratory.

These are starting configurations, not globally optimal parameters. On the current bearings split, Validation covers two units through 90 minutes and only one unit from 120 minutes onward; long-range estimates therefore remain exploratory. Validation chooses recurrent checkpoints, boosting iterations or MaleCNS ridge strength. Training shows model MAE against the last-value baseline by horizon, and Results shows both errors on held-out Test. A model that fails to beat the baseline should not be treated as a reliable warning.

Full MaleCNS replay runs in a session-local background worker. The chart immediately shows measurements at the selected cursor, then adds the saved direct forecast when it finishes. Play waits for the current forecast to finish and be displayed before advancing; Pause and the future-actual overlay remain responsive during calculation. Moving the cursor or changing the model or unit cancels obsolete work. Leaving Results or disconnecting the session also cancels it. Only one calculation runs per session, and results for an old cursor cannot replace the active chart.

Without a saved absolute override, bearing limits use only the unit's initial observed baseline; HSE defaults remain provisional 300/600 Pa signal bands. Results uses saved limits, not an unsaved Data Quality preview. User-entered limits express operating rules; they are not inferred fault diagnoses. A signal crossing does not establish mechanical failure or usable maintenance lead time. Boosting's pointwise quantiles are unvalidated distribution estimates, without a calibrated coverage guarantee; recurrent runs do not invent uncertainty bounds.

A RED-entry time window is a separate model output, not a band around forecast signal values. Results draws it only when a saved model explicitly provides finite earliest/latest event times. Current signal runs do not provide that output, so Results says the window is unavailable instead of constructing one from future Test measurements. With the future overlay enabled, Results can compare a model-issued window to the first recorded RED sample. The [warning-quality plan](warning_model_quality_plan_20260930_ru.md) defines the proposed one-third lead criterion, its feasibility limits on the current data, and the event-level coverage and width checks needed before such a window can be trusted.

## Persistence and deletion

Default storage is `data/projects/`, overrideable with `PDM_PROJECTS_ROOT`. The registry tracks project names and active source/snapshot/run. Each project owns `source/`, `snapshots/` and `runs/`; saved results survive app restarts.

**Projects → Delete project** requires a project-specific confirmation and moves that project's owned directory to `.trash/`, preserving a recovery receipt in the registry. There is no automatic purge or restore button in this version. Linked project deletion archives only its owned metadata/new artifacts and tombstones its registration; existing raw data, processed research data and historical runs stay in place. Deletion is blocked while that project's worker is active.

## Validation boundary

The implementation includes contract, isolation, split, artifact-binding, causal inference, worker and UI regression tests. The [29 September review and corrective verification](create_ml_review_20260929_ru.md) records the original findings and their fixes: 718 tests pass, with desktop/mobile and light/dark browser checks. The original delivery report predates these changes. Short fixture or real-data smoke training verifies the workflow; it is not evidence of field prediction accuracy, calibrated intervals, or production maintenance readiness.
