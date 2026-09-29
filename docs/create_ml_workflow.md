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

- **GRU / LSTM:** numeric multi-horizon signal heads, configurable history, horizon times, seed, epochs, hidden size and batch size. The retained checkpoint is selected using unit-equal Validation mean absolute error.
- **Quantile boosting:** independent horizon models with pointwise quantiles. The iteration budget is configurable; candidate iteration counts are selected on Validation, then the frozen model is tested.

Scaling and cadence are derived from Training only. Missing future targets beyond recorded history or across gaps are masked. Model artifacts, parameters, preprocessing, snapshot fingerprints and bindings are checked when a saved run is opened.

The existing Fly, Random and Full MaleCNS research engines remain in their existing APIs/CLI. They are not offered for this new numeric signal task because a validated signal head/runtime has not been implemented for them. An old future-red classifier or RUL checkpoint does not unlock this Results screen.

## Read Results

There is one Results view. Select a saved model and a held-out Test unit.

- **Gray:** measurements received up to the Now cursor.
- **Blue:** saved model forecasts at the requested future horizon points.
- **Yellow and red:** warning/critical signal limits in the same physical unit.
- **Red crossing marker:** the first supported forecast point across the red limit. This is discrete horizon resolution, not an exact continuous failure timestamp.

Play supplies observations in order. Pause holds the cursor; Reset returns to the start. Moving the observation slider pauses playback. A new project, model or unit starts a fresh replay. No later measurement enters an earlier forecast. The app waits for enough continuous history and clearly reports unavailable forecasts, no predicted crossing inside the horizon, and measurements that are already red.

Without a saved absolute override, bearing limits use only the unit's initial observed baseline; HSE defaults remain provisional 300/600 Pa signal bands. Results uses saved limits, not an unsaved Data Quality preview. User-entered limits express operating rules; they are not inferred fault diagnoses. A signal crossing does not establish mechanical failure or usable maintenance lead time. Boosting's pointwise quantiles are unvalidated distribution estimates, without a calibrated coverage guarantee; recurrent runs do not invent uncertainty bounds.

## Persistence and deletion

Default storage is `data/projects/`, overrideable with `PDM_PROJECTS_ROOT`. The registry tracks project names and active source/snapshot/run. Each project owns `source/`, `snapshots/` and `runs/`; saved results survive app restarts.

**Projects → Delete project** requires a project-specific confirmation and moves that project's owned directory to `.trash/`, preserving a recovery receipt in the registry. There is no automatic purge or restore button in this version. Linked project deletion archives only its owned metadata/new artifacts and tombstones its registration; existing raw data, processed research data and historical runs stay in place. Deletion is blocked while that project's worker is active.

## Validation boundary

The implementation includes contract, isolation, split, artifact-binding, causal inference, worker and UI regression tests. The [29 September review and corrective verification](create_ml_review_20260929_ru.md) records the original findings and their fixes: 718 tests pass, with desktop/mobile and light/dark browser checks. The original delivery report predates these changes. Short fixture or real-data smoke training verifies the workflow; it is not evidence of field prediction accuracy, calibrated intervals, or production maintenance readiness.
