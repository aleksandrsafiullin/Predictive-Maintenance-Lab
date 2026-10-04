# Live monitor, demo feed and Teams alerts

**Live monitor** is the sixth workflow step (after Results). It watches a folder of CSV
files, and on every refresh it shows each machine's current zone and the saved model's
forecast, with the most urgent machine first. Escalations can be posted to a Microsoft
Teams channel.

It reuses the project's saved model and saved yellow/red limits. Its forecast for a
machine is the same direct model output the Results screen shows for the same history
(`tests/test_live_monitor.py` checks this).

## Data folder

Default: `data/live/<project_id>/incoming/` (git-ignored). Change it on the page.

Every `*.csv` file in the folder is read. One row per measurement:

```csv
unit_id,timestamp_s,combined_rms
pump-01,0,0.61
pump-01,60,0.63
```

- `unit_id`: one stable id per physical machine. Files may hold one or many machines.
- `timestamp_s` (seconds), or `timestamp` with date-times such as `2026-10-04T08:00:00Z`.
- The signal column: the project's signal column (`combined_rms` for XJTU-SY bearings,
  your chosen column for Sensor CSV projects), or `signal` / `value`.
- Rows may be appended or files replaced; duplicate (unit, time) rows keep the last value.
- A clock jump of more than 1.5 Training intervals breaks continuous history; the model
  needs `history_length` measurements after the last gap before it forecasts.
- Bearing projects use each machine's first five measurements as its healthy baseline,
  so a machine's file should start from its healthy period. With absolute limits saved on
  Data Quality, this does not matter.

### Connecting SharePoint

The simplest route needs no IT setup: in the SharePoint document library choose
**Sync** (or **Add shortcut to My files**) so OneDrive mirrors it on the computer running
the app, then set **Watched folder** to that local path, for example
`C:\Users\<you>\<Company>\Maintenance data - Bearings`. New files uploaded to SharePoint
appear on the next refresh.

For a shared server, a Power Automate flow ("When a file is created in a folder" →
"Create file") can copy uploads into the watched folder. A direct Microsoft Graph
connection needs an Entra app registration and is the next step for a product.

## Demo feed

The **Demo feed** panel replays recorded machines into the watched folder as if sensors
were sending data: each tick appends the next real measurement of every selected machine
to `sim_<machine>.csv`. Only the clock is accelerated. It defaults to the Test and
Validation machines, which were not used to fit the model.

**Clear demo data** removes only `sim_*.csv` files and resets the alert history. Other
files in the folder are never touched.

From a terminal:

```bash
.venv/bin/python -m pdm live-simulate --project legacy-bearings --units test,validation --tick-s 1
```

## Teams alerts

1. In Teams, open the channel → **⋯** → **Workflows** →
   **Post to a channel when a webhook request is received**, and copy the URL it shows.
2. On Live monitor, open **Teams alerts**, paste the URL, turn alerts on, and press
   **Send test alert**.

The URL is stored only in `data/live/<project_id>/config.json` on that computer, or can
be supplied as the `PDM_TEAMS_WEBHOOK_URL` environment variable instead (the variable
wins). Each card names the machine, the signal value, the change (for example
`green → yellow`) and, for yellow, the forecast time to red.

One alert per escalation: green → yellow, then yellow → red. A machine that returns to
green can alert again later. All alerts, including ones not sent, are listed under
**Recent alerts** and in `alerts.jsonl`.

Alerts are checked whenever the page refreshes. To keep alerting with no browser open:

```bash
.venv/bin/python -m pdm live-watch --project legacy-bearings      # every refresh interval
.venv/bin/python -m pdm live-watch --project legacy-bearings --once
```

Run either the page or `live-watch` for alerting, not both, or an escalation may be
posted twice.

## Presenting

1. Train a model on the project (Training), open Results once, then open Live monitor.
2. Optionally set up Teams alerts and send a test alert to the channel on screen.
3. **Clear demo data**, then **Start demo feed** at 1 measurement/s.
4. Within about a minute the short-lived test bearings turn yellow, then red, and alerts
   arrive. Pick a machine under **Machine detail** to show its history and the forecast.

## Limits

- Zones are signal limits, not confirmed fault diagnoses. Red is a measured value beyond
  the red limit; it does not establish mechanical failure or a maintenance lead time.
- "Red in ~X" is the first direct forecast point beyond the red limit. It has the
  resolution of the saved horizons and no calibrated uncertainty. With the default
  bearing model it can be early or late by hours on long-lived bearings; check Results'
  held-out error before trusting it.
- Some bearings fail abruptly: vibration barely changes until the last minutes, so
  yellow can last only seconds in the demo.
- The page reads the whole folder on each refresh. That is fine for thousands of rows
  per machine; a product would keep an incremental store.
