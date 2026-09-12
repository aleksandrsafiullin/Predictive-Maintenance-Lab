# Neural Activity Explorer

Streamlit screen that replays **real** reservoir states from a trained run. It is not a live brain viewer.

Required caption (shown on the page):

> Computational activity in a connectome-based reservoir. This is not a biophysical simulation or recorded activity of a living fly brain.

When the selected run uses the synthetic fixture, the page also shows:

> Synthetic test graph — not a biological connectome

## Four modes

The explorer is a fourth Streamlit screen next to Data, Train, and Test & Replay (`app.py` uses an explicit 4-way branch; there is no `else: screen_replay()`).

| Mode | What it shows |
|------|----------------|
| **Overview** | Full trace window for the selected unit |
| **Equipment replay** | Frames up to the current replay step (aligned with Test & Replay) |
| **Inside prediction window** | The history window that produced the current prediction |
| **Alert inspection** | Trace sliced to the stored alert timestamp (prefix ≤ t; stored predicted RUL, not a rescore from future rows) |

## Animation

Playback runs in the **browser** with `requestAnimationFrame`. Streamlit does not rerun the Python app on each frame. Node flashes come from stored `states` arrays, not `Math.random`.

## Trace artifacts

Traces are written only when requested (Build trace, or `pdm evaluate --with-trace`). Paths:

```text
runs/<run_id>/traces/<unit_id>/meta.json
runs/<run_id>/traces/<unit_id>/states.npz
runs/<run_id>/traces/<unit_id>/contributions.npz
runs/<run_id>/traces/<unit_id>/frame_map.json
```

Under the dataset root this is `runs/<dataset_id>/<run_id>/traces/<unit_id>/`. `predict` and `predict_with_trace` share one state-update function. Contribution identity is on **raw** linear readout (before Softplus / Weibull median / `time_scale_s`).

## Layout

Anatomical xyz coordinates are used when present on nodes or in `connectome/layout.json`. **Topological layout is the required fallback** (seeded spring / ring) so the scene is never blank.

## No CDN

Three.js and the Streamlit component bridge are vendored at:

```text
src/pdm/visualization/component/frontend/vendor/
```

(`three.min.js`, `streamlit-component-lib.js`). Frontend HTML/JS must not use `https://` script or module `src`, and must not reference `unpkg`, `cdn.jsdelivr`, `cdnjs`, or `googleapis`.

## Build trace while a worker is busy

If a heavy job is already running, **Build trace** shows the error caption `A heavy job is already running. Please wait.` and does **not** run inline inference or spawn a second worker.

## GRU / LSTM runs

Neural Activity Explorer requires a reservoir run (`fly_connectome_reservoir` or `random_reservoir`). Selecting a GRU/LSTM checkpoint shows:

> Neural Activity Explorer requires a reservoir run

It does not invent biological activity or fake reservoir states.

Interrupt / Stop writes worker status **`cancelled`**, never `completed` or `stopped`.
