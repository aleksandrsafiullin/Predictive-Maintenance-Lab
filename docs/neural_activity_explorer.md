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

## Schematic fallback vs MaleCNS anatomy

The labeled Drosophila CNS cartoon (two optic lobes, central brain, ventral nerve cord) is the **fallback**, not the default anatomy view. JS `fitNodesIntoHull` / `polarToCns` run only when `hull_mode === "schematic_cns"`. Fallback is used when:

- the soma table is missing
- the filename allowlist misses (empty `unavailable` table)
- the run is synthetic (`synthetic_fixture`, or `random_rewire` of a synthetic parent)
- soma schema fails (unknown columns, weights feather, `syn-points-*`, `x_pre` table) — UI catches `ValueError`, shows the columns message, sets `anatomy_missing`, and does not crash

That cartoon is **not** a registered FlyEM or VFB template. Meshes are not vendored. Synthetic 8-node graphs still show the synthetic banner and stay schematic even if a local soma file exists.

**`layout.json` from `layout_positions()` spring (2D or 3D) is schematic, not FlyEM anatomy.** A non-zero Z span does not make spring coordinates anatomical. For `real_connectome` (or `random_rewire` of a real parent) a usable soma table **wins over** spring `layout.json`. Graph node `x,y,z` attrs count as anatomical only if they were stored on every node (train does not write those today).

### Anatomy mode (`hull_mode=malecns_anatomy`)

When a usable soma table is present for a real / rewired-real run:

- **hull polyline** — 2D XY convex hull of soma context (median Z), not a second optic-lobe cartoon
- **full soma cloud** — `context_positions` (N_viz after cap), dim, normal blending
- **reservoir glow** — display nodes only; flashes from stored `states` (optional interpolation between provided frames), not `Math.random` or `sin(time)` color

Unmatched reservoir ids stay in `nodes` / `states` and sit at the centroid of matched somas.

### Counts

| Flag | Meaning |
|------|---------|
| `n_model` | ESN width from the **run / provenance / checkpoint** (`real_connectome` 500–2000). Not the live 512-capped length. |
| `n_viz` / `context_n` | Soma context count **after** the display cap (scene 80k; live explorer 20k) |
| `downsampled` / `n_nodes_display` | Live **reservoir** display cap (`LIVE_NODE_CAP=512`). `states` width equals `n_nodes_display`. Never padded to N_viz. |

Do not raise ESN `n_nodes` to ~166k to fill the cloud. Soma join is viz-only.

Missing anatomy on a real / rewired-real run shows:

> Anatomical soma coordinates are missing; showing a labeled schematic, not FlyEM MaleCNS anatomy.

## Neuroglancer is reference only

There is **no Neuroglancer iframe**, CAVE client, or official-scene runtime. The MaleCNS hub / Neuroglancer demo JSON is a **visual reference** for hull + soma cloud + subset glow. Download and allowlist details: [malecns_visualization.md](malecns_visualization.md).

**AppTest does not certify WebGL look.** Pytest greps + payload flags are the merge gate; exercise `http://127.0.0.1:8501` when you care about pixels.

## Animation

Playback runs in the **browser** with `requestAnimationFrame`. Streamlit does not rerun the Python app on each frame. Node flashes come from stored `states` arrays (optional interpolation between consecutive provided frames), not `Math.random` or `sin(time)` color. Camera auto-orbit may use time. A ventral sensor strip is driven by `inputs[frame, feature]`. Fake bloom is a second additive Points layer on **active reservoir** nodes only (no CDN EffectComposer). Soma context uses normal blending.

## Work overlay

The Plotly chart under the 3D view plots the **unit sensor** on one time axis (bearings RMS in minutes; filters Δp in internal seconds) with:

- faded full-history recording
- prefix seen by the model (timestamp ≤ Now)
- highlighted input window
- predicted failure zone from stored `predicted_rul_s` (Now → Now + RUL)

It does **not** plot RUL on the sensor y-axis and does not invent future RMS/Δp. The faded samples to the right of Now are historical. `Show ground truth` only adds an actual-event line; it does not change `predicted_rul_s` or the predicted zone. Caption:

> The faded recording to the right of Now is historical. The reservoir only sees the highlighted window.

## Live training

While a reservoir train job runs, a snapshot of **one train-split window** is written after each gradient epoch and after a ridge solve:

```text
runs/_worker/live_activity.json
runs/_worker/live_activity.npz
```

JSON is metadata only (no large arrays, no `hull_mode` / `context_positions` stamp). GRU/LSTM jobs write `status: not_reservoir` and do not invent spikes. Explorer and Train can show this snapshot with the banner `Live training — train-split window only`. Non-train worker jobs clear the files. The UI scene helper always wins `hull_mode`.

## Trace artifacts

Traces are written only when requested (Build trace, or `pdm evaluate --with-trace`). Paths:

```text
runs/<run_id>/traces/<unit_id>/meta.json
runs/<run_id>/traces/<unit_id>/states.npz
runs/<run_id>/traces/<unit_id>/contributions.npz
runs/<run_id>/traces/<unit_id>/frame_map.json
```

Under the dataset root this is `runs/<dataset_id>/<run_id>/traces/<unit_id>/`. `predict` and `predict_with_trace` share one state-update function. Contribution identity is on **raw** linear readout (before Softplus / Weibull median / `time_scale_s`).

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

> Neural Activity Explorer requires a reservoir run.

It does not invent biological activity or fake reservoir states.

Interrupt / Stop writes worker status **`cancelled`**, never `completed` or `stopped`.
