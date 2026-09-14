# Neural Activity Explorer

The operational screen contains one scenario: select the prepared model and bearing, then Start, Pause, Next measurement or Reset. There is no Modes selector or trace-building prerequisite. The newest ready anatomy-complete continuous model is selected by default, with a held-out bearing. Archived models remain separate files.

> Computational activity in a connectome-based reservoir. This is not a biophysical simulation or recorded activity of a living fly brain.

Synthetic fixtures retain the banner **Synthetic test graph — not a biological connectome**. They are test fixtures, not the primary experience.

## The computing brain

The real model has 1,000 MaleCNS neurons and actual synapse-derived recurrent connections. Every computing body ID must have a curated anatomical soma coordinate. The operational screen blocks incomplete anatomy instead of silently drawing only a subset or moving missing cells to invented positions. The older September 13 checkpoint has only 116 matches and is therefore unsuitable for this screen.

A deterministic 40,000-cell gray soma cloud supplies the recognizable anatomical reference. It never receives reservoir activity. Cyan/amber computing nodes show signed actual states, including small and zero states. All computing nodes are represented. The legend and counter distinguish them from background anatomy. Fit brain restores the complete anatomical framing. Edges are hidden by default; an internal diagnostic option can draw up to 350 real connections.

Soma positions come from the exact allowlisted body-annotations feather. Spring coordinates are schematic, never anatomical. The payload preserves model body-ID order; state width and anatomical coverage must match it. `n_viz` is a legacy context-count field, not the number of computing neurons. See [MaleCNS data and provenance](malecns_visualization.md).

## Continuous history and failure interval

New models use the shared leaky ESN equation with persistent chronological state. Training and replay use the same state semantics and saved preprocessing. The first 20 measurements in each uninterrupted segment warm up the forecast; the neurons already compute and display during warmup. A gap resets that segment, and rewind reconstructs the exact causal prefix. Old window-reset checkpoints retain their original semantics.

Training gives every physical bearing equal total weight. Whole-bearing cross-validation inside the training split selects the readout regularization and target transform. The neural readout estimates remaining life; a fixed causal filter combines successive implied failure dates. It uses the observations through Now, never the actual endpoint.

An empirical interval is calibrated on separate validation bearings, with error scaling estimated from training out-of-fold predictions. It is not a guaranteed 90% or 95% probability statement: there are only three independent calibration bearings. Interval width may decrease with better evidence or increase on contradictory observations. The interface does not force it to narrow. Read the saved forecast evaluation for coverage, width, early/late error and limitations. Repeated holdout inspection during development must be labeled exploratory.

`interval_profile.json` binds the checkpoint, preprocessing, graph, dataset fingerprint and disjoint train/calibration/test IDs. Replay also verifies current processed data against the saved run before inference.

## Synchronized display

One Streamlit fragment advances the measurement clock. The browser has no independent playback clock in this screen. The current neural state, HUD, sensor chart, forecast interval and observed endpoint all use the same Now.

The sensor panel shows the recorded vibration, the observed prefix and the possible failure-time band. A separate remaining-life panel shows the evolving forecast range, center and actual remaining life. No future vibration is invented. The faded future recording is evaluator context only:

> The faded recording to the right of Now is historical. The reservoir only sees the highlighted window.

Ground-truth visibility does not change inference, interval bounds or the forecast date. Bearing failure labels are the last recorded sample, not independently confirmed industrial failure timestamps. Censored units never receive invented actual remaining life.

**Inspect a computing neuron** shows every normalized sensor input product, recurrent source body ID and weighted previous state, the leak equation, and raw readout contributions. The raw neural forecast and the causally filtered forecast are distinguished. Sensor projection and trained readout are engineering components; recurrent connectivity is derived from the fly connectome.

## Safety and validation

A heavy worker blocks inference with **A heavy job is already running. Please wait.** Changing dataset, model or unit pauses playback. Reset and rewind reproduce the same prefix. GRU/LSTM checkpoints show **Neural Activity Explorer requires a reservoir run.** They do not receive fabricated biological activity.

Three.js and the Streamlit bridge are vendored in `src/pdm/visualization/component/frontend/vendor/`; no CDN, Neuroglancer iframe or external scene runtime is used. The existing trace CLI and live-training diagnostics remain separate from this operational screen.

Run `.venv/bin/python -m pytest tests -q` and `.venv/bin/ruff check src tests scripts/train_brain_forecast.py`. Payload tests do not certify appearance: inspect the real app on desktop and mobile, confirm the canvas drawn/model count, clock agreement, interval toggles and browser console.
