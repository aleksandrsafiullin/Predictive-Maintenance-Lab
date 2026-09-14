# Neural Activity Explorer

The operational screen contains one scenario: select the prepared model and bearing, then Start, Pause, Next measurement or Reset. There is no Modes selector or trace-building prerequisite. The newest ready complete-classified-CNS model is selected by default, with a held-out bearing. Old 1,000-neuron training results were removed; other model families remain separate experiments.

> Computational activity in a connectome-based reservoir. This is not a biophysical simulation or recorded activity of a living fly brain.

Synthetic fixtures retain the banner **Synthetic test graph — not a biological connectome**. They are test fixtures, not the primary experience.

## The computing brain

The primary model is now the complete **166,700 classified MaleCNS v1.0 neurons** (`superclass.notna()`), with **25,582,938 directed neuron pairs** representing **124,177,617 synapses**. Every source connection between these bodies is retained, including weight-one and self connections. Unclassified segments are excluded by the documented population rule. There is no BFS, node cap or synapse-strength threshold. The previous four trained 1,000-neuron real-connectome runs were deleted at the user's request; the new readout is trained from scratch.

The recurrent operator is stored as CSR, in `W_res[target, source]` orientation. SciPy performs the CPU sparse products; the shared leaky tanh equation preserves chronological state. There is no dense 166,700-square allocation. Synapse counts undergo `log1p` and one global spectral scaling. These dynamics do **not** model neurotransmitter signs, membrane voltages or biological spikes. Sensor inputs and the trained forecast are engineering additions.

All neurons compute, including the **26,062** without a curated point location. **139,662** cells have a soma location and **976** additional cells have a to-soma point: **140,638** plotted locations in total. Missing cells are not placed at invented coordinates. Their states still participate in recurrence and the forecast. The browser shows this coverage explicitly.

The morphology layer contains **123 actual reconstructed neuron arbors**, selected deterministically across annotated classes from Janelia's public SWC files. It preserves branch points, endpoints and original source vertices while simplifying unbranched chains. This is a declared morphology display subset, independent of the full computing population. Brain / Whole CNS buttons control camera framing. Activity / Cell classes controls coloring. The optional connection sample draws at most 12,000 real directed pairs as straight schematic links, explicitly separate from anatomical arbors. It never changes the recurrent graph.

The state-history panel shows actual continuous values for 120 deterministic neuron IDs and up to 120 observed measurements. These are not generated spikes; pausing adds no history. Coordinates, morphology and states share body-ID order and one anatomical transform. Source URLs, hashes, counts and license are recorded with artifacts. See [full model implementation and limitations](full_cns.md).

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
