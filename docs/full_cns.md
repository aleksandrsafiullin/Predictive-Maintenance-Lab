# Full MaleCNS model

The primary Neural Activity Explorer uses every classified neuron from the local MaleCNS v1.0 release. It is a computational reservoir with biological connectivity, not a reconstruction of a living fly's electrophysiology.

## Reproduce

Put the public `connectome-weights-male-cns-v1.0-minconf-0.5.feather` and `body-annotations-male-cns-v1.0-minconf-0.5.feather` in `data/raw/connectome/`. Then:

```sh
.venv/bin/python -c 'from pdm.connectome.morphology import prepare_morphology; prepare_morphology()'
.venv/bin/python scripts/train_brain_forecast.py
```

The first command downloads the explicitly selected public SWC arbors. Training requires the full local tables and has no synthetic fallback. It uses a fresh input projection and readout; no prior checkpoint is accepted. All graph data is streamed into sparse arrays and cached with verified hashes. Old GRU/LSTM and synthetic test infrastructure is independent of this path.

## Population and graph

| Quantity | MaleCNS v1.0 |
|---|---:|
| Classified neurons (`superclass.notna()`) | 166,700 |
| Directed neuron pairs | 25,582,938 |
| Summed synaptic contacts between those neurons | 124,177,617 |
| Plotted soma / to-soma locations | 140,638 |
| Neurons that compute without a point location | 26,062 |
| Reconstructed arbors in the display subset | 123 |

The source weights file has 151,856,684 **segment pairs**, including unclassified fragments. That is not the number of neuron connections. These values should never be interchanged with synapse counts. Counts above are calculated from the source files, not hardcoded into the UI.

## Computation and learning

`W_res[i,j]` represents source `j` to target `i`. Every retained source count is transformed with `log1p`, then globally scaled to spectral radius 0.9 using a verified sparse Perron iteration. Each neuron has its own continuous state under the shared leaky tanh equation. The model does not infer electrophysiological parameters or neurotransmitter signs from connectivity alone.

The regression readout uses mean state per `superclass / class / somaSide` group (98 groups in this release). All 166,700 states compute before pooling; every neuron is included in one group. The trained group coefficients are expanded back into mathematically equivalent per-neuron readout weights for inference and inspection. This regularizes the forecast and avoids a dense neuron-by-neuron regression solve.

Training keeps at most 32 full state frames at a time. Three whole-bearing folds select regularization and the linear/log target transform using training bearings only. Calibration bearings fit the empirical interval; the existing test bearings are labeled an exploratory reused holdout. Forecast quality is measured separately from whether the graph is complete. Biological topology alone does not establish predictive accuracy.

Runtime replay keeps the last two full state vectors, a bounded state-history display and the causal readout history. Appending computes only new measurements; rewind reconstructs the prefix. Matrix and coordinate artifacts are hash checked on reload. The recurrent matrix is saved independently of the much smaller readout checkpoint.

Validate a saved run against the real held-out data using `.venv/bin/python scripts/validate_full_cns.py runs/bearings/<run_id>`. It checks reload integrity, actual state width, cached continuation, rewind, incoming contribution sums and agreement with the saved forecast report.

## Anatomy and rendering

Soma and to-soma positions come directly from curated annotations. Missing positions remain missing. SWC arbors are selected independently of equipment outcomes and use the same 8 nm voxel space. Simplification retains the original vertices at branches and endpoints, plus spaced vertices along chains; no synthetic arbors are generated.

The complete graph computes on the CPU. GPU point buffers display all available point locations. Arbors and the optional 12,000-pair straight-link preview are display subsets with separate counters. Activity is the actual continuous tanh state, not membrane voltage, recorded activity or spikes. No biological 10 ms clock is claimed: the clock follows the equipment measurements.

## Sources

- [Official MaleCNS data and SWC formats](https://male-cns.janelia.org/download/)
- [Google Research announcement](https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/)

MaleCNS data attribution: FlyEM / HHMI Janelia, University of Cambridge / MRC LMB, Google Research. CC-BY 4.0. Source-specific hashes and the selected SWC file list are stored in the local provenance manifests.
