# Fly connectome validation notes

Which tests lock which constraints. CI uses the **synthetic fixture** (not a biological connectome) with explicit tiny `n_nodes` (8–16). MaleCNS is not required in CI. `--smoke` is not a quality claim.

Ridge target space is **normalized RUL** (`y = target_rul_s / time_scale_s`). Contribution identity is on **pre-display raw** (`W_x @ x + W_u @ u + b`) before Softplus / Weibull median / `time_scale_s`. GRU `compat` ignores reservoir YAML defaults (C2).

## Numbered acceptance tests (1–14)

| # | Test | File | Constraint |
|---|------|------|------------|
| 1 | `test_graph_orientation` | `tests/test_reservoir.py` | `W_res[i,j] = edge j→i`; one-edge graph, no silent transpose |
| 2 | `test_state_update_hand_calculation` | `tests/test_reservoir.py` | Leaky tanh step matches a numpy hand calculation |
| 3 | `test_n_nodes_mismatch_vs_weights_npz_raises` | `tests/test_reservoir.py` | Artifact / checkpoint size mismatch raises. **Not** synthetic oversize (that clamps) |
| 4 | `test_dataset_checkpoint_isolation` | `tests/test_reservoir.py` | Bearings reservoir checkpoint is incompatible with filters / GRU / random / other `graph_hash` |
| 5 | `test_seed_reproducibility` | `tests/test_reservoir.py` | Same seed → same `W_res` and `W_in` |
| 6 | `test_split_preprocess_isolation` | `tests/test_reservoir.py` | Train-only scaler; mutating test rows does not change it; `W_in` unchanged across same-seed retrain |
| 7 | `test_no_future_frames` | `tests/test_reservoir.py` | Rows after t do not change prediction / states at t |
| 8 | `test_predict_trace_parity` | `tests/test_reservoir.py` | `predict` and `predict_with_trace` return the same display RUL and raw |
| 9 | `test_window_reset` | `tests/test_reservoir.py` | `x=zeros` at the start of each window (not carried from the previous window) |
| 10 | `test_edge_drive_previous_state` | `tests/test_reservoir.py` | Within a window, `x[t]` uses `x[t-1]` from the same window |
| 11 | `test_contribution_sum` | `tests/test_reservoir.py` | intercept + sum(input) + sum(neuron) == raw within `1e-5` |
| 12 | `test_raw_vs_display_postprocess` | `tests/test_reservoir.py` | `raw` is before Softplus / ReLU display and `time_scale_s` |
| 13 | `test_filters_censoring_not_rul_zero` | `tests/test_reservoir.py` | `event=0` is not treated as failure / RUL=0; Weibull NLL differs from the counterfactual |
| 14 | `test_trace_artifact_reload` | `tests/test_reservoir.py` | Save + reload `states.npz` / contributions; states and live prediction match |
| 14b | `test_synthetic_fixture_label` | `tests/test_reservoir.py` | Fixture has `is_synthetic=True` and label `Synthetic test graph — not a biological connectome` |

Clamp vs raise:

| Test | File | Constraint |
|------|------|------------|
| `test_synthetic_n_nodes_clamps_not_raises` | `tests/test_reservoir.py` | Request 1000 against the fixture → `fixture.N`, no exception, not `real_connectome` |

## Named extras

| Test | File | Constraint |
|------|------|------------|
| `test_stop_flag_sets_cancelled_not_completed` | `tests/test_worker_and_app.py` | `stop.flag` → status `cancelled`, never `completed` / `stopped` |
| `test_gru_checkpoint_compat_ignores_reservoir_yaml_defaults` | `tests/test_spec_invariants.py` | GRU `compat` blobs stay compatible when live YAML has reservoir keys (C2); those keys are not written onto GRU blobs |
| `test_filters_ridge_raises_before_targets` | `tests/test_reservoir.py` | `readout=ridge` on filters raises before processed data or targets are touched |
| `test_ridge_uses_forward_states_kernel` | `tests/test_reservoir.py` | Ridge feature collection calls the shared `forward_states` kernel |
| `test_ridge_no_backward` | `tests/test_reservoir.py` | Ridge path does not call `Tensor.backward` |
| `test_ridge_bearings_no_double_softplus` | `tests/test_reservoir.py` | Bearings `forward()` is ReLU of raw (no extra Softplus); ridge residual ≤ `1e-3` |
| `test_load_trained_model_missing_weights_npz_raises` | `tests/test_reservoir.py` | Missing `weights.npz` raises; no seed rebuild |
| `test_predict_and_trace_share_update_function` | `tests/test_reservoir.py` | `predict` and `predict_with_trace` share one update function object |
| `test_random_reservoir_parent_graph_hash` | `tests/test_reservoir.py` | Random reservoir records `graph_mode=random_rewire` and `parent_graph_hash` |
| `test_frozen_reservoir_weights_not_updated` | `tests/test_reservoir.py` | Optimizer does not update frozen `W_in` / `W_res` / `b_res` |
| `test_no_cdn_in_frontend` | `tests/test_neural_explorer.py` | No `https://` script/module src; no unpkg / jsDelivr / cdnjs / googleapis, including the component bridge |
| `test_screen_switch_by_label` | `tests/test_neural_explorer.py` | AppTest selects Screen by **label**, not radio index |
| `test_worker_busy_shows_error_not_inline` | `tests/test_neural_explorer.py` | Build trace while worker busy → error caption, no inline inference |
| `test_gru_run_does_not_show_fake_biological_activity` | `tests/test_neural_explorer.py` | GRU run shows “Neural Activity Explorer requires a reservoir run” |
| `test_explorer_caption_present` | `tests/test_neural_explorer.py` | Required computational-activity caption + synthetic disclaimer |
| `test_synthetic_banner_present` | `tests/test_neural_explorer.py` | Synthetic banner when `is_synthetic=True` |
| `test_explorer_screen_present_in_app` | `tests/test_neural_explorer.py` | Explicit 4-way Screen radio |

Existing GRU/LSTM leakage, Weibull censoring, and AppTest suites in `tests/test_spec_invariants.py` and `tests/test_worker_and_app.py` remain the product gate.

## Spaces (do not mix)

- **Ridge / Smooth L1 target:** normalized RUL `y = target_rul_s / time_scale_s`.
- **Contribution identity:** pre-display **raw** linear readout, tolerance `1e-5`.
- **Display RUL:** bearings `max(0, raw) * time_scale_s`; filters Weibull median of Softplus `(λ, k)`.

## What CI did **not** validate

- Full MaleCNS download from GCS
- 30-epoch reservoir accuracy on XJTU-SY / HSE
- Browser visual verification of the WebGL explorer (AppTest is the merge gate)

The synthetic fixture is **not a biological connectome**. Do not treat smoke metrics or explorer frames as model quality.
