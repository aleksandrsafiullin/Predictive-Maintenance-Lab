# Frozen Bearings endpoint collector: preparation only

The new collector is `output/bearings-learned-funnel-20261002/review/collect_frozen_bearings_endpoint.py`. No actual invocation, binding file, candidate selection or evaluation was created. Producer41707 remains under root control. Existing source/helper/protocol/receipt files were not modified. The preparation used source text and invented/mocked tests only, including the readiness scope addendum.

## Future root bindings contract

The CLI requires `--actual`, exact token `ROOT_AUTHORIZED_FROZEN_BEARINGS_ENDPOINT_COLLECTION`, an absolute `--bindings` path and its `--bindings-sha256`. The parsed binding file is checked against those bytes and rehashed throughout/before and after collection. Direct `actual()` calls also require this external binding path and SHA. Token rejection precedes all I/O.

Root must create the binding JSON only after directly observing terminal producer41707 with exit0 and independently closing the Train audit. This is a new collector-specific schema, not a claim that such proof artifacts already exist:

- `schema`: `frozen_bearings_endpoint_bindings_v1`.
- `partition`: exactly `Validation` or `Test`; one partition per invocation.
- Absolute paths: `status`, `terminal_receipt`, `root_terminal_observation`, `train_audit_closure`, `train_audit`, `candidate_lock`, `checkpoint`, `config`, `metadata`, `protocol`, `snapshot_fingerprint`, `snapshot_dir`, `output`.
- `project_id`, `snapshot_id`: original protocol IDs; `partition_uids`: exact ordered IDs from the original saved split for the specified partition; `feature_names`: original33 Train names; `scaler_sha256`: canonical JSON SHA256 of the unchanged C3 Train scaler; `producer_root_ready_sha256`: producer's frozen ready-file SHA.
- `pins`: absolute file path to SHA256. Include every explicit input, this collector source, the seven fixed source paths in `SOURCE_SHAS`, every current `src/pdm/**/*.py`, and every snapshot file named in `processed_fingerprint.json.file_hashes`. All pin bytes are checked before runtime imports and repeatedly/after collection. Snapshot file digests must match the fingerprint. The binding file has its separate externally supplied SHA to avoid a self-hash cycle.

The terminal JSON proof requires complete5840 status/receipt, terminal40, original helper/protocol/ready provenance and false quality/application/goal flags. Before any checkpoint/NPZ/source byte hash or runtime model import, the following pinned proof records must pass:

- `root_terminal_observation`: `root_direct_terminal_observed:true`, `producer_session_id:41707`, integer `producer_exit_code:0`, exact `terminal_receipt_sha256` and `status_sha256`.
- `train_audit_closure`: `independent_train_audit_closed:true`, `audit_passed:true`, exact `terminal_receipt_sha256` and `train_audit_sha256`. This closes the engineering audit; it does not imply passing Train scientific progression or final quality gates.
- `candidate_lock`: `root_authorized:true`, `selection_train_only:true`, `heldout_selection:false`, one `arm` (`control` or `treatment`), `epoch:40`, `optimizer_steps:2920`; exact `checkpoint_sha256`, `config_sha256`, `metadata_sha256`, `scaler_sha256`, original `feature_names`, `terminal_receipt_sha256`, `train_audit_closure_sha256`. Its `partition`, `checkpoint`, `config`, `metadata`, `output`, `project_id`, `snapshot_id`, and `snapshot_fingerprint` must equal the root bindings.

Checkpoint and config paths must be the selected arm's exact `matched_hidden_capacity_run/{arm}/epoch_40_state.pt` and `config.json`. Its state SHA must match the producer terminal endpoint receipt and strict-reload declaration. No checkpoint comparison or automatic endpoint choice is implemented. `output` must be a new direct child directory of the study's `review`, without pinned-input overlap, existing content or symlink paths/ancestors. No automatic suffix/retry is implemented.

## Source and collection contract

Actual mode retains original config plus the selected arm's hidden size, original C3 metadata/scaler/33feature names,60s519-bin grid, S256, coverage0.9 and fixed16-origin batches. It uses `load_snapshot(..., feature_partitions=(partition.lower(),))`, the original split, `build_trajectory_frame(..., cap=None)`, one CPU `torch.load(..., weights_only=True)`, `_make_model('gru', ...)` and strict state load. No optimizer is created/restored and no fit, calibration, candidate ranking, Train replay or endpoint comparison is implemented.

Only `x`, `current`, `red_threshold`, `feature_names` enter unchanged production `predict_learned`. Future targets, masks and event annotations never enter predictor inputs. `bundle_for` and `capture_prediction` are AST-extracted verbatim from the pinned base/duration helpers; their training mains and optimizer helpers are not imported or called. Prediction state tensors must remain unchanged across collection. Native q is never renormalized/replaced.

The output contains all constructed origins in original order, native H30/H519 bands, sampled median, unconditional520-class `issued_q`, identity/current/RED/as-of/grid, mask and NaN unknown targets, and original event/no-entry/warning evidence under review-only names. It validates the current builder's observed singleton finite bucket and uninterrupted-prefix/no-entry authority, without dropping partial/unobserved origins. It does not score or choose new gates. The ledger and receipt bind feature-x/raw-history bytes, UID names, checkpoint/scaler/config/source/snapshot provenance and original split.

## Source-contract limitations

- `load_snapshot` verifies the complete snapshot's file hashes, including the shared features Parquet file, and reads complete split/unit/schema/report metadata. Only the declared partition's feature rows are decoded by its Parquet filter. The collector does not change this existing integrity behavior or decode another partition's features.
- Observed event masks are singleton buckets in the current pinned Bearings builder. The collector rejects generic multi-bucket observed masks rather than silently using a first-only score. Coordinate masks may contain later admitted targets after an interruption; event/no-entry authority uses only the first uninterrupted prefix. Future independent saved-array scoring must retain first+last bracket/censoring rules and complete/partial/unobserved support.
- Prior-segment/previous-RED warning exclusions are supplied by the unchanged causal builder; their source is pinned. The collector preserves those flags and verifies eligible origins are at risk. It does not reconstruct sensor chronology as an independent validation.
- Native full-grid H519 bands and H30 bands from the existing capture policy remain distinct. Paths are used transiently for capture and are not persisted; this ledger cannot independently reconstruct raw Monte Carlo energy terms.
- Validation has been reused and Test explored. Only three independent heldout units are available. Collection or later origin counts do not establish nominal calibration, independent confirmatory testing, generalization, quality/application acceptance or goal completion. Existing gates must be applied later by independent saved-array scoring.

## Synthetic checks and hashes

```sh
.venv/bin/python -B output/bearings-learned-funnel-20261002/review/collect_frozen_bearings_endpoint.py
.venv/bin/python -B output/bearings-learned-funnel-20261002/qa/collect_frozen_bearings_endpoint_check.py
```

Both exited zero. Default CLI reports invented checks only;24 focused checks verify root-token/no-I/O rejection, missing/live/failed/incomplete terminal rejection, root exit0/direct observation/Train closure, Train-only candidate lock, checkpoint identity/terminal40, fresh/symlink/overlap output guards, invented singleton/prefix/NaN/identity semantics, fixed16+1 batching, target-label exclusion, native survival-class retention, no Torch/pdm runtime import and all seven original source hashes unchanged. No real actual bindings or scientific evidence were read.

- Collector SHA256: `0b0ddf631360f946667ce86c9fe3bca9e67b81f2163cb315ec5f7267fb56c3de`.
- Synthetic checker SHA256: `869705e972bc1d490c977170d3332869036c917426c6a16085bee460f8c49a24`.

Actual execution remains forbidden pending new explicit root authorization after terminal Train audit closure and independent review of this preparation.
