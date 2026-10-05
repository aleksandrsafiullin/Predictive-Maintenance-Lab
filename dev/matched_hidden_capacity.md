# Fresh matched hidden capacity producer

The new helper `output/bearings-learned-funnel-20261002/review/matched_hidden_capacity.py` implements the frozen Train-only fresh hidden64x1 versus hidden128x1 study. Its SHA256 is `e65ad2f9818ac44fe6131831106a09853c1a54ca2e97af498f57e08b14ac248f`. Protocol SHA256 is `949a5f79a0b03f498655d6e3caadce84231e8831c4d775f9ebfdb1ed5c58820c`.

Only these two new files were created for development. Existing source, UI, tests, frozen protocol and scientific artifacts were not modified. Source and JSON/Markdown contracts were read; no NPZ, trained checkpoint, raw data or frame was accessed, and no project model was imported, constructed, forwarded or trained. No Validation/Test access occurred.

## Dispatch boundary

Importing the helper defines stdlib functions/constants only. Root must independently review this exact helper SHA and pass synthetic tests, then create a ready JSON with `authorized:true`, exact `helper_sha256` and `protocol_sha256`, both exact gates (`independent_code_review_passed`, `synthetic_continuation_tests_passed`), complete `immutable_files` repository-relative hash map, and `protected_trees` repository-relative paths.

The map must bind the helper/protocol, all twelve pinned protocol references, the comparator frozen skill reference, every `src/pdm/**/*.py`, and every file in each protected tree. All earlier closed/failed `*_run` trees in this study's review and QA folders must be covered by protected trees. Root can include additional protected trees/files. The ready JSON itself is captured by SHA. Before scientific imports/IO the helper verifies full byte maps, exact protected file inventories, source/protocol/ready hashes, symlink-free input/output ancestors and no output/immutable/protected overlap. The output must be the absent, non-symlink exact directory `review/matched_hidden_capacity_run`. Guards repeat for each batch/native capture; full maps/inventories repeat at epoch boundaries, saves, terminal reload and native capture. Directory inventories are preserved after initial gate verification. Scientific module bytecode writes are disabled after authorization.

Root execution, after those separate gates, uses `PYTHONPATH=src` and the project's Python runtime:

```sh
python output/bearings-learned-funnel-20261002/review/matched_hidden_capacity.py \
  --protocol output/bearings-learned-funnel-20261002/matched_hidden_capacity_protocol.json \
  --root-ready <root-created-ready-json> \
  --output output/bearings-learned-funnel-20261002/review/matched_hidden_capacity_run
```

This command was not run during development. A missing root-ready receipt, missing protection or either absent QA gate blocks scientific execution.

## Fixed experiment and initialization

Each arm starts from epoch0 and takes exactly73 original physical_group batches in each of epochs1..40:2920 AdamW updates per arm,5840 total. Only hidden_size changes (64 to128); parameter counts must be833458/1687218. Original rank6, depth1, history8, dropout.25, clip5, optimizer, all nine C3 weights3/.4/5/1/2/.25/5/2/2, S128 training, S256 native inference, masks and population weights remain fixed. There is no best selection, early stopping, budget extension, checkpoint transplant or old trained-state deserialization.

Each arm uses production `_make_model('gru', arm_config, 33, None)` after seed20261002, then zeros event-head weights while preserving compatible production biases. Training Torch stream is independently reset to20261002 after construction; the independent NumPy sampler also starts at20261002. Different widths imply different initialization tensors and dropout geometry. Exact initial full eventq parity is required because the zero head makes predictions bias-driven; initial raw objective metrics need not match.

Sampler proposal/index probability hashes and NumPy start/end hashes must match at every epoch. Objective MC seed remains `seed + epoch*10000 + int(idx[0])`; fixed evaluation remains `seed + start`. Torch start/end hashes are recorded, but shape-dependent ending streams need not match. The reused correction changes only the original q sum assertion tolerance, without new objective math, q normalization or replacement.

Fresh AdamW correctly has empty state at step0. `optimizer_steps` explicitly admits that case and always verifies full optimizer/model membership. After actual updates, every model parameter must have state and the expected step. `restore_arm` deep-copies saved CPU state to avoid AdamW tensor aliases and supports exact fresh or trained same-arm reload. Terminal40 restores that arm's saved state, optimizer and Torch/NumPy streams, then requires full fixedq and all nine raw metrics/total exact parity. Evaluation and native inference use the existing private RNG utility plus bitwise model/optimizer guards.

## Saved contract

Root dispatch/progress/status, weight and ordered identity verification, initialization and terminal receipts are atomic JSON. Progress increments immediately after each successful actual optimizer.step; failures retain truthful completed counts, arm/epoch/phase, exception and protected-integrity status. Failed or partial output is not resumable under this dispatch.

Each arm writes config and learning curve; epochs0..40 state.pt and fixed probabilities.npz; epochs1..40 optimizer audit with ordered idx/pi hash, NumPy/Torch stream hashes,73 clip norms/factors/postnorms/draws and MC seeds, training/evaluation durations and a labeled remaining-training estimate. Epoch0 elapsed time is its fixed evaluation duration. Native inference occurs at terminal40 only.

Terminal NPZ matches the prior clip terminal array contract: H30/H519 lower/upper, sampled median, native `issued_q`, separate `fixed_objective_q`, original target mask and known-target NaNs, physical/unit identity/asof/current/threshold, warning/event/no-entry flags, frozen465 cohort with primary288/supplementary215/overlap38, and horizon grid. Native-versus-fixed q is recorded only as maximum absolute batch-arithmetic difference; there is no equality gate, renormalization or substitution. All4613 origins and9 physical units remain available, including failures. The unchanged native evaluator produces the funnel JSON; independent saved-array scientific auditing remains required.

## Verification and limitations

Development checks passed: AST parse, stdlib-only inert runpy import with absence of NumPy/Torch/pdm modules, exact frozen JSON protocol/config/reference validation, invented protocol-drift rejection, and fake-parameter fresh-empty/full-optimizer-membership checks (including missing-state rejection). Scientific run, terminal reload on real models and native inference were not performed. Independent code review and synthetic tests must complete before root authorizes a producer run. Internal paired Train progress cannot establish calibration, generalization, application readiness or final funnel quality. Every receipt keeps quality_accepted, application_release and goal_complete false.
