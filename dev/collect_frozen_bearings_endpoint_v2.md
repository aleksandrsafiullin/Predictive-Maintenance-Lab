# Frozen Bearings endpoint collector v2: full closed-run inventory

Preparation only. This new sibling addresses only review finding C1/P2 in `qa/collect_frozen_bearings_endpoint_review.json`, whose SHA256 was verified as `b022139d5a68e36e77e0b7f9df30936e50492be183ed7cf0374cd75b320a98d1`. No real producer inventory, checkpoint, arrays, Train audit closure, heldout features or actual bindings were read or created. Root's live41707 report remains context, not terminal authorization. The original collector, checker and developer report remain byte-identical.

## Narrow diff

- Bindings now require schema `frozen_bearings_endpoint_bindings_v2`, `producer_run`, `producer_inventory_receipt`, and `producer_inventory_sha256` in addition to the unchanged v1 root/endpoint/partition/input fields.
- `closed_inventory()` enforces agreement among a pinned root inventory receipt, the pinned independent Train auditor's `output_inventory`, and the pinned Train closure's `run_inventory`. Maps must have identical keys and SHA256 values, and canonical digest agreement. Both full arms' mandatory producer outputs are required, including every epoch0..40 state/probability file, epoch1..40 optimizer audit, both terminal native evidence/funnel files and global producer receipts. Extra producer artifacts must also be present in the exact agreed map. Both terminal state hashes must agree with the producer endpoint receipt.
- `producer_inventory_guard()` verifies exact file membership, directory membership implied by those files, and absence of symlinks/special filesystem entries. Full mode hashes every producer file. Added/deleted files, added empty directories, linked files/directories and byte mutations cannot pass a full boundary check.
- Full producer bytes and membership are checked at dispatch before runtime model imports/loading and at completion before output creation and again after the output receipt. At each16-origin prediction guard, full producer membership/directory/symlink checks and required selected checkpoint/config/proof/source/snapshot byte pins are checked. Unselected historical producer bytes are rehashed at the full boundaries, not every prediction batch. Optional unselected producer pins in `pins` follow the same timing. A byte-only unselected mutation can survive a batch guard but must reject completion; a selected/input/source mutation is detected by the next guard.
- The resulting receipt records full producer-byte boundary protection and per-batch tree protection separately, without overstating per-batch byte checks.

Root token, complete terminal40/5840, directly observed41707 exit0, independently closed Train audit and single Train-only candidate selection still precede all producer checkpoint/NPZ hashing and runtime model imports. Missing/live/failed terminal proofs reject first. No inference arithmetic, native S256/H519/H30 capture, fixed16-origin batching, frozen scaler/features/config, partition restriction, NaN unknown-target authority or review-only labels changed. No optimizer/training/calibration/selection or new gate choice was introduced.

## Required inventory records

Root must create and pin these only after terminal closure. No such real bindings were created during preparation.

The new bindings fields are:

```text
schema = frozen_bearings_endpoint_bindings_v2
producer_run = absolute exact study/review/matched_hidden_capacity_run
producer_inventory_receipt = absolute root inventory JSON path
producer_inventory_sha256 = canonical JSON SHA256 of full run_inventory map
pins[producer_inventory_receipt] = SHA256 of that complete JSON file
```

The inventory receipt should be saved outside the producer run and contain:

```text
schema = closed_matched_hidden_capacity_inventory_v1
producer_run = same exact absolute producer path
run_inventory = { producer-relative file path: SHA256, ... }  # ALL files/both arms
inventory_sha256 = same canonical map SHA256
status_sha256 = pins[status]
terminal_receipt_sha256 = pins[terminal_receipt]
train_audit_sha256 = pins[train_audit]
```

Canonical map hashing uses JSON with sorted keys, separators `(',', ':')`, UTF-8 and no NaN, exactly `canonical_sha()`. Paths must be normalized producer-relative paths without traversal; digests must be64 lowercase hexadecimal characters. `status` and `terminal_receipt` must be the producer run's exact JSON paths, with their map values equal to the corresponding root pins.

The existing pinned `train_audit` JSON must provide its actual `output_inventory` and copied terminal `status`; the former must equal the root full map and the latter must equal the root-pinned producer status record. The existing pinned `train_audit_closure` must additionally contain:

```text
producer_run = same exact absolute producer path
run_inventory = identical full map
producer_inventory_sha256 = identical canonical map digest
producer_inventory_receipt_sha256 = pins[producer_inventory_receipt]
```

Its original independent-close/audit/terminal provenance fields remain required. The existing root lock still binds the chosen terminal checkpoint/config and partition, not a heldout ranking. The external binding-file SHA and original source/input pin rules remain unchanged.

## Synthetic validation

```sh
.venv/bin/python -B output/bearings-learned-funnel-20261002/review/collect_frozen_bearings_endpoint_v2.py
.venv/bin/python -B output/bearings-learned-funnel-20261002/qa/collect_frozen_bearings_endpoint_v2_check.py
```

Default inert checks passed. The focused checker passed39 checks, preserving the previous root/terminal/selection/output/causal-mask/native-capture checks and exercising the real returned preflight guard against isolated invented temporary trees. Added/deleted/changed unselected files, file/directory symlinks, added empty directories, selected checkpoint mutation, inventory-proof mutation, omitted root/auditor inventories, closure mismatch and omitted treatment inventory reject; unchanged full inventory passes. The tests explicitly permit an unselected byte-only change at a partial batch guard and require its rejection by the full completion guard. Wrong/missing/live terminal inputs reject before producer hashing. Temporary `.pt`/`.npz` filenames contain invented plain text, never scientific checkpoints/arrays. No Torch/pdm runtime import, model, real evidence or actual execution occurred.

## Exact hashes

- New collector `review/collect_frozen_bearings_endpoint_v2.py`: `2d8423d6d9911df8c657bd06520707d41ffb4e26a15a7e2fa03c1ef23992c8b5`.
- New checker `qa/collect_frozen_bearings_endpoint_v2_check.py`: `55bd11b78206bd42f3ca6378b4b6177f28580a0b040bdb0872df6fd0ef3cdc99`.
- Original collector unchanged: `0b0ddf631360f946667ce86c9fe3bca9e67b81f2163cb315ec5f7267fb56c3de`.
- Original checker unchanged: `869705e972bc1d490c977170d3332869036c917426c6a16085bee460f8c49a24`.
- Original developer report unchanged: `eb5dfbc1db34476e048c7f9945bc7db3031b0a75efe9efbb9c7266594410202f`.

All seven fixed original source hashes were also verified unchanged. The prior snapshot-loader caveat and reused Validation/explored Test/three independent heldout-unit limitations remain unchanged. No quality/calibration/application acceptance or goal completion is established. Actual execution remains forbidden until new explicit root authorization after terminal Train audit closure and independent review.
