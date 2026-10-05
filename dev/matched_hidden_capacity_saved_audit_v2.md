# Saved auditor v2: bounded revision for independent review

The v2 sibling addresses the two findings in `output/bearings-learned-funnel-20261002/qa/matched_hidden_capacity_saved_auditor_review.md`. The frozen original remains byte-identical. The producer, protocol, ready file, receipts and scientific evidence were not modified or read during this revision.

## Narrow source diff

1. `terminal_records()` reads only terminal JSON and validates terminal status, source/provenance bindings, acceptance flags and the existing fixed-budget complete receipt. `actual()` calls it after the root-token and layout checks, before all pin byte hashes, protected byte hashes, producer inventory traversal/hashing, prediction loads or checkpoint byte reads. Missing/nonterminal status and invalid provenance reject at this boundary. The existing protected checks still surround the independent audit, and the entire producer inventory must remain unchanged.
2. `progression()` retains finite arithmetic, the exact seven frozen criterion keys/definitions, thresholds, memberships and support checks. It copies the summaries and represents missing/None/NaN/infinite numeric evidence as null. Required unsupported gates emit null plus `unsupported_criteria` reasons; overall status is `inconclusive` if any required gate is unsupported, including mixed false/missing outcomes. Finite all-pass and finite failures remain `pass` and `fail`. Markdown renders null criteria as `INCONCLUSIVE`.
3. The existing default synthetic toy now expects a missing required MAE criterion to be null. A separate synthetic-only check script verifies the revised ordering and classifications using invented inputs and mocked terminal JSON/hash functions. It never uses real actual-mode bindings.

Native issued-q authority, unconditional no-entry mass, censoring rules, complete-only whole coverage, known-coordinate arithmetic, macro/micro calculations, frozen persistence skill arithmetic, masks, lost-origin checks, protected inventories and final acceptance boundaries remain unchanged. No Torch/model import, NPZ/checkpoint/raw/frame load, fit, forward, Validation/Test access or producer stop/restart occurred. No quality, application or goal acceptance is established.

## Hashes

- Original `qa/matched_hidden_capacity_saved_audit.py`: `c82bcf4cb78a218b98638b36b1a5589526c060e2fe1cd610f7c9453f5c4b2c9a` (unchanged before and after checks).
- New `qa/matched_hidden_capacity_saved_audit_v2.py`: `24fb6af83e64160fea609ea9ab53a66a74ed11b217b3e3cb39cc3467d7be89a0`.
- New `qa/matched_hidden_capacity_saved_audit_v2_check.py`: `7179bcb2f54da2b91c300476ed18c773be6fe81f41894410dfd0fb6ede25c922`.

## Synthetic validation

Executed from the repository root:

```sh
.venv/bin/python -B output/bearings-learned-funnel-20261002/qa/matched_hidden_capacity_saved_audit_v2.py
.venv/bin/python -B output/bearings-learned-funnel-20261002/qa/matched_hidden_capacity_saved_audit_v2_check.py
```

Both exited zero: 14 default synthetic NumPy checks and 19 focused revision checks passed. Focused coverage includes missing/running terminal status before any hash, invalid terminal provenance before hashing, wrong token before all I/O, finite pass/fail, absent/None/NaN/positive and negative infinite required MAE, unsupported whole/score/baseline/membership evidence, mixed false/missing classification, JSON null serialization, no Torch import, and the unchanged original hash. These checks validate branch behavior and invented arithmetic only. Actual mode remains unavailable without the same explicit root terminal token and separately SHA-pinned root bindings; no actual scientific evidence was audited.
