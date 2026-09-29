# Independent review of subtask A — 2026-09-29

## Verdict

APPROVED after rechecking source replacement, HSE clock admission, ZIP grouping, archive isolation, and the updated focused tests. Reviewed read-only: `projects.py`, `data/project_import.py`, `data/generic_csv.py`, `data/project_prepare.py`, focused tests and their compatibility with B worker/runtime. The updated project/import/invariant suite passed (123 tests), and Ruff passed for A-owned paths.

## Critical findings and disposition

1. **XJTU ZIP group identity — fixed**: `_owned_adapted` maps the single allowed ZIP's extracted units to `primary`; the new ZIP grouping test passes. Mixed ZIP/CSV input is rejected.
2. **Replacement source/snapshot binding — fixed**: `import_project` persists a staged replacement manifest but keeps the old active source, snapshot and run together. `prepare_project(pid, new_manifest_id)` verifies that staged manifest and file hashes, then atomically swaps the active source and snapshot while clearing selected run. The tampered staged-source rollback test and successful replacement test pass.
3. **HSE invalid clock — fixed**: `_canonicalize_adapted` rejects a unit flagged `nonmonotonic_source_time` before sorting can conceal its acquisition error; a focused test passes. Signal-only omission of dust/flow is acceptable because the saved input allowlist is `['signal']`; malformed bearing fragments lose RMS and create gaps.

## Warnings resolved or pending

- An HSE ZIP is rejected during import with a clear extracted-folder instruction, so the UI should offer folder CSV import for new HSE projects.
- First segment start remains `gap_before=True` for modeling, while user-facing gap counts exclude that artificial initial boundary.

## Review evidence and remaining gate

The checked source has project-scoped metadata roots, safe ID/path validation, a shared launch/archive lock, linked legacy archive+tombstone preserving original files, independently allocated physical units, cross-folder duplicate content detection, signed generic input admission, and snapshot file hashes. Browser import and real training remain separate integration gates owned by D.
