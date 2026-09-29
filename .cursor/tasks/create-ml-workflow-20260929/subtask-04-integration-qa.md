# Subtask 04 (D): integration, compatibility and final evidence

## Goal
Join A–C into a complete project workflow, resolve contract mismatches, preserve established research behavior, and document verified limits.

## Owner files
`README.md`, `docs/create_ml_workflow.md`, scoped cross-slice tests, and integration fixes only after owners finish. Preserve all preexisting dirty work, including `docs/frontend_create_ml_plan.md` and unrelated `output/` content. Do not edit historic `.cursor/tasks/*` outside this new directory.

## Acceptance criteria
- [x] Fresh workspace through restart: create two projects, import generic CSV, verify independent snapshots/runs, archive one with receipt, delete a linked project by archiving its owned metadata/signal-run root and tombstoning the external binding while preserving original legacy files, and verify active/queued job blocks own-project deletion until matching job terminal and process exit. Linked legacy project loads normalized prepared data as a tiny immutable project snapshot without copying the multi-GB raw tree; old research runs never unlock the signal Results view.
- [x] End-to-end automatic, Validation-only manual, Testing-only manual, and both-manual folder paths pass. Check realized counts against renormalized desired weights, whole-unit disjointness, duplicate physical/content rejection, minimum counts, rollback, train-only preprocessing and unknown/missing target masks. Confirm scoped new XJTU/HSE source adapters still import, HSE author test remains Test-only, and existing raw artifacts remain accessible.
- [x] Train GRU and LSTM on the signed/gapped generic sensor fixture through UI/worker, reopen run after restart, and check numerical prefix forecast parity and negative-value preservation. Train boosting if eligible. Test run binding and artifact mismatch errors. Include a unit with no predicted red crossing and a unit where first supported discrete forecast point crosses the saved instantaneous red limit. No classifier-derived signal curve appears.
- [x] One report view only. Inspect desktop and narrow browser views, Play/stop/restart, project switching, visual unit/threshold labels and browser console. Data Quality is readable without technical dump. Old research CLI/APIs and their regression tests remain working; no need to expose those views in the new navigation.
- [x] Run `.venv/bin/python -m pytest tests -q` and `.venv/bin/python -m ruff check src tests` (plus scripts if edited); record exact pass/fail counts. Baseline before this task was 534 collected tests, all passing, with Ruff passing. Report any native real-data or model-quality gates not executed honestly. Update README and new workflow doc with CSV schema, folder semantics, project/archive storage, model capability limits, thresholds, forecast horizon and scientific caveats.

## Review gate
Code reviewer checks project isolation, path/delete safety, leakage, causal replay, saved run identity, and plain-language UI claims against actual artifacts. Fix Critical/Warnings and rerun impacted checks. Parent integrator performs the final full-suite and browser acceptance.

## Completion evidence

Completed and independently reviewed on 2026-09-29. See `docs/create_ml_implementation_report_ru.md`, scoped review files, and `output/orchestration-20260929/` evidence. Final suite: 581 tests passed. D used browser GRU on real prepared observations, three actual worker engines on the signed/gapped fixture, contract tests for error/causality cases, and desktop/narrow visual checks. Full raw-data retraining and field model-quality certification were not part of this acceptance.
