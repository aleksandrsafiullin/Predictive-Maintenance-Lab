# Condition monitoring implementation plan

Baseline: HEAD `3143fabf3a11928b915f24b03bafe89bd10a6e1c`, newer than specification `e026c0d`.
Initial worktree: untracked `.playwright-cli/`, `output/`. Concurrent changes subsequently appeared in
`src/pdm/worker.py` and `tests/test_worker_and_app.py`; preserve them. Git read via
`/Library/Developer/CommandLineTools/usr/bin/git` (system shim requests Xcode license).

| Already implemented | Extend | New |
|---|---|---|
| admission_v1, immutable snapshots, damaged endpoint quarantine | runtime channel quality and unit/event profiles | monitoring contracts and MAT table export validator |
| origin split checks | binding validation for new bundles | train-only provisional regime reference |
| base_v1, degradation_v1; causal prefix features | additional versioned multiscale recipes | signal target masks and quantile forecast |
| training_v2 optimizer/scheduler/RNG resume | fixed 20/40/60 and trained variable windows | packed recurrent batches and history identity |
| common Predictor, forecast_tensors, trace/replay | history resolution, monitoring adapters | deterministic state/episodes/action timing |
| individual/matrix/study worker | condition-study, freeze, monitor-evaluate | bounded fit ledger, separate task metrics |
| Data Quality / Training / Model Report / Compare Models | condition-first report with saved artifacts | monitoring UI and feedback |

Available immutable data: bearings `20260915T153526Z_04154804` (9,216 measurements / 15 objects),
filters `20260915T153833Z_4182d92b` (78,236 / 99). Counts are read from manifests, not contracts.
Training v2 report records 42 prior fits; these will not be overwritten or counted as new fits.
Baseline test command launched before edits: `PATH=/Library/Developer/CommandLineTools/usr/bin:$PATH .venv/bin/python -m pytest tests -q`.
Actual baseline outcome and final commands are recorded in the implementation report.

Implementation order: contracts/quality → history/reference/features → sensor models → policy/bundle/evaluation
→ existing UI/worker integration → bounded real experiments and browser/test evidence.
No test selection, no automatic operational promotion, no compulsory Full CNS fitting.
