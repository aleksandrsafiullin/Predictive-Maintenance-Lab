# Matrix profile routing consistency

The matrix now uses the same existing route as normal Training in
`src/pdm/project_training_ui.py`: try
`learned_bearings_training_profile(snapshot, engine)`, then use
`funnel_training_profile(snapshot, engine)` when the learned profile is falsey.
The entire resolved profile is retained without a field allowlist, normalization,
or injected experimental coefficients. Future explicitly accepted advanced fields
returned by the existing profile will therefore survive this route.

This is route consistency only. The normal profile remains scientifically
unaccepted. It does not adopt the experimental nine-coefficient profile, establish
warning usefulness, or close the zero RED-extras gap: the existing normal learned
defaults still leave the additional RED corridor/finite-bound/known-CDF objective
weights at zero. Those decisions need separate scientific acceptance and evidence.

## Dataset selection and frozen contract

The optional repeatable `--dataset` flag accepts exact rebuild manifest keys.
Omitting it means all existing projects. An explicit empty selection, empty key,
duplicate, or unknown key is rejected before any snapshot load. Excluded projects
are skipped before snapshot decoding or active-snapshot inspection. Manifest order
and the existing four-engine order are retained, regardless of selector order.
The selector is carried through prepare-only, freeze, and run.

Each job records the resolved forecast mode, horizon grid, training/path sample
counts, physical-group origin cap, batch sampling, and checkpoint policy:

- Learned: dense cadence-aligned paths, best Validation physical-group-equal joint
  objective checkpoint, patience/min-delta early stopping, learned uncalibrated band.
- Residual: sparse supported grid, fixed final model fit, grouped Train OOF
  residuals, and Validation-only interval calibration.

Test remains frozen evaluation. The contract no longer falsely says that neither
route uses Validation for selection. Exact implementation hashes and snapshot
fingerprints remain part of the contract. Resume requires an identical regenerated
contract; old contracts are refused, never overwritten or refitted. Fresh output
directories must be empty. Worker execution, result verification, timeout behavior,
and in-run source/snapshot guards are unchanged apart from selector forwarding.

## Verification and execution boundary

`tests/test_signal_funnel_matrix_profile_routing.py` extracts named functions from
source AST and injects only standard-library objects and invented dictionary
stubs. It imports no project runtime module and uses no real features, scientific
arrays, model construction, forward calls, fit calls, snapshots, or worker jobs.
Temporary pytest JSON files contain invented contracts only.

Verified command (25 passed):

```text
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest --noconftest -q -p no:cacheprovider tests/test_signal_funnel_matrix_profile_routing.py
```

Checks cover all four learned engine routes and arbitrary advanced fields,
unadmitted-snapshot fallback, all-versus-selected snapshot reads, manifest/engine
ordering, selector validation before loads, mixed policies, CLI forwarding,
exact-resume refusal, changed bindings, empty fresh directories, and AST identity
of the preserved runner body and refusal tail. The existing launch test file was
preserved and its real worker training test was not run. The matrix CLI itself
was neither imported nor executed while study 41707 was live.

## File identity

Only the matrix script was edited; only this note and the source-only test were
created. No `src/pdm` file or protected old file was edited.

| File | SHA-256 |
| --- | --- |
| Original matrix and preserved review backup | `274d108000e58813a2ac5fbc630a8a53c69efa0860fd3148b53557d48e69ac0f` |
| Updated `scripts/run_signal_funnel_matrix.py` | `d3093cb0c2a6ecbcfa1c77468988fd654cd98aeec68247f7d2db26684b1702d6` |
| New `tests/test_signal_funnel_matrix_profile_routing.py` | `c1b7065d9a32e0216f0e6f12ceb53bb72b532b9ee638690e16b5eae8c3fb29f0` |
| Preserved `tests/test_signal_funnel_launch.py` | `219853db97a2dfe0904b87323c0cb9a3cfe322679685b15bc39125b3daba4e29` |

The final note hash is reported in the task handoff rather than embedded in itself.
