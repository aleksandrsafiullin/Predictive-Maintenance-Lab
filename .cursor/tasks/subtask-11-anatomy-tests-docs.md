# Subtask 11: Anatomy explorer tests and docs

## Goal

Close remaining tests and docs for the MaleCNS-style explorer: N_model vs N_viz, allowlist soma download (CC-BY, human-only), schematic fallback, **layout.json spring is schematic**, Neuroglancer as reference not runtime, CDN/`Math.random` greps green. Do not treat `--smoke` as quality. **AppTest does not certify look.**

## Context

Phase K. 07–10 implement loader, payload, WebGL, and app wiring. Docs still treat the schematic hull as the whole story. Tests must **inject a tmp soma dir** and cover syn-points by name + `x_pre` schema.

Do **not** edit `tests/test_spec_invariants.py` unless you accidentally touched splits/losses (you should not).

## Acceptance Criteria

- [ ] `tests/test_neural_explorer.py` covers (fill gaps from 07–10; grep-able names). **Every soma-file test monkeypatches `default_soma_dir` (or equivalent) to `tmp_path`.** Never assume workspace `data/raw/connectome/` is empty.
  - synthetic banner + explorer disclaimer exact strings
  - no CDN / no `https://` script src / no `Math.random` in `main.js`
  - JS: `fitNodesIntoHull` / `polarToCns` only when `hull_mode === "schematic_cns"`
  - allowlist miss → empty + `unavailable`, no raise
  - unknown columns on **explicit** path raise (unit); UI catch → columns caption + schematic + `anatomy_missing`, `not at.exception`
  - **explicit** `MALEMCNS_FILENAME` / weights-like table raises (unit); UI catch as above
  - **`syn-points-*` filename:** default search skips by name; explicit path raises
  - **`x_pre`/`y_pre`/`z_pre` table:** explicit path ValueError (not soma xyz)
  - payload `hull_mode=malecns_anatomy` + **non-empty** `context_positions`; `n_viz != n_model`; `states` width == `n_nodes_display`; `n_model` from provenance (not 512-capped length)
  - downsample deterministic + **numeric body-id sort**
  - **3D spring** layout is non-anatomical; soma wins
  - `anatomy_missing` caption on mocked real_connectome without soma
  - AppTest: `random_rewire` + **synthetic parent** stays synthetic banner / schematic (not FlyEM anatomy)
  - `EXPLORER_MODES` unchanged; `Build trace` / `Load demo scenario` present
  - worker busy Build trace error; overlay `show_gt` zone unchanged; GRU no fake activity
- [ ] `tests/test_reservoir.py` green; no new ESN width=166k tests.
- [ ] `docs/neural_activity_explorer.md`:
  - schematic is **fallback** (missing soma, allowlist miss, synthetic, schema error)
  - **`layout.json` from `layout_positions()` spring (2D or 3D) is schematic, not FlyEM anatomy**
  - anatomy mode: hull + full soma cloud + reservoir glow; flashes from `states`
  - `n_model` = ESN width from run; `n_viz` = context after cap; `downsampled` / `n_nodes_display` are the live reservoir cap
  - no Neuroglancer iframe; official JSON is visual reference only
  - overlay / live / Build-trace-busy / no CDN remain true
  - AppTest does not certify WebGL look
- [ ] `docs/malecns_visualization.md` (create):
  - Hub + Neuroglancer demo URLs as **reference**
  - CC-BY
  - Allowlist: `body-annotations-male-cns-v1.0-minconf-0.5.feather` first; no glob; skip weights + `syn-points-*` by name
  - Weights feather has no xyz; syn-points xyz are synapses, not somas
  - CI never requires 1 GB feather or 166k soma file
  - Do not raise ESN `n_nodes` to 166k
  - Soma provenance `viz_soma_xyz` / `unavailable` is **not** written over run `provenance.json`
- [ ] `docs/fly_connectome.md`: short “Soma xyz (viz only)” + pointer. Do not change orientation / BFS / 500–2000 text.
- [ ] README: at most one pointer; GCS is not a required setup step.
- [ ] Full suite + ruff green.

## Implementation Notes

**Files to modify**

- `tests/test_neural_explorer.py`
- `docs/neural_activity_explorer.md`
- `docs/fly_connectome.md`
- `docs/malecns_visualization.md` (create)
- `README.md` only if a single pointer is missing

**Do not**

- Enable `filters_full_history`
- Add CAVE/Neuroglancer dependencies
- Commit soma/feather binaries
- Rewrite 01–06 task files
- Claim quality from `--smoke` or AppTest screenshots

**Doc URLs (reference only)**

- https://neuroglancer-demo.appspot.com/#!gs://flyem-male-cns/v1.0/male-cns-v1.0.json
- https://male-cns.janelia.org/explore/
- https://male-cns.janelia.org/download/
- GCS prefix `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/`
- neuPrint https://neuprint.janelia.org/ (human, not CI)

## Dependencies

Subtasks 07–10.

## Verification

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check src tests
```

If doctor gained a soma flag:

```bash
.venv/bin/python -m pdm doctor
```

Browser on `:8501` when tools exist; AppTest is not a look certificate.
