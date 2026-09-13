# Subtask 8: Anatomy scene builder (N_model vs N_viz payload)

## Goal

Build the explorer JSON payload that layers **full soma context** (N_viz after cap) under the **reservoir subset** on true xyz when a usable soma table exists, and labeled schematic otherwise. `flags.n_model` is ESN width from the **run/provenance**, not `len(nodes)` after the live 512 cap. Freeze payload keys. Do not raise ESN width. Do not write run provenance.

## Context

Phase H. Subtask 07 loads soma xyz. Today `load_scene_from_run` / `build_explorer_payload` only know reservoir nodes; `flags.hull_mode` defaults to `schematic_cns`. `layout.json` is written by `layout_positions()` spring (`dim=2` today; `dim=3` is still spring). **Any spring is non-anatomical**, including 3D spring with a Z span. Real / rewired-real soma table **wins over spring**. Prefer `src/pdm/visualization/scene.py` called from `explorer.py`.

## Acceptance Criteria

- [ ] New function `build_anatomy_scene(...)` (name flexible) consumed by `build_explorer_payload`. Inputs: run dir or loaded reservoir scene, optional `SomaTable`, graph_mode / is_synthetic, **`n_model` from run/provenance**, context cap default **80_000**.
- [ ] **`flags.n_model`:** integer ESN width from provenance / checkpoint / run row (`real_connectome` 500–2000). **Not** `len(nodes)` after `subset_scene` / `LIVE_NODE_CAP`. If provenance is missing, fall back to artifact `graph.json` node count **before** display downsample — still not the 512-capped list.
- [ ] **`flags.n_nodes_display`:** `len(nodes)` / `states` width after reservoir display downsample. **`flags.n_viz` / `context_n`:** context length after cap. **`flags.downsampled`:** existing live reservoir-activity flag only. These four must stay distinct. **Never** expand `states` to N_viz.
- [ ] **Anatomical vs spring on raw `layout.json`:** inspect the file as written (and graph node attrs if present). A position set is anatomical only if it did **not** come from `layout_positions()` spring. Treat as non-anatomical if: missing z-from-soma provenance, or positions match spring (including **3D spring**). Do **not** use “non-zero Z span ⇒ anatomical.” Helper `is_anatomical_positions` must fail a **3D spring** fixture, not only `z=0` planar.
- [ ] **Merge order for reservoir `positions`:**
  1. Raw layout / graph attrs **only if anatomical** (true soma/template xyz already stored)
  2. else, if real or `random_rewire` of a real parent **and** soma table non-empty: **soma join wins** (including over spring layout.json)
  3. else topological / `fallback_positions`
- [ ] **Context cloud:** all soma ids, not the BFS subset. Empty when no soma table or synthetic run.
- [ ] **Normalize** context + reservoir + `hull_polyline` into one shared bbox (center / scale to ~unit box). Same transform on all three. Optional `flags.anatomy_scale`.
- [ ] **Synthetic** (`is_synthetic` or `graph_mode=synthetic_fixture`): always `hull_mode="schematic_cns"`, empty context, never join FlyEM somas onto `"0"`…`"N"`.
- [ ] **`real_connectome` or `random_rewire` of a real parent without soma:** `hull_mode="schematic_cns"`, `anatomy_missing=True`, empty `context_positions`. No FlyEM label.
- [ ] **With soma + real / rewired-real:** `hull_mode="malecns_anatomy"`, join on `str(body_id)==node_id`, context filled. Unmatched reservoir ids: centroid of matched somas (deterministic; no RNG). `n_unmatched_reservoir` in flags. Do not drop them from `nodes`/`states`.
- [ ] Downsample context if `len > CONTEXT_CAP`: **numeric body-id sort** (`int(id)` when the whole string is an integer, else lexicographic), then even stride / `np.linspace`. `context_downsampled=True`. Caption `Soma context downsampled for display`.
- [ ] `hull_polyline`: anatomy + ≥3 context points → 2D XY convex hull (scipy) with median Z. Empty on schematic. Derived from **somas**, not a second optic-lobe cartoon.
- [ ] `build_explorer_payload` adds `context_positions`, `hull_polyline`; flag defaults: `hull_mode="schematic_cns"`, `context_n=0`, `context_downsampled=False`, `anatomy_missing=False`, `n_model`, `n_viz`, `n_nodes_display`. Existing keys unchanged in meaning.
- [ ] JSON-serializable. Context is `[[x,y,z], …]` only. Round ~4–6 decimals OK.
- [ ] `subset_scene` is reservoir-only and must not delete context. Live 512 cap applies to reservoir `nodes`/`states` only; attach capped context separately.
- [ ] Tests (tmp soma dir / in-memory `SomaTable`; **do not** rely on workspace `data/raw/connectome/`):
  - synthetic → schematic, empty context
  - fake somas ⊃ reservoir ids → `hull_mode=malecns_anatomy`, **non-empty** `context_positions`, `n_viz > n_model`, `states.shape[1] == n_nodes_display`, `n_model` equals provenance width (e.g. 8) even if display nodes later capped to 4
  - downsample N=100 cap=10 → 10 points, deterministic, numeric-id order
  - missing soma + `real_connectome` → `anatomy_missing`, schematic
  - **3D spring** `layout.json` (non-zero Z, produced as spring) is **not** anatomical; soma table still wins
  - planar spring also non-anatomical
- [ ] No writes to `weights.npz` or run `provenance.json`. No change to train `n_nodes` validation.

## Implementation Notes

**Files to create**

- `src/pdm/visualization/scene.py` (preferred)

**Files to modify**

- `src/pdm/visualization/explorer.py` — `build_explorer_payload` keys; keep `load_scene_from_run` backward compatible
- `src/pdm/connectome/layout.py` — `is_anatomical_positions` / spring detection that treats **any** `layout_positions()` result as non-anatomical (tag in layout.json later is OK; for existing files, a `source`/`method` key is absent ⇒ assume spring unless graph nodes carried raw `x,y,z` from a soma join at train time — they do not today)
- `tests/test_neural_explorer.py`

**Spring detection (required)**

Existing train writes `{"positions": {id: [x,y,z?]}, "node_order": [...]}` with **no** anatomical flag. Those files are spring. Do not infer anatomy from Z span. Optional: 08 may start writing `layout.json` `"method": "spring"|"anatomical"` for **new** runs only; readers treat missing `method` as spring. Graph node attrs with `x,y,z` from a future train-time join could be anatomical; **do not** add train-time soma join in this epic unless free — viz-time soma win is enough.

**Do not** call `load_malemcns()` / full 152 M-edge graph.

**Gotchas**

- Live `LIVE_NODE_CAP=512` must not become `n_model`.
- `random_rewire` of a **synthetic** parent is still synthetic (schematic). Anatomy join only if parent is real.
- Overlay / `show_gt` — do not touch.

## Dependencies

Subtask 07 (soma loader + synthetic xyz fixture).

## Verification

```bash
.venv/bin/python -m pytest tests/test_neural_explorer.py -q
.venv/bin/python -m ruff check src tests
.venv/bin/python -m pytest tests -q
```

No `--smoke` train. No 1 GB feather.
