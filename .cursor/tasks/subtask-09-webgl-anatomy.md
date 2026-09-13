# Subtask 9: WebGL MaleCNS layering (context vs reservoir)

## Goal

Update the vendored Three.js explorer so `hull_mode=malecns_anatomy` shows a **dim soma context cloud** + **reservoir glow from `states`** + soma-derived hull, and `schematic_cns` keeps the existing labeled cartoon. Idle almost black; additive blending only for active reservoir nodes. No CDN, no `Math.random`, no Neuroglancer.

## Context

Phase I. Subtask 08 freezes `context_positions`, `hull_polyline`, and `flags.hull_mode`. Today `main.js` always runs `polarToCns` / `fitNodesIntoHull` / `buildHullPositions` and additive bloom on **all** reservoir points.

Both `ThreeRenderer` and the raw `WebGLRenderer` fallback must honor the contract. **AppTest does not certify look.** Pytest greps + a Python payload test are the merge gate; browser on `:8501` when tools exist.

## Acceptance Criteria

- [ ] `applyArgs` reads `context_positions` and `hull_polyline`; missing keys → empty. **Missing / null `flags.hull_mode` defaults to `"schematic_cns"`.**
- [ ] When `flags.hull_mode === "malecns_anatomy"`:
  - Do **not** call `polarToCns` or `fitNodesIntoHull` cartoon mapping; use payload xyz (08 bbox-normalized).
  - Do **not** add parametric optic-lobe / VNC hull or those sprites.
  - Draw `hull_polyline` with low opacity, **normal** blending.
  - Context Points: **not** `AdditiveBlending`; dim grey dust.
  - Reservoir points at payload `positions`; color/size from `states` only.
- [ ] When `flags.hull_mode === "schematic_cns"`: keep cartoon hull + labels. No fake MaleCNS context cloud.
- [ ] **Idle almost black:** below-threshold reservoir nodes near-background, `haloSize=0`. Additive **only** for active reservoir nodes. Context never additive.
- [ ] Point size scales with N. 80k context must not occlude reservoir glow.
- [ ] Play / Pause / speed / frame slider stay in JS. No Streamlit per-frame rerun.
- [ ] `Math.random` absent from `main.js` and non-vendor frontend JS.
- [ ] `index.html` loads only `./vendor/streamlit-component-lib.js`, `./vendor/three.min.js`, `./main.js`. No `https://` script/module src.
- [ ] HUD: exact synthetic banner when `flags.is_synthetic`. Anatomy-missing / downsample lines from flags only; no invented FlyEM claims.
- [ ] Camera: anatomy `lookAt` bbox center of context+reservoir. Schematic may keep current camera.
- [ ] Edges optional; anatomy mode: reservoir endpoints only. Prefer **hide sensor strip** when `malecns_anatomy` (one-line comment).
- [ ] Tests (`tests/test_neural_explorer.py`):
  - existing `test_no_cdn_in_frontend` still green; `Math.random` not in `main.js`
  - **JS source grep (required):** `fitNodesIntoHull` and `polarToCns` are only invoked when `hull_mode === "schematic_cns"` (guard visible in source; fail if either is called unconditionally)
  - `context_positions` / `malecns_anatomy` / `schematic_cns` appear in `main.js`
  - AdditiveBlending remains for reservoir; context path uses `NormalBlending` (or non-additive blend)
  - **Python payload test:** `build_explorer_payload` / scene helper with fake somas yields `flags.hull_mode == "malecns_anatomy"` **and** non-empty `context_positions` (`n_viz > 0`, `n_model` from provenance)
- [ ] UI English.

## Implementation Notes

**Files to modify**

- `src/pdm/visualization/component/frontend/main.js`
- `src/pdm/visualization/component/frontend/index.html` — HUD hint only if needed

**Do not**

- Add EffectComposer, CDN postprocessing, or new vendor JS
- Iframe Neuroglancer
- Change Python payload keys (08 owns them)

**Renderer parity**

Both rebuild/render paths: context buffer once; skip cartoon hull when anatomy; additive only on the reservoir pass.

**Gotchas**

- Trust `hull_mode`, not Z-span. Anatomy after 08 normalize may look “planar.”
- Do not sample context colors from `states`.
- Auto-orbit may stay; it is not node flashing.

## Dependencies

Subtask 08 (payload contract).

## Verification

```bash
.venv/bin/python -m pytest tests/test_neural_explorer.py -q
.venv/bin/python -m ruff check src tests
.venv/bin/python -m pytest tests -q
```

**Browser when tools exist** (`http://127.0.0.1:8501`):

- Synthetic: cartoon + exact banner; no FlyEM claim
- Soma + real_connectome (if local file): dim cloud + small glow; no optic-lobe sprites
- Missing soma + real: schematic + (after 10) anatomy-missing caption

If no browser tools: say so. **AppTest passing does not certify MaleCNS look.**
