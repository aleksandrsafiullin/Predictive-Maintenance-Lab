# Subtask 4: Neural Activity Explorer (WebGL component + Streamlit page)

## Goal

Add a Streamlit screen **Neural Activity Explorer** that animates **real** reservoir states from `predict_with_trace` in a local WebGL/Three.js custom component (assets bundled in-repo, no CDN at demo time). Animation runs in the browser, not via Streamlit rerun-per-frame.

## Context

Phase D. Phase C produced lazy trace artifacts and frame maps. `src/pdm/app.py` currently has three sidebar screens: `Data`, `Train`, `Test & Replay` (`st.sidebar.radio`, ~line 126). Worker still owns heavy jobs. Streamlit AppTest in `tests/test_worker_and_app.py` must keep starting without exceptions.

This is **not** a React SPA replacing the lab, and not FastAPI. A small custom component (vanilla JS + vendored Three.js) is in scope because the spec requires WebGL.

## Acceptance Criteria

- [ ] Sidebar screen list becomes `Data`, `Train`, `Test & Replay`, `Neural Activity Explorer`. Existing three screens still render.
- [ ] Page caption **exactly**: `Computational activity in a connectome-based reservoir. This is not a biophysical simulation or recorded activity of a living fly brain.`
- [ ] When `graph_mode=synthetic_fixture` (or `is_synthetic=True`), the page **always** shows: `Synthetic test graph — not a biological connectome` (banner/caption, not only in a tooltip).
- [ ] Custom component lives in `src/pdm/visualization/component/` and is declared with a **local path** (`streamlit.components.v1.declare_component`). `index.html` / JS have **no** `cdn.jsdelivr`, `unpkg`, `cdnjs`, `googleapis.com` script tags. Three.js is vendored under `frontend/vendor/`.
- [ ] Node colors/sizes/flashes are mapped from loaded `states[frame, node]` (or contributions), not from `Math.random`, timers, or CSS animation unrelated to state. A unit test or component contract test asserts the payload includes the trace array that Python just computed.
- [ ] Animation playback (play/pause, frame index, speed) is handled **inside** the component (JS). Streamlit does not `st.rerun` every frame. Sync controls (mode, unit, run, window) may rerun once on user change.
- [ ] Modes: **Overview**, **Equipment replay**, **Inside prediction window**, **Alert inspection**. Switching mode does not retrain. Alert inspection may be a stub that reads existing `alerts.csv` / episodes if Phase E is not done, but must not crash.
- [ ] Layout: topological coordinates from `connectome/layout.json` are required. Anatomical 3D used only if xyz exist in the graph artifact; otherwise topological 3D/2D fallback (never a blank scene).
- [ ] Equipment replay and Inside-window modes consume `frame_map.json` from subtask 03 (replay step ↔ window frames). No future frames.
- [ ] Heavy trace generation for non-tiny graphs goes through `pdm.worker` (`kind=trace` or evaluate-with-trace). UI does not train. Stop → `cancelled`, not `completed`.
- [ ] GRU/LSTM-only runs: page explains that explorer requires a reservoir run; does not invent spikes from GRU hidden size.
- [ ] `pyproject.toml` package-data includes `pdm.visualization.component` frontend files so the component works from the editable install.
- [ ] `tests/test_neural_explorer.py`: AppTest opens the new screen without exception; caption and synthetic banner present on a synthetic fixture mock; HTML/JS fixtures contain no CDN URLs. `tests/test_worker_and_app.py` still green.
- [ ] UI English. Bind remains `127.0.0.1:8501`.

## Key Files to Create/Modify

**Create**

- `src/pdm/visualization/component/__init__.py` — `declare_component` wrapper, Python args: nodes, edges, layout, states, frame_map, labels, flags
- `src/pdm/visualization/component/frontend/index.html`
- `src/pdm/visualization/component/frontend/main.js` — Three.js scene, play loop, `Streamlit.setComponentValue` for current frame if needed
- `src/pdm/visualization/component/frontend/vendor/three.min.js` (vendored; record version + license in a short `VENDOR.txt`)
- `tests/test_neural_explorer.py`

**Modify**

- `src/pdm/app.py` — sidebar radio + `screen_explorer(dataset_id)`; do not inline train/eval; reuse `list_runs` / selected `run_id` where possible
- `src/pdm/worker.py` — `kind=trace` if not finished in 03; status `cancelled` on stop
- `src/pdm/cli.py` — `pdm app` unchanged except the new page is inside the same app
- `pyproject.toml` — `[tool.setuptools.package-data]`
- `tests/test_worker_and_app.py` — only if sidebar radio index assumptions break (prefer locating by label, not hardcoded index 2 everywhere)

**Do not**

- Pull Three.js from a CDN in HTML
- Add FastAPI or a separate Node dev server as a runtime requirement
- Enable `filters_full_history`
- Flash nodes without a state vector

## Implementation Notes

### Vendoring Three.js

Commit a pinned `three.min.js` (MIT). `VENDOR.txt` names version. Tests walk `frontend/**/*.html` and `*.js` (except vendor) for `https://` script sources and fail on CDN hosts. Vendor file may be local only.

If the environment cannot fetch Three.js during implementation, ship a **minimal WebGL renderer** (points + lines) that satisfies “no CDN” and still animates from state arrays; document that Three.js can replace it. Do **not** leave a blank “waiting for internet” explorer.

### Streamlit custom component

Use `declare_component(path=str(frontend_dir))` with a static `index.html` that talks to Streamlit’s component API. Avoid `npm run build` in CI. No `http://localhost:3001` dev URL in production `declare_component`.

Pass JSON/Arrow-serializable payloads: `node_id` list, edge list, `xyz`, `states` (may downsample N for UI if needed — if downsampled, caption must say so; tests use tiny N so no downsample).

### Modes (UI copy)

| Mode | Behavior |
|------|----------|
| Overview | Full graph, idle or looping a selected unit’s last window |
| Equipment replay | Frame index follows replay step mapping; same unit as Test & Replay if `st.session_state` has one |
| Inside prediction window | Play only frames `0..H-1` of the window that produced the selected prediction |
| Alert inspection | Select an alert episode timestamp; load that window’s trace (full jump wiring in subtask 05 if alerts table not yet on this page) |

Shared session keys with Test & Replay (`run_id`, `unit_id`, replay step) should be read if set; explorer must still work if the user never opened Replay.

### Fake flashes

Forbidden: `sin(time)`, random node selection, looping a canned GIF. Allowed: interpolate displayed color between consecutive **provided** state frames for smoothness (still data-driven).

### Worker vs UI

Clicking “Build trace” spawns a worker if `n_nodes` is large or the unit has no artifact. Poll `status.json` like Train. Tiny synthetic tests can call `predict_with_trace` in-process.

### AppTest

`AppTest.from_file(.../app.py)`. Set sidebar Screen to `Neural Activity Explorer`. Assert `not at.exception`. Search `at.markdown` / captions / `at.info` for the two required sentences. Mock `list_runs` / trace loader so CI needs no real connectome file.

## Dependencies

Subtask 03 (traces, frame maps, contributions). Subtask 01 synthetic label strings.

## Verification

```bash
.venv/bin/python -m pytest tests/test_neural_explorer.py -q
.venv/bin/python -m pytest tests/test_worker_and_app.py -q
.venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check src tests
```

If browser tools are available: open `http://127.0.0.1:8501`, select Neural Activity Explorer, load a synthetic-fixture run, press Play, confirm nodes change with frame index and both captions are visible. If not available, AppTest + static CDN grep are the merge gate; say so in the subtask summary.

## Notes on constraints

- No CDN (hard constraint 10).
- Real states from the same inference as prediction (no fake flashes).
- Synthetic banner always on (hard constraint 4).
- Required biological-disclaimer caption (spec).
- Do not break Data / Train / Test & Replay.
- Cancel trace jobs → `cancelled`, not `completed`.
- English UI; localhost bind unchanged.
