# Subtask 4: Neural Activity Explorer (WebGL component + Streamlit page)

## Goal

Add a Streamlit screen **Neural Activity Explorer** that animates **real** reservoir states from `predict_with_trace` in a local WebGL/Three.js custom component (Three.js **and** the Streamlit component bridge vendored in-repo; **no CDN**, no `https://` script/module src). Animation runs in the browser, not via Streamlit rerun-per-frame. `app.py` uses an explicit **4-way** screen branch. AppTest selects screens **by label**.

## Context

Phase D. Phase C produced lazy trace artifacts and frame maps. `src/pdm/app.py` currently has three sidebar screens: `Data`, `Train`, `Test & Replay` (`st.sidebar.radio`, then `else: screen_replay()` ~121–133). That `else` **must** go away. Worker still owns heavy jobs. Streamlit AppTest in `tests/test_worker_and_app.py` currently uses `at.sidebar.radio[1]` — those selectors must move to **label/options**, not index.

This is **not** a React SPA replacing the lab, and not FastAPI. A small custom component (vanilla JS + vendored Three.js + vendored component bridge) is in scope because the spec requires WebGL.

## Acceptance Criteria

- [ ] Sidebar screen list becomes `Data`, `Train`, `Test & Replay`, `Neural Activity Explorer`. Existing three screens still render.

- [ ] `main()` uses an **explicit 4-way branch** (W6), never `else: screen_replay()`:

  ```python
  if page == "Data":
      screen_data(dataset_id)
  elif page == "Train":
      screen_train(dataset_id)
  elif page == "Test & Replay":
      screen_replay(dataset_id)
  elif page == "Neural Activity Explorer":
      screen_explorer(dataset_id)
  else:
      st.error(f"Unknown screen: {page}")
  ```

- [ ] Page caption **exactly**: `Computational activity in a connectome-based reservoir. This is not a biophysical simulation or recorded activity of a living fly brain.`

- [ ] When `graph_mode=synthetic_fixture` (or `is_synthetic=True`), the page **always** shows: `Synthetic test graph — not a biological connectome` (banner/caption, not only in a tooltip).

- [ ] Custom component lives in `src/pdm/visualization/component/` and is declared with a **local path** (`streamlit.components.v1.declare_component`). **No** `https://` script or ES module `src` in `frontend/**/*.html` or non-vendor `*.js`. **No** `unpkg`, `cdn.jsdelivr`, `cdnjs`, `googleapis` URLs **anywhere** under that frontend (including comments that copy-paste CDN tags into HTML). Three.js **and** the Streamlit component bridge are vendored under `frontend/vendor/`.

- [ ] Test walks `src/pdm/visualization/component/frontend/**/*.html` and non-vendor `*.js` and fails if it finds `https://` as a script/module src or those CDN host strings.

- [ ] Node colors/sizes/flashes are mapped from loaded `states[frame, node]` (or contributions), not from `Math.random`, timers, or CSS animation unrelated to state. A unit test or component contract test asserts the payload includes the trace array that Python just computed.

- [ ] Animation playback (play/pause, frame index, speed) is handled **inside** the component (JS). Streamlit does not `st.rerun` every frame. Sync controls (mode, unit, run, window) may rerun once on user change.

- [ ] Modes: **Overview**, **Equipment replay**, **Inside prediction window**, **Alert inspection**. Switching mode does not retrain. Alert inspection may be a stub that reads existing `alerts.csv` / episodes if Phase E is not done, but must not crash.

- [ ] Layout: topological coordinates from `connectome/layout.json` are required. Anatomical 3D used only if xyz exist in the graph artifact; otherwise topological 3D/2D fallback (never a blank scene).

- [ ] Equipment replay and Inside-window modes consume `frame_map.json` from subtask 03 (replay step ↔ window frames). No future frames.

- [ ] Heavy trace generation for non-tiny graphs goes through `pdm.worker` (`kind=trace` or evaluate-with-trace). UI does not train. Stop → **`cancelled`**, not `completed` or `stopped`.

- [ ] **Build trace while worker busy (W8):** clicking “Build trace” (or equivalent) when `worker_alive()` is true shows an **error caption** and does **not** run `predict_with_trace` inline and does **not** spawn a second worker. Test this in `tests/test_neural_explorer.py` or AppTest.

- [ ] GRU/LSTM-only runs: page explains that explorer requires a reservoir run; does not invent spikes from GRU hidden size.

- [ ] `pyproject.toml` package-data includes `pdm.visualization.component` frontend files **and** `pdm.connectome.fixtures` (01) so the component and fixture work from the editable install.

- [ ] `tests/test_neural_explorer.py`: AppTest opens the new screen **by label** (options contain `Neural Activity Explorer`), not `sidebar.radio[2]` / `[1]`; `not at.exception`; caption and synthetic banner present on a synthetic fixture mock; HTML/JS contain no CDN URLs.

- [ ] `tests/test_worker_and_app.py`: every screen switch locates the Screen radio **by label/options** (e.g. a helper `select_screen(at, "Train")` that finds the radio whose options include `"Data"`), **not** `at.sidebar.radio[1]`. Existing flows still green.

- [ ] UI English. Bind remains `127.0.0.1:8501`.

## Key Files to Create/Modify

**Create**

- `src/pdm/visualization/component/__init__.py` — `declare_component` wrapper, Python args: nodes, edges, layout, states, frame_map, labels, flags
- `src/pdm/visualization/component/frontend/index.html` — local script tags only
- `src/pdm/visualization/component/frontend/main.js` — Three.js scene, play loop, `Streamlit.setComponentValue` for current frame if needed
- `src/pdm/visualization/component/frontend/vendor/three.min.js` — vendored; version + license in `VENDOR.txt`
- `src/pdm/visualization/component/frontend/vendor/streamlit-component-lib.js` (or equivalent Streamlit component bridge) — vendored; listed in `VENDOR.txt`
- `tests/test_neural_explorer.py`

**Modify**

- `src/pdm/app.py` — sidebar radio + **4-way** branch + `screen_explorer(dataset_id)`; busy-worker error caption on Build trace; do not inline train/eval; reuse `list_runs` / selected `run_id`
- `src/pdm/worker.py` — `kind=trace` if not finished in 03; status `cancelled` on stop
- `src/pdm/cli.py` — `pdm app` unchanged except the new page is inside the same app
- `pyproject.toml` — `[tool.setuptools.package-data]` for component frontend (`*.html`, `*.js`, `*.txt`)
- `tests/test_worker_and_app.py` — replace `sidebar.radio[1]` / index-based Screen selection with label/options helpers

**Do not**

- Pull Three.js, component-lib, fonts, or modules from a CDN
- Add FastAPI or a separate Node dev server as a runtime requirement
- Enable `filters_full_history`
- Flash nodes without a state vector
- Use `else: screen_replay()`

## Implementation Notes

### Vendoring (W5)

Commit pinned `three.min.js` (MIT) **and** the Streamlit custom-component JS bridge. `VENDOR.txt` names versions and licenses. Tests fail on:

- `<script src="https://...">` / `<script src='https://...'>`
- `type="module"` imports from `https://`
- host substrings `unpkg`, `cdn.jsdelivr`, `cdnjs`, `googleapis` anywhere in frontend HTML and non-vendor JS

Vendor files are local-only and excluded from the `https://` script-src scan (still scan them for accidental CDN redirects if easy).

If the environment cannot fetch Three.js during implementation, ship a **minimal WebGL renderer** (points + lines) that satisfies “no CDN” and still animates from state arrays; document that Three.js can replace it. Do **not** leave a blank “waiting for internet” explorer. The component bridge must still be local.

### Streamlit custom component

Use `declare_component(path=str(frontend_dir))` with a static `index.html`. Avoid `npm run build` in CI. No `http://localhost:3001` dev URL in production `declare_component`.

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

### Worker vs UI (W8)

Clicking “Build trace” when idle spawns a worker if `n_nodes` is large or the unit has no artifact. Poll `status.json` like Train. Tiny synthetic tests can call `predict_with_trace` in-process.

When the worker is **already** busy: **error caption only** — do not inline a second inference on the UI process.

### AppTest (W6)

```python
def _screen_radio(at):
    return next(r for r in at.sidebar.radio if "Data" in list(r.options) and "Train" in list(r.options))

_screen_radio(at).set_value("Neural Activity Explorer")
```

Do **not** use `at.sidebar.radio[1]`. Dataset radio is found by options containing `"Bearings"` / `"Filters"`, not `[0]`.

Mock `list_runs` / trace loader so CI needs no real connectome file.

## Dependencies

Subtask 03 (traces, frame maps, contributions). Subtask 01 synthetic label strings and `cancelled`.

## Verification Commands

```bash
.venv/bin/python -m pytest tests/test_neural_explorer.py -q
.venv/bin/python -m pytest tests/test_worker_and_app.py -q
.venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check src tests
```

If browser tools are available: open `http://127.0.0.1:8501`, select Neural Activity Explorer **from the Screen control**, load a synthetic-fixture run, press Play, confirm nodes change with frame index and both captions are visible. Confirm Build trace while a train job is running shows an error caption. If browser tools are not available, AppTest + static CDN grep are the merge gate; say so in the subtask summary.

## Notes/Constraints

- No CDN and no `https://` script/module src (W5, hard constraint 10). Vendor Three.js **and** the component bridge.
- Real states from the same inference as prediction (no fake flashes).
- Synthetic banner always on (hard constraint 4).
- Required biological-disclaimer caption (spec).
- Explicit 4-way branch; AppTest by label (W6).
- Busy worker → error caption, no inline trace (W8).
- Do not break Data / Train / Test & Replay.
- Cancel trace jobs → `cancelled`, not `completed` or `stopped`.
- English UI; localhost bind unchanged.
