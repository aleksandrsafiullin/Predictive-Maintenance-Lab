# Subtask 10: Explorer + live widget consume anatomy payload

## Goal

Wire `screen_explorer` and the train-live widget to the 08 scene helper. **UI scene helper always wins** `hull_mode`. Catch soma unknown-schema `ValueError` (show columns, schematic + `anatomy_missing`, no Streamlit exception). Synthetic stays schematic + exact banner. Overlay, modes, and button labels stay as shipped.

## Context

Phase J. Today `app.py` hardcodes `"hull_mode": "schematic_cns"` and `live.py` stamps `hull_mode: schematic_cns` into `live_activity` extra — that stamp **must not overwrite** anatomy. Worker snapshot must stay light (no 80k xyz). Cache the soma table.

Do **not** change `EXPLORER_MODES` or labels `Build trace` / `Load demo scenario`.

## Acceptance Criteria

- [ ] `screen_explorer` builds payload via the 08 helper. `states` from `load_trace` (or empty). Context attached even with no trace. **`flags.n_model` from run/provenance**, not `len(display nodes)`.
- [ ] **Soma load is cached** (session / `functools.lru_cache` / module cache keyed by resolved path+mtime or hash). Not re-read every widget rerun. Tests may clear the cache.
- [ ] **Unknown-schema / explicit weights path `ValueError`:** catch in UI. Caption includes the exception text (**columns found**). Set `hull_mode="schematic_cns"`, `anatomy_missing=True`, empty context. **`not at.exception`**. Test with a junk file and with an explicit weights-like path (tmp dir).
- [ ] Allowlist miss / missing file: anatomy-missing caption, schematic, no exception (07 already returns empty).
- [ ] Synthetic / demo: exact `Synthetic test graph — not a biological connectome`. `hull_mode=schematic_cns`. Do not join FlyEM somas. AppTest: synthetic harness does not claim the cartoon is registered MaleCNS anatomy.
- [ ] Page caption still exact explorer disclaimer.
- [ ] `real_connectome` or **`random_rewire` of a real parent** without soma: `ANATOMY_MISSING_CAPTION` (grep-stable), `anatomy_missing=True`. **`random_rewire` of a synthetic parent:** schematic + synthetic banner, not anatomy-missing-as-FlyEM. AppTest both.
- [ ] Soma present: caption N_viz vs N_model; downsample caption if `context_downsampled`.
- [ ] Train-live: same scene helper. Reservoir states still `LIVE_NODE_CAP`. Context cap ≤ 20k with caption. **Never** upsample states. GRU/LSTM still `not_reservoir`.
- [ ] **`live.py`:** **remove** `hull_mode: "schematic_cns"` from snapshot extra (or stop writing `hull_mode` entirely). Snapshot must not include `context_positions` / 80k xyz. UI helper overwrites any leftover `hull_mode` key if present. Do not set ESN `n_nodes`.
- [ ] Overlay unchanged. `test_overlay_show_gt_does_not_move_predicted_zone` stays green. Build-trace-busy and GRU-required messages unchanged.
- [ ] AppTest: `Build trace` / `Load demo scenario` present; `EXPLORER_MODES` unchanged; screen switch by label; real_connectome without soma → anatomy-missing text; junk/weights path → columns caption, no exception.
- [ ] Tests monkeypatch soma dir to tmp. Never assume workspace `data/raw/connectome/` is empty.
- [ ] UI English. Bind `127.0.0.1:8501`.

## Implementation Notes

**Files to modify**

- `src/pdm/app.py` — stop hardcoding schematic for real anatomy; catch `ValueError` from soma/scene; cache
- `src/pdm/visualization/live.py` — **delete** the `hull_mode` stamp in `write_training_live_snapshot` extra (currently `"hull_mode": "schematic_cns"`)
- `src/pdm/visualization/explorer.py` — `ANATOMY_MISSING_CAPTION`, `SOMA_DOWNSAMPLE_CAPTION`
- `tests/test_neural_explorer.py`

**Do not modify**

- `overlay.py` leakage
- `EXPLORER_MODES` values
- Worker train/eval loops, `n_nodes` range, `filters_full_history`

**Captions**

```text
ANATOMY_MISSING_CAPTION = (
    "Anatomical soma coordinates are missing; showing a labeled schematic, not FlyEM MaleCNS anatomy."
)
SOMA_DOWNSAMPLE_CAPTION = "Soma context downsampled for display"
```

Schema-error caption should include the `ValueError` message (columns list).

**Gotchas**

- `subset_scene` then attach context — do not slice context to 512.
- Demo must not become `anatomy_missing` FlyEM warning.
- Do not load soma inside the worker epoch loop.

## Dependencies

Subtask 08 (required). Subtask 09 (look). AppTest can pass 08+10 without certifying WebGL look.

## Verification

```bash
.venv/bin/python -m pytest tests/test_neural_explorer.py tests/test_worker_and_app.py -q
.venv/bin/python -m ruff check src tests
.venv/bin/python -m pytest tests -q
```

Browser when tools exist: `:8501` four modes, overlay, busy Build trace, synthetic banner, anatomy-missing, schema-error caption. **AppTest does not certify look.**
