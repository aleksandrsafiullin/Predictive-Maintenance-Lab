# Subtask 7: Soma anatomy artifact (xyz keyed by body id)

## Goal

Add a **viz-only** MaleCNS soma xyz loader: string body ids matching `graph.json` node ids, **ordered filename allowlist**, provenance `graph_mode=viz_soma_xyz` (miss → `unavailable`) counted as **`n_points`**, a tiny labeled synthetic xyz fixture, and loud errors only on **explicit** weights/syn-points/unknown-schema paths. Do not change ESN `n_nodes`, and do **not** write `runs/.../connectome/provenance.json`.

## Context

Phase G of the MaleCNS explorer epic. Subtasks 01–06 already ship the leaky ESN, BFS subgraph (500–2000), and schematic WebGL explorer. The weights feather (`body_pre` / `body_post` / `weight`) has **no xyz**. This loader is the anatomy table for N_viz (~1e5 somas), not a new reservoir.

MaleCNS is **CC-BY**. Official scene is a visual reference only (no Neuroglancer runtime). CI must not require the soma file. Tests must **monkeypatch the soma dir** to a tmp path — never assume workspace `data/raw/connectome/` is empty.

## Acceptance Criteria

- [ ] New module `src/pdm/connectome/anatomy.py`. Public API includes at least:
  - `default_soma_dir() -> Path` → `data/raw/connectome/` (same dir as `default_malemcns_path()`)
  - `SOMA_ALLOWLIST`: ordered tuple/list of **exact filenames**, first entry `body-annotations-male-cns-v1.0-minconf-0.5.feather`. Additional names only if documented. **No `*.feather` glob.**
  - `load_soma_table(path: str | Path | None = None) -> SomaTable`
- [ ] `SomaTable` fields: `positions: dict[str, list[float]]` (body id → `[x,y,z]`), `provenance: dict`, `is_synthetic: bool`, **`n_points: int`** (not `n_nodes`). Body ids are **`str`** via `as_node_id`.
- [ ] **Default search (`path is None`):** walk `SOMA_ALLOWLIST` in order under `default_soma_dir()`. Skip a candidate **by name** if it equals `MALEMCNS_FILENAME` or matches `syn-points-*` **before** opening the file / reading schema. First remaining allowlisted file that exists is loaded. If **none** exist → empty `positions`, `source="unavailable"`, `graph_mode="unavailable"`, **do not raise**.
- [ ] **Explicit `path`:** load that file (file or dir+allowlist join). If the basename is `MALEMCNS_FILENAME` or `syn-points-*` → **`ValueError`** (skip-by-name still applies; do not parse 152 M edges). If schema is unknown → `ValueError` listing **actual** columns (same style as `_map_malemcns_columns`). Explicit path to a weights-like or syn-points table **still raises** even if default search would have skipped it.
- [ ] **Do not invent columns.** Map documented FlyEM-style names (case-insensitive). Id candidates: `bodyId`, `body_id`, `bodyid`, `body`. Xyz candidates: `somaLocation` (list/tuple/dict with x,y,z), numeric `x,y,z`, or `soma_x,soma_y,soma_z`. A table whose columns are `x_pre`/`y_pre`/`z_pre` (and/or `body_pre`/`body_post`/`weight`) is **not** a soma table → `ValueError`.
- [ ] `load_synthetic_somas()` reads package fixture `pdm.connectome.fixtures` (`synthetic_somas.json`) with **≤64** points, `is_synthetic=true`, disclaimer containing exact `Synthetic test graph — not a biological connectome`. `graph_mode="viz_soma_xyz"` (or fixture-specific synthetic) plus `is_synthetic=true`. Never returned from `load_soma_table()` on a missing real file.
- [ ] Provenance via `build_provenance` **returned on the SomaTable only**:
  - loaded file: `graph_mode="viz_soma_xyz"`, `n_points` in extra (do **not** put soma count in provenance `n_nodes` as if it were ESN width; pass `n_nodes=0` or omit-equivalent and store count as `n_points`)
  - miss / allowlist miss: `graph_mode="unavailable"`, `source="unavailable"`, `n_points=0`
  - `file_hash` when a local file exists; `column_names_read` = columns actually used; `license="CC-BY"`; `role="viz_soma_xyz"`; `n_model_unaffected=true`
  - **Never** call `write_provenance` / `write_graph_artifact` on `runs/.../connectome/provenance.json`
- [ ] Documented human download constants (strings only): GCS prefix `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/`, hub `https://male-cns.janelia.org/download/`, existing `NEUPRINT_NOTE`. No `neuprint` / `caveclient` / `neuroglancer` import. Never download in this function.
- [ ] `pyproject.toml` package-data already covers `*.json` under fixtures. Do not add the 166k file to git.
- [ ] Optional: `pdm doctor` reports `soma_table_present: bool` (allowlist hit) and path; missing is **not** an error.
- [ ] Tests in `tests/test_neural_explorer.py`, **all using a tmp dir + monkeypatch of `default_soma_dir` / search path** (never the workspace `data/raw/connectome/`):
  - synthetic fixture labeled + string ids + xyz length 3
  - empty tmp dir / allowlist miss → empty, `graph_mode="unavailable"`, not `real_connectome`, **does not raise**
  - allowlisted tiny fake table with `bodyId` + `x,y,z` (or `somaLocation`) loads; `graph_mode="viz_soma_xyz"`; `n_points` set
  - unknown columns on **explicit** path raise `ValueError` matching `columns found`
  - **explicit** path to a weights-like table (`body_pre`/`body_post`/`weight`, basename `MALEMCNS_FILENAME`) raises
  - **`syn-points-*` filename** in default dir is skipped by name (empty / next allowlist hit); **explicit** path to `syn-points-….feather` **raises**
  - table with `x_pre`/`y_pre`/`z_pre` on explicit path raises (not treated as soma xyz)
  - same file twice → same id→xyz mapping (deterministic)
- [ ] No change to `resolve_n_nodes`, `load_malemcns_subgraph`, `W_res`, or train.

## Implementation Notes

**Files to create**

- `src/pdm/connectome/anatomy.py`
- `src/pdm/connectome/fixtures/synthetic_somas.json` (tiny, labeled synthetic)

**Files to modify**

- `src/pdm/connectome/__init__.py` — optional re-export; do not pull heavy IO at import
- `src/pdm/cli.py` — doctor field only if cheap (stat allowlist paths)
- `tests/test_neural_explorer.py`

**Do not modify**

- `src/pdm/models/**`, `losses.py`, `splits.py`, `windows.py`, `connectome/weights.py`, `connectome/sampling.py` BFS, ESN update
- `.cursor/tasks/subtask-01-*.md` … `subtask-06-*.md`
- Any `runs/**/connectome/provenance.json` writer (`train.py` `_write_connectome_artifacts`)

**Allowlist vs explicit**

```
path is None:
  for name in SOMA_ALLOWLIST:
    if name == MALEMCNS_FILENAME or name.startswith("syn-points-"):
      continue  # by name, before open
    candidate = default_soma_dir() / name
    if candidate.is_file():
      return parse(candidate)   # schema ValueError only if this allowlisted file is corrupt
  return empty(unavailable)

path is not None:
  if basename is MALEMCNS_FILENAME or syn-points-*:
    raise ValueError(...)
  parse(path)  # unknown schema / x_pre table → ValueError
```

**Column policy (file opened)**

1. `pd.read_feather` / parquet / JSON by suffix.
2. Log `list(frame.columns)`; never guess unseen names into the mapper.
3. `somaLocation` may be `[x,y,z]`, `{"x","y","z"}`, or stringified JSON list — parse only those; drop non-finite xyz and record `n_dropped`.
4. Units (nm vs voxels) stored raw; scene builder (08) normalizes bbox.

**Gotchas**

- Weights feather and possible syn-points dumps live in the same directory — **name skip first**, never glob.
- ~166k rows is fine for pandas; do not build NetworkX of somas.
- `build_provenance` currently requires `n_nodes`. Pass `n_nodes=0` (or the unused graph size) and put the soma count in **`extra["n_points"]`**. Do not advertise soma count as ESN `n_nodes`.

## Dependencies

None (first new-epic subtask). Historical 01–06 are already on the branch.

## Verification

```bash
.venv/bin/python -m pytest tests/test_neural_explorer.py tests/test_reservoir.py tests/test_spec_invariants.py -q
.venv/bin/python -m ruff check src tests
```

Full suite before claiming done:

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check src tests
```

Optional: `.venv/bin/python -m pdm doctor` shows soma present/missing without failing.
