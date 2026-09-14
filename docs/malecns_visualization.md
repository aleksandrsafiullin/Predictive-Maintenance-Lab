# MaleCNS anatomy and model provenance

The operational WebGL explorer displays every computing reservoir neuron inside a **soma context cloud**. The cloud is display only; soma eligibility also determines new graph sampling. Visualization never rewrites the trained run's `provenance.json`.

MaleCNS / FlyEM Male CNS data is **CC-BY**. Place files yourself; the loader never downloads.

## Official scene — visual reference, not a runtime

Do **not** embed a Neuroglancer iframe. Do **not** add CAVE / Neuroglancer / neuPrint Python clients. These URLs are **reference** for the intended look (neuropil hull + soma field + a highlighted subset):

- Neuroglancer demo JSON: https://neuroglancer-demo.appspot.com/#!gs://flyem-male-cns/v1.0/male-cns-v1.0.json
- Explore hub: https://male-cns.janelia.org/explore/
- Download hub: https://male-cns.janelia.org/download/

CI and `pdm doctor` never fetch them. Low-level diagnostic views can use a labeled schematic fallback. The operational brain test run instead blocks a real model when any computing neuron lacks curated anatomy.

## Allowlist (no glob)

Soma xyz is loaded from `data/raw/connectome/` (same directory as the weights feather). Search walks an **ordered exact-filename allowlist**. **No `*.feather` glob.**

First (and currently only) name:

```text
body-annotations-male-cns-v1.0-minconf-0.5.feather
```

Skip **by name** before opening schema:

- `connectome-weights-male-cns-v1.0-minconf-0.5.feather` (`MALEMCNS_FILENAME`) — weights only
- any `syn-points-*` file — synapse locations, **not** somas

Allowlist miss → empty table, `source=unavailable`, `graph_mode=unavailable`. **Do not raise.** An **explicit** path to the weights feather, a `syn-points-*` file, an `x_pre`/`y_pre`/`z_pre` table, or unknown columns still raises `ValueError` (UI catches it and falls back to schematic).

Mapped columns (do not invent names): `bodyId` / `body_id` / `bodyid` / `body` plus `somaLocation` or `x,y,z` or `soma_x,soma_y,soma_z`.

## Weights vs synapses vs somas

| File | What the xyz/weight columns are | Use |
|------|----------------------------------|-----|
| Weights feather | `body_pre` / `body_post` / `weight` — **no xyz** | ESN subgraph (BFS 500–2000). Not a soma table. |
| `syn-points-*` | synapse xyz | **Not** soma xyz. Skip by name. |
| Body-annotations feather | soma xyz keyed by body id | Eligibility for new graph sampling; reservoir positions and optional context. |

Do **not** raise ESN `n_nodes` to ~166k to match the soma file. Reservoir width stays the run’s BFS subgraph.

## Human download (not CI)

GCS prefix (human):

```text
gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/
```

neuPrint (`https://neuprint.janelia.org/`) is an alternate **human** import step. Tests never call it. **CI never requires** the ~1 GB weights feather or a ~166k soma file.

`pdm doctor` reports `soma_table_present` / `soma_table_path` when the allowlist hits. Missing is not an error.

## Soma provenance is not run provenance

`load_soma_table()` returns viz provenance on the `SomaTable` object:

- loaded: `graph_mode=viz_soma_xyz`, count as **`n_points`** (`n_nodes=0`), `role=viz_soma_xyz`, `license=CC-BY`
- miss: `graph_mode=unavailable`

That record is **not** written over `runs/<dataset>/<run_id>/connectome/provenance.json` (that file stays the ESN graph from train).

## Layout.json is still spring

Train writes `connectome/layout.json` via `layout_positions()` spring (2D or 3D). That file is **schematic**, not FlyEM anatomy. Explorer joins soma xyz at **read time**. See [neural_activity_explorer.md](neural_activity_explorer.md).

`--smoke` is not a quality claim. AppTest does not certify WebGL look.

## Soma-aware sampling for new models

New `real_connectome` graphs use the allowlisted annotations file **beside the
selected weights file** when it is present. Only bodies with finite soma xyz are
eligible. BFS runs on the induced synapse graph; it retries another component if
needed, never pads with disconnected bodies. If no component is large enough,
training reports insufficient connected size. Missing annotations retain the
all-body BFS and the labeled anatomy fallback. No files are downloaded.

Graph provenance records `sampling_method=seeded_bfs_soma_xyz`,
`soma_file_hash`, `soma_local_path`, and `n_with_soma_available`. The soma reader's
own provenance remains separate. Node count stays 500–2000. This changes the
sampled graph, not the ESN update equation or the readout method.

Existing checkpoints retain their original graph and weights. The September 13
1000-node checkpoint has 116 soma matches and is blocked in the new operational
brain test run. A newly trained anatomy-complete model displays all 1000 computing
neurons. The 40,000 gray context points never compute or receive model states.
A spring position inside an anatomical bounding box is never accepted as anatomy.
Low-level diagnostic payloads may retain unmatched IDs numerically; they are not
accepted as a complete operational scene.
