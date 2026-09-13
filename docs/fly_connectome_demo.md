# Connectome reservoir demo

Working commands only. Use the project `.venv` Python. Smoke is **not** a quality benchmark. The default graph is a synthetic test graph unless you imported MaleCNS (see [fly_connectome.md](fly_connectome.md)).

```bash
# Setup (already done if .venv exists)
.venv/bin/python -m pdm doctor

# Smoke train with synthetic graph (8 nodes for CI speed)
.venv/bin/python -m pdm train --dataset bearings --arch fly_connectome_reservoir --smoke --n-nodes 8

# Start the UI
.venv/bin/python -m pdm app
```

UI: `http://127.0.0.1:8501` (localhost only).

Always pass `--n-nodes 8` for `synthetic_fixture` smoke. Do not run `--smoke` without an explicit tiny `--n-nodes`. The UI shows the synthetic banner unless a MaleCNS feather was imported with `graph_mode=real_connectome`.

Smoke MAE / explorer pictures are not model quality. Full 30-epoch training is not part of this demo.

For real MaleCNS data, see [docs/fly_connectome.md](fly_connectome.md) for import steps.
