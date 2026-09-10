#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv. Run ./scripts/setup.sh first." >&2
  exit 1
fi
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec .venv/bin/streamlit run "$ROOT/src/pdm/app.py" \
  --server.address 127.0.0.1 \
  --server.port 8501 \
  --browser.gatherUsageStats false
