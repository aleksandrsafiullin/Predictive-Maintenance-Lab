#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if command -v python3.11 >/dev/null 2>&1; then
  PY=python3.11
elif command -v python3.12 >/dev/null 2>&1; then
  PY=python3.12
else
  PY=python3
fi

echo "Using $($PY -c 'import sys; print(sys.executable, sys.version.split()[0])')"
"$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip wheel
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install -e .
.venv/bin/python -m pdm doctor
echo "Setup complete. Launch with: ./scripts/run.sh"
