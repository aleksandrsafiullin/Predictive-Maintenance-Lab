#!/usr/bin/env bash
# Double-click in Finder, or run from Terminal, to start/restart the Streamlit UI.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
PORT=8501
APP="$ROOT/src/pdm/app.py"

pause_if_tty() {
  if [[ -t 0 ]]; then
    echo
    read -r -p "Press Return to close..."
  fi
}

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv. Run ./scripts/setup.sh first." >&2
  pause_if_tty
  exit 1
fi

is_pdm_streamlit() {
  local pid="$1"
  local cmd
  cmd="$(ps -p "$pid" -o command= 2>/dev/null || true)"
  [[ "$cmd" == *streamlit* && "$cmd" == *pdm/app.py* ]]
}

stop_existing() {
  local pids pid leftover i
  pids="$(lsof -nP -t -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -z "${pids}" ]]; then
    return 0
  fi
  leftover=0
  for pid in $pids; do
    if is_pdm_streamlit "$pid"; then
      echo "Stopping previous UI (pid $pid)..."
      kill "$pid" 2>/dev/null || true
    else
      leftover=1
      echo "Port $PORT is in use by pid $pid (not this app)." >&2
    fi
  done
  if [[ "$leftover" -eq 1 ]]; then
    echo "Refusing to start: something else is listening on $PORT." >&2
    pause_if_tty
    exit 1
  fi
  for i in $(seq 1 20); do
    if ! lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.15
  done
  pids="$(lsof -nP -t -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
  for pid in $pids; do
    if is_pdm_streamlit "$pid"; then
      kill -9 "$pid" 2>/dev/null || true
    fi
  done
  sleep 0.2
}

stop_existing

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_SERVER_HEADLESS=true
export MallocNanoZone=0

# Headless Streamlit does not open a browser. Wait for a real 200 before
# probing — hitting the app during first import can SIGTRAP on this machine.
if [[ -z "${PDM_NO_BROWSER:-}" ]]; then
  nohup bash -c "
    sleep 3
    for _ in \$(seq 1 25); do
      if curl -sf -o /dev/null 'http://127.0.0.1:${PORT}/'; then
        open 'http://127.0.0.1:${PORT}/'
        exit 0
      fi
      sleep 1
    done
  " >/dev/null 2>&1 &
  disown || true
fi

echo "Predictive Maintenance Lab"
echo "http://127.0.0.1:${PORT}"
echo "Close this window to stop the UI."
echo

exec .venv/bin/python -m streamlit run "$APP" \
  --server.address 127.0.0.1 \
  --server.port "$PORT" \
  --server.headless true \
  --server.fileWatcherType none \
  --browser.gatherUsageStats false
