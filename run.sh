#!/bin/bash
# Mac / Linux: first run sets up .venv (about a minute), then starts teleop.
#   ./run.sh              (robot "hrg" over Tailscale)
#   ./run.sh --dry        (camera + preview only, nothing sent)
#   ./run.sh --robot 100.97.141.88
cd "$(dirname "$0")" || exit 1
PY=${PYTHON:-python3}
command -v "$PY" >/dev/null || { echo "Python 3 not found -- install it from python.org"; exit 1; }
REQ=$(cat requirements.txt | (md5sum 2>/dev/null || md5) | cut -c1-32)
if [ ! -x .venv/bin/python ] || [ "$(cat .venv/.req 2>/dev/null)" != "$REQ" ]; then
    echo "setting up .venv (one time)..."
    "$PY" -m venv .venv || exit 1
    .venv/bin/python -m pip install -q --upgrade pip || exit 1
    .venv/bin/python -m pip install -q -r requirements.txt || exit 1
    echo "$REQ" > .venv/.req
fi
exec .venv/bin/python teleop.py "$@"
