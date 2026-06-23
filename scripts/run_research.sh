#!/usr/bin/env bash
# Run the research pipeline with Python 3.9+ (avoids 3.8 import hangs on Apple Silicon).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -x "${CONDA_PREFIX:-}/bin/python" ]] && "$CONDA_PREFIX/bin/python" -c 'import sys; exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
  PYTHON="$CONDA_PREFIX/bin/python"
elif command -v python3 >/dev/null 2>&1 && python3 -c 'import sys; exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
  PYTHON="python3"
else
  echo "ERROR: No Python 3.9+ found. Install miniforge/conda or python3.10+." >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
echo "Using: $PYTHON ($("$PYTHON" --version 2>&1))"
exec "$PYTHON" run_research.py "$@"
