#!/usr/bin/env bash
# End-to-end reproduction: download data → backtest → publish docs for GitHub.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -x "${CONDA_PREFIX:-}/bin/python" ]] && "$CONDA_PREFIX/bin/python" -c 'import sys; exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
  PYTHON="$CONDA_PREFIX/bin/python"
elif command -v python3 >/dev/null 2>&1 && python3 -c 'import sys; exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
  PYTHON="python3"
else
  echo "ERROR: Python 3.9+ required (miniforge/conda or system python3)." >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
CONFIG="${1:-configs/base.yaml}"

echo "==> Using $PYTHON ($("$PYTHON" --version 2>&1))"
echo "==> Installing dependencies (if needed)…"
"$PYTHON" -m pip install -q -r requirements.txt

echo "==> Downloading market data (150 stocks + benchmarks)…"
"$PYTHON" scripts/download_yfinance.py \
  --symbols-file configs/symbols_n150.txt \
  --start 2018-01-01 \
  --end 2026-06-23 \
  --output-dir csv

echo "==> Running research pipeline…"
"$PYTHON" run_research.py --config "$CONFIG"

echo "==> Publishing results to docs/results/…"
"$PYTHON" scripts/publish_results.py

echo ""
echo "Reproduction complete."
echo "  Interactive report: docs/results/latest_report.html"
echo "  Summary doc:        docs/RESULTS.md"
