# runtime_checks.py
# Fail fast before heavy imports when the environment is misconfigured.

import sys
import time

MIN_PYTHON = (3, 9)
RECOMMENDED_PYTHON = "/Users/bhavanbhatt/miniforge3/bin/python"


def check_python_version() -> None:
    """Exit with a clear message if Python is too old (3.8 hangs on imports)."""
    if sys.version_info >= MIN_PYTHON:
        return
    print(
        f"\nERROR: Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required.\n"
        f"  Current: {sys.executable} ({sys.version.split()[0]})\n"
        f"  Python 3.8 on Apple Silicon can take 20+ minutes to import pandas "
        f"and appears hung.\n\n"
        f"  Fix: use miniforge Python 3.10+\n"
        f"    {RECOMMENDED_PYTHON} run_research.py --config configs/base.yaml\n"
        f"  Or:  ./scripts/run_research.sh\n",
        file=sys.stderr,
    )
    sys.exit(1)


def timed_import(module_name: str, label: str, timeout_sec: float = 120.0) -> None:
    """Import a module with a visible progress message (guards against silent hangs)."""
    print(f"  Loading {label}…", flush=True)
    t0 = time.perf_counter()
    __import__(module_name)
    elapsed = time.perf_counter() - t0
    if elapsed > 10:
        print(f"  (took {elapsed:.1f}s — consider upgrading Python if this is >60s)", flush=True)
