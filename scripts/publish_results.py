#!/usr/bin/env python3
"""Copy latest backtest outputs into docs/results/ for GitHub publishing."""

from __future__ import annotations

import argparse
import glob
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
DOCS = ROOT / "docs" / "results"
CHARTS = DOCS / "charts"

CHART_ALIASES = {
    "_equity.png": "equity_curve.png",
    "_drawdown.png": "drawdown.png",
    "_monthly.png": "monthly_returns.png",
    "_rolling_sharpe.png": "rolling_sharpe.png",
    "_trades.png": "trade_distribution.png",
    "_sharpe_ci.png": "sharpe_confidence_interval.png",
}


def _latest(pattern: str) -> Path | None:
    matches = sorted(glob.glob(pattern), key=lambda p: Path(p).stat().st_mtime)
    return Path(matches[-1]) if matches else None


def publish(run_id: str | None = None) -> None:
    CHARTS.mkdir(parents=True, exist_ok=True)

    if run_id:
        prefix = run_id
    else:
        metrics = sorted((OUTPUT / "logs").glob("*_metrics.csv"))
        if not metrics:
            raise SystemExit("No metrics CSV found in output/logs/. Run the backtest first.")
        prefix = metrics[-1].name.replace("_metrics.csv", "")

    print(f"Publishing run: {prefix}")

    metrics_src = OUTPUT / "logs" / f"{prefix}_metrics.csv"
    if metrics_src.exists():
        shutil.copy2(metrics_src, DOCS / "metrics.csv")
        print(f"  metrics → {DOCS / 'metrics.csv'}")

    for suffix, alias in CHART_ALIASES.items():
        src = OUTPUT / "charts" / f"{prefix}{suffix}"
        if src.exists():
            dst = CHARTS / alias
            shutil.copy2(src, dst)
            print(f"  chart   → {dst}")

    report = OUTPUT / "reports" / f"{prefix}_report.html"
    if report.exists():
        dst = DOCS / "latest_report.html"
        shutil.copy2(report, dst)
        print(f"  report  → {dst}")

    dashboard = OUTPUT / "reports" / f"{prefix}_dashboard.html"
    if dashboard.exists():
        dst = DOCS / "latest_dashboard.html"
        shutil.copy2(dashboard, dst)
        print(f"  dashboard → {dst}")

    print("\nDone. Commit docs/results/ to share results on GitHub.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish backtest outputs to docs/results/")
    parser.add_argument("--run-id", help="Specific run prefix (default: latest metrics file)")
    args = parser.parse_args()
    publish(args.run_id)
