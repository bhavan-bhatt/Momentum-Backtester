#!/usr/bin/env python3
# run_walk_forward.py
# ============================================================
# ENTRY POINT — WALK-FORWARD VALIDATION
#
# Usage:
#   python run_walk_forward.py
#   python run_walk_forward.py --strategy combined --train 3 --test 1
#   python run_walk_forward.py --strategy rsi
# ============================================================

import argparse
import logging
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _setup_logging() -> None:
    os.makedirs("output/logs", exist_ok=True)
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("output/logs/walk_forward.log", mode="a"),
        ],
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward validation")
    parser.add_argument(
        "--strategy", choices=["dual_ma", "rsi", "combined"], default="combined",
    )
    parser.add_argument("--train", type=int, default=None, help="Train window in years.")
    parser.add_argument("--test",  type=int, default=None, help="Test window in years.")
    parser.add_argument("--step",  type=int, default=None, help="Step size in years.")
    parser.add_argument("--metric", default="sharpe_ratio",
                        help="Metric to chart across splits.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _setup_logging()

    from config import CONFIG

    if args.train:
        CONFIG.walk_forward.train_years = args.train
    if args.test:
        CONFIG.walk_forward.test_years = args.test
    if args.step:
        CONFIG.walk_forward.step_years = args.step

    CONFIG.verbose = True

    if args.strategy == "dual_ma":
        from strategies.dual_ma import DualMAStrategy as StrategyClass
    elif args.strategy == "rsi":
        from strategies.rsi_strategy import RSIStrategy as StrategyClass
    else:
        from strategies.combined import CombinedStrategy as StrategyClass

    from performance.walk_forward import WalkForwardEngine
    engine  = WalkForwardEngine(CONFIG, StrategyClass)
    splits  = engine.run()
    summary = engine.summary()

    print("\n" + "=" * 80)
    print("  WALK-FORWARD RESULTS SUMMARY")
    print("=" * 80)
    display_cols = [c for c in [
        "split", "test_start", "test_end",
        "total_return", "cagr", "sharpe_ratio", "max_drawdown",
        "win_rate", "num_trades",
    ] if c in summary.columns]
    print(summary[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 80 + "\n")

    # Save summary CSV
    os.makedirs("output", exist_ok=True)
    safe_id = StrategyClass.__name__
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"output/walk_forward_{safe_id}_{ts}.csv"
    summary.to_csv(csv_path, index=False)
    print(f"[WalkForward] Summary saved → {csv_path}")

    # Chart across splits
    try:
        from reports.charts import walk_forward_summary_chart
        chart_path = f"output/charts/wf_{safe_id}_{args.metric}_{ts}.png"
        os.makedirs("output/charts", exist_ok=True)
        walk_forward_summary_chart(
            summary,
            metric=args.metric,
            save_path=chart_path,
        )
        print(f"[WalkForward] Chart saved → {chart_path}")
    except Exception as e:
        print(f"[WalkForward] Chart skipped: {e}")


if __name__ == "__main__":
    main()
