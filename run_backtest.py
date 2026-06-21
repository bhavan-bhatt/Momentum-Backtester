#!/usr/bin/env python3
# run_backtest.py
# ============================================================
# ENTRY POINT — SINGLE BACKTEST RUN
#
# Usage:
#   python run_backtest.py                        # defaults from config.py
#   python run_backtest.py --strategy dual_ma
#   python run_backtest.py --strategy rsi
#   python run_backtest.py --strategy combined
#   python run_backtest.py --strategy combined --no-report
# ============================================================

import argparse
import logging
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    os.makedirs("output/logs", exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("output/logs/backtest.log", mode="a"),
        ],
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Event-driven backtester — single run")
    parser.add_argument(
        "--strategy", choices=["dual_ma", "rsi", "combined"], default="combined",
        help="Strategy to run (default: combined).",
    )
    parser.add_argument("--start",   default=None, help="Override start date (YYYY-MM-DD).")
    parser.add_argument("--end",     default=None, help="Override end date (YYYY-MM-DD).")
    parser.add_argument("--capital", type=float, default=None, help="Override initial capital (INR).")
    parser.add_argument(
        "--sizing", choices=["atr", "equal_weight", "fixed"], default=None,
        help="Override position sizing method.",
    )
    parser.add_argument("--no-report", action="store_true", help="Skip HTML report and charts.")
    parser.add_argument("--quiet",     action="store_true", help="Suppress progress output.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    from config import CONFIG

    # CLI overrides
    if args.start:   CONFIG.data.start_date = args.start
    if args.end:     CONFIG.data.end_date   = args.end
    if args.capital: CONFIG.portfolio.initial_capital = args.capital
    if args.sizing:  CONFIG.portfolio.sizing_method   = args.sizing
    if args.quiet:   CONFIG.verbose = False

    _setup_logging(CONFIG.verbose)

    # Select strategy class
    if args.strategy == "dual_ma":
        from strategies.dual_ma import DualMAStrategy as StrategyClass
    elif args.strategy == "rsi":
        from strategies.rsi_strategy import RSIStrategy as StrategyClass
    else:
        from strategies.combined import CombinedStrategy as StrategyClass

    # Build all components and run
    from engine.backtest import build_backtest_engine
    engine  = build_backtest_engine(CONFIG, StrategyClass)
    results = engine.run()

    equity_curve     = results["equity_curve"]
    trade_log        = results["trade_log"]
    metrics          = results["metrics"]
    benchmark_curve  = results["benchmark_curve"]

    if equity_curve.empty:
        print("\nNo equity data recorded — no trades may have been generated.")
        return

    # Generate report
    if not args.no_report:
        from reports.generator import ReportGenerator
        gen = ReportGenerator(CONFIG)
        gen.save_csv_artifacts(equity_curve, trade_log, engine.strategy.strategy_id)
        gen.save_static_charts(
            equity_curve, trade_log, engine.strategy.strategy_id, benchmark_curve
        )
        gen.generate(
            equity=equity_curve,
            trade_log=trade_log,
            metrics=metrics,
            strategy_id=engine.strategy.strategy_id,
            benchmark_equity=benchmark_curve,
        )


if __name__ == "__main__":
    main()
