#!/usr/bin/env python3
# run_backtest.py
# ============================================================
# ENTRY POINT — SINGLE BACKTEST RUN
#
# Usage:
#   python run_backtest.py                        # uses defaults from config.py
#   python run_backtest.py --strategy dual_ma
#   python run_backtest.py --strategy rsi
#   python run_backtest.py --strategy combined
#   python run_backtest.py --strategy combined --no-report
# ============================================================

import argparse
import logging
import os
import sys

# Ensure project root is on the path (needed when run from a different CWD)
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
    parser.add_argument(
        "--start", default=None,
        help="Override start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end", default=None,
        help="Override end date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--capital", type=float, default=None,
        help="Override initial capital (INR).",
    )
    parser.add_argument(
        "--sizing", choices=["atr", "equal_weight", "fixed"], default=None,
        help="Override position sizing method.",
    )
    parser.add_argument(
        "--no-report", action="store_true",
        help="Skip HTML report and chart generation.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    from config import CONFIG

    # Apply CLI overrides
    if args.start:
        CONFIG.data.start_date = args.start
    if args.end:
        CONFIG.data.end_date = args.end
    if args.capital:
        CONFIG.portfolio.initial_capital = args.capital
    if args.sizing:
        CONFIG.portfolio.sizing_method = args.sizing
    if args.quiet:
        CONFIG.verbose = False

    _setup_logging(CONFIG.verbose)

    # Select strategy class
    if args.strategy == "dual_ma":
        from strategies.dual_ma import DualMAStrategy as StrategyClass
    elif args.strategy == "rsi":
        from strategies.rsi_strategy import RSIStrategy as StrategyClass
    else:
        from strategies.combined import CombinedStrategy as StrategyClass

    # Run backtest
    from engine.backtest import Backtest
    bt = Backtest(CONFIG, StrategyClass)
    bt.run()

    equity    = bt.get_equity_curve()
    trade_log = bt.get_trade_log()

    if equity.empty:
        print("\nNo equity data recorded — no trades may have been generated.")
        return

    # Compute metrics
    from performance.metrics import compute_all_metrics
    bench_equity = None
    if bt.data_handler.benchmark_data is not None and "close" in bt.data_handler.benchmark_data.columns:
        bc = bt.data_handler.benchmark_data["close"]
        bench_equity = bc / bc.iloc[0] * CONFIG.portfolio.initial_capital

    metrics = compute_all_metrics(
        equity["equity"],
        trade_log,
        benchmark_equity=bench_equity,
        risk_free_rate=CONFIG.report.risk_free_rate,
        trading_days=CONFIG.report.trading_days_per_year,
    )

    # Print summary
    print("\n" + "─" * 50)
    print("  PERFORMANCE SUMMARY")
    print("─" * 50)
    fmt_map = {
        "total_return":          ("Total Return",         "{:.2%}"),
        "cagr":                  ("CAGR",                 "{:.2%}"),
        "annualised_volatility": ("Annual Volatility",    "{:.2%}"),
        "sharpe_ratio":          ("Sharpe Ratio",         "{:.3f}"),
        "sortino_ratio":         ("Sortino Ratio",        "{:.3f}"),
        "max_drawdown":          ("Max Drawdown",         "{:.2%}"),
        "max_dd_duration_days":  ("Max DD Duration",      "{:.0f} days"),
        "win_rate":              ("Win Rate",             "{:.2%}"),
        "profit_factor":         ("Profit Factor",        "{:.3f}"),
        "num_trades":            ("# Trades",             "{:.0f}"),
        "alpha":                 ("Alpha (annual)",       "{:.2%}"),
        "beta":                  ("Beta",                 "{:.3f}"),
    }
    for key, (label, fmt) in fmt_map.items():
        val = metrics.get(key)
        if val is None or (isinstance(val, float) and val != val):
            print(f"  {label:<25} N/A")
        else:
            try:
                print(f"  {label:<25} {fmt.format(val)}")
            except Exception:
                print(f"  {label:<25} {val}")
    print("─" * 50 + "\n")

    # Generate report
    if not args.no_report:
        from reports.generator import ReportGenerator
        gen = ReportGenerator(CONFIG)
        gen.save_csv_artifacts(equity["equity"], trade_log, bt.strategy.strategy_id)
        gen.save_static_charts(equity["equity"], trade_log, bt.strategy.strategy_id, bench_equity)
        gen.generate(
            equity=equity["equity"],
            trade_log=trade_log,
            metrics=metrics,
            strategy_id=bt.strategy.strategy_id,
            benchmark_equity=bench_equity,
        )


if __name__ == "__main__":
    main()
