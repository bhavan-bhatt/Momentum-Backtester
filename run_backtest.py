# run_backtest.py
# ============================================================
# ENTRY POINT — RUN A FULL BACKTEST OR WALK-FORWARD VALIDATION
# Edit config.py to change parameters.
#
# Usage:
#   python run_backtest.py
#   python run_backtest.py --strategy dual_ma
#   python run_backtest.py --mode walk_forward --strategy combined --verbose
# ============================================================

import argparse
import logging
import os
import sys
from collections import deque

from runtime_checks import check_python_version

check_python_version()

from config import CONFIG, BacktestConfig
from engine.backtest import BacktestEngine
from engine.data_handler import DataHandler
from engine.execution import ExecutionHandler
from engine.execution_advanced import AdvancedExecutionHandler
from engine.portfolio import PortfolioManager
from engine.regime_filter import RegimeFilter
from reports.generator import ReportGenerator
from strategies.combined import CombinedStrategy
from strategies.cross_sectional_momentum import CrossSectionalMomentumStrategy
from strategies.dual_ma import DualMAStrategy
from strategies.pairs_stat_arb import PairsStatArbStrategy
from strategies.rsi_strategy import RSIStrategy
from strategies.volatility_breakout import VolatilityBreakoutStrategy
from data.constituents import ConstituentTracker

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging(config: BacktestConfig) -> None:
    """Configure the root logger for the entire run."""
    level = logging.INFO if config.verbose else logging.WARNING
    fmt   = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"

    handlers = [logging.StreamHandler(sys.stdout)]

    log_dir  = os.path.join(config.report.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    from datetime import datetime
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"backtest_{ts}.log")
    handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# STRATEGY FACTORY
# ──────────────────────────────────────────────────────────────────────────────

_STRATEGY_MAP = {
    "dual_ma":              DualMAStrategy,
    "rsi":                  RSIStrategy,
    "combined":             CombinedStrategy,
    "cross_sectional":      CrossSectionalMomentumStrategy,
    "pairs_stat_arb":       PairsStatArbStrategy,
    "volatility_breakout":  VolatilityBreakoutStrategy,
}

_REGIME_NAMES = {
    "dual_ma":              "DualMA",
    "rsi":                  "RSI",
    "cross_sectional":      "CrossSectionalMomentum",
    "pairs_stat_arb":       "PairsStatArb",
    "volatility_breakout":  "VolatilityBreakout",
}


def get_strategy(name: str, config: BacktestConfig, event_queue, regime_filter=None):
    """
    Factory function — return the correct strategy instance by name.

    Parameters
    ----------
    name        : Strategy key from _STRATEGY_MAP.
    config      : BacktestConfig.
    event_queue : Shared event deque.
    regime_filter : Optional RegimeFilter for entry gating.

    Returns
    -------
    An instantiated BaseStrategy subclass.
    """
    cls = _STRATEGY_MAP.get(name.lower())
    if cls is None:
        raise ValueError(
            f"Unknown strategy: '{name}'. "
            f"Valid choices: {list(_STRATEGY_MAP.keys())}"
        )

    if cls is CrossSectionalMomentumStrategy:
        tracker = ConstituentTracker(config)
        return cls(
            config,
            event_queue,
            constituent_tracker=tracker,
            regime_filter=regime_filter,
        )

    if cls in (PairsStatArbStrategy, VolatilityBreakoutStrategy):
        return cls(config, event_queue, regime_filter=regime_filter)

    strategy = cls(config, event_queue)
    regime_name = _REGIME_NAMES.get(name.lower())
    if regime_filter is not None and regime_name:
        strategy.set_regime_filter(regime_filter, regime_name)
    return strategy


def build_execution(config: BacktestConfig, queue):
    """Return Phase 1 or advanced execution handler based on config."""
    if config.advanced.advanced_execution.use_liquidity_aware_slippage:
        return AdvancedExecutionHandler(config, queue)
    return ExecutionHandler(config, queue)


def get_strategy_class(name: str):
    """Return the strategy class (not instance) for walk-forward use."""
    cls = _STRATEGY_MAP.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown strategy: '{name}'.")
    return cls


# ──────────────────────────────────────────────────────────────────────────────
# SINGLE BACKTEST
# ──────────────────────────────────────────────────────────────────────────────

def run_single_backtest(
    strategy_name: str = "combined",
    config: BacktestConfig = CONFIG,
) -> None:
    """
    Run one full backtest, generate all reports, and print a summary.

    Parameters
    ----------
    strategy_name : Strategy key from _STRATEGY_MAP.
    config        : BacktestConfig (defaults to the global CONFIG).
    """
    setup_logging(config)
    logger.info("Starting single backtest — strategy=%s", strategy_name)

    # ── Wire components ───────────────────────────────────────────────────
    queue          = deque()
    data_handler   = DataHandler(config, queue)
    regime_filter  = RegimeFilter(config)
    strategy       = get_strategy(strategy_name, config, queue, regime_filter)
    portfolio      = PortfolioManager(config, queue)
    execution      = build_execution(config, queue)
    engine         = BacktestEngine(
        config, data_handler, strategy, portfolio, execution,
        regime_filter=regime_filter,
    )

    # ── Run ───────────────────────────────────────────────────────────────
    results = engine.run()

    # ── Report ────────────────────────────────────────────────────────────
    reporter = ReportGenerator(config)
    report_path = reporter.generate_all(
        equity_curve    = results["equity_curve"],
        trade_log       = results["trade_log"],
        metrics         = results["metrics"],
        benchmark_curve = results["benchmark_curve"],
        strategy_name   = strategy.strategy_id,
    )
    print(f"\nReport saved to: {report_path}")
    logger.info("Backtest complete. Report: %s", report_path)


# ──────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD VALIDATION
# ──────────────────────────────────────────────────────────────────────────────

def run_walk_forward(
    strategy_name: str = "combined",
    config: BacktestConfig = CONFIG,
) -> None:
    """
    Run walk-forward validation across all date splits.

    Parameters
    ----------
    strategy_name : Strategy key from _STRATEGY_MAP.
    config        : BacktestConfig (defaults to the global CONFIG).
    """
    from performance.walk_forward import WalkForwardEngine

    setup_logging(config)
    logger.info("Starting walk-forward validation — strategy=%s", strategy_name)

    strategy_class = get_strategy_class(strategy_name)
    wf_engine      = WalkForwardEngine(config)

    splits = wf_engine.generate_splits()
    print(f"\nWalk-forward splits generated: {len(splits)}")
    for i, (ts, te, ss, se) in enumerate(splits, 1):
        print(f"  Split {i:2d}: Train [{ts} → {te}]  |  Test [{ss} → {se}]")

    if not splits:
        print("ERROR: No splits generated — check date range and walk_forward config.")
        return

    wf_results = wf_engine.run_all_splits(strategy_class)
    aggregate  = wf_engine.aggregate_results()
    wf_df      = wf_engine.generate_walk_forward_report()

    # ── Print aggregate table ─────────────────────────────────────────────
    print("\n" + "═" * 70)
    print(f"  WALK-FORWARD RESULTS — {strategy_name.upper()}")
    print("═" * 70)
    print(wf_df.to_string(index=False))
    print("═" * 70)

    for metric, stats in aggregate.items():
        if isinstance(stats, dict):
            mean = stats.get("mean")
            std  = stats.get("std")
            if mean is not None:
                print(f"  {metric:<25} mean={mean:>8.4f}  std={std or 0:>8.4f}")
    print()

    # ── Save CSV ──────────────────────────────────────────────────────────
    out_dir = os.path.join(config.report.output_dir, "logs")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"walk_forward_{strategy_name}.csv")
    wf_df.to_csv(csv_path, index=False)
    print(f"Walk-forward results saved to: {csv_path}")
    logger.info("Walk-forward complete. CSV: %s", csv_path)


# ──────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quantitative Backtesting Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_backtest.py\n"
            "  python run_backtest.py --strategy dual_ma\n"
            "  python run_backtest.py --mode walk_forward --strategy combined --verbose\n"
        ),
    )
    parser.add_argument(
        "--strategy",
        choices=list(_STRATEGY_MAP.keys()),
        default="combined",
        help="Strategy to backtest (default: combined)",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "walk_forward"],
        default="single",
        help="Run mode: single backtest or walk-forward validation (default: single)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging and progress output",
    )
    args = parser.parse_args()

    if args.verbose:
        CONFIG.verbose = True

    if args.mode == "single":
        run_single_backtest(args.strategy, CONFIG)
    else:
        run_walk_forward(args.strategy, CONFIG)
