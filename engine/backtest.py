# engine/backtest.py
# ============================================================
# THE MAIN EVENT LOOP — THE HEART OF THE BACKTESTER
# Iterates through time, dispatching events to the correct handlers.
# ============================================================

import logging
import time
from collections import deque
from typing import Dict, Optional, Type

import pandas as pd

from engine.events import EventType, MarketEvent, SignalEvent, OrderEvent, FillEvent
from engine.data_handler import DataHandler
from engine.strategy import BaseStrategy
from engine.portfolio import PortfolioManager
from engine.execution import ExecutionHandler
from config import BacktestConfig

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Orchestrates the entire backtest simulation.

    Event flow per bar
    ------------------
    DataHandler.update_bars()          → N × MarketEvent (one per symbol)
      MarketEvent  → strategy.calculate_signals(event, data_handler)
                       → may produce SignalEvent(s)
      SignalEvent  → portfolio.process_signal(event, data_handler)
                       → may produce OrderEvent(s)
      OrderEvent   → execution.execute_order(event, data_handler)
                       → produces FillEvent(s)
      FillEvent    → portfolio.update_portfolio_on_fill(event)
    End of bar: portfolio.update_equity_curve(timestamp)

    Constructor
    -----------
    Receives pre-built component instances. Creates its own event queue and
    injects it into every component so they all share the same deque.

    Usage
    -----
    engine = BacktestEngine(config, data_handler, strategy, portfolio, execution)
    results = engine.run()
    """

    def __init__(
        self,
        config: BacktestConfig,
        data_handler: DataHandler,
        strategy: BaseStrategy,
        portfolio: PortfolioManager,
        execution: ExecutionHandler,
    ) -> None:
        self._config    = config
        self.data_handler = data_handler
        self.strategy     = strategy
        self.portfolio    = portfolio
        self.execution    = execution

        # Shared event queue — injected into all components
        self.event_queue: deque = deque()
        data_handler._event_queue  = self.event_queue
        portfolio._event_queue     = self.event_queue
        execution._event_queue     = self.event_queue
        strategy._event_queue      = self.event_queue

        self._iteration_count: int = 0
        self._event_count:     int = 0
        self._results: Optional[Dict] = None

    # ──────────────────────────────────────────────────────────────────────
    # MAIN RUN
    # ──────────────────────────────────────────────────────────────────────

    def run(self) -> Dict:
        """
        Execute the full backtest from start to end date.

        Returns
        -------
        dict with keys:
          "equity_curve"   → pd.Series (date → portfolio value)
          "trade_log"      → pd.DataFrame of closed trades
          "metrics"        → dict of performance metrics
          "open_positions" → dict of positions still open at end
          "benchmark_curve"→ pd.Series (normalised benchmark), or None
        """
        symbols    = self.data_handler.get_symbols()
        start_date = self._config.data.start_date
        end_date   = self._config.data.end_date

        logger.info(
            "Backtest starting. Symbols: %s. Date range: %s → %s.",
            ", ".join(symbols), start_date, end_date,
        )

        if self._config.verbose:
            print(f"\n{'='*60}")
            print(f"  Strategy : {self.strategy.strategy_id}")
            print(f"  Symbols  : {', '.join(symbols)}")
            print(f"  Capital  : ₹{self._config.portfolio.initial_capital:,.0f}")
            print(f"{'='*60}\n")

        t_start = time.perf_counter()

        # ── MAIN LOOP ─────────────────────────────────────────────────────
        while self.data_handler.has_more_bars():
            self.data_handler.update_bars()

            # Drain the event queue completely for this bar
            while self.event_queue:
                event = self.event_queue.popleft()
                self._process_event(event)
                self._event_count += 1

            self._iteration_count += 1

            # End-of-bar equity snapshot
            current_dt = self.data_handler.get_current_datetime()
            if current_dt is not None:
                self.portfolio.update_equity_curve(current_dt)

            # Progress output every ~252 bars (roughly one trading year)
            if (
                self._config.verbose
                and self._iteration_count % 252 == 0
                and current_dt is not None
            ):
                eq = self.portfolio.get_final_equity()
                print(
                    f"  {current_dt.strftime('%Y')} processed. "
                    f"Portfolio value: ₹{eq:,.0f}"
                )

        # ── FINALISE ──────────────────────────────────────────────────────
        self._finalise()

        elapsed = time.perf_counter() - t_start
        if self._config.verbose:
            eq  = self._results["equity_curve"].iloc[-1] if not self._results["equity_curve"].empty else 0
            ret = eq / self._config.portfolio.initial_capital - 1.0
            print(f"\n{'='*60}")
            print(f"  Backtest complete in {elapsed:.2f}s")
            print(f"  Bars processed : {self._iteration_count}")
            print(f"  Events routed  : {self._event_count}")
            print(f"  Final equity   : ₹{eq:,.2f}")
            print(f"  Total return   : {ret:+.2%}")
            print(f"{'='*60}\n")

        logger.info(
            "Backtest finished: bars=%d, events=%d, elapsed=%.2fs",
            self._iteration_count, self._event_count, elapsed,
        )
        return self._results

    def _process_event(self, event) -> None:
        """Route a single event to the correct handler."""
        etype = event.event_type

        if etype == EventType.MARKET:
            self.strategy.calculate_signals(event, self.data_handler)

        elif etype == EventType.SIGNAL:
            self.portfolio.process_signal(event, self.data_handler)

        elif etype == EventType.ORDER:
            self.execution.execute_order(event, self.data_handler)

        elif etype == EventType.FILL:
            self.portfolio.update_portfolio_on_fill(event)

        else:
            logger.warning("Unrecognised event type: %s — skipped.", etype)

    def _finalise(self) -> None:
        """
        Compute final results after the main loop.

        Steps
        -----
        1. Close all open positions at the last available price (mark-to-market).
           This correctly credits unrealised PnL to the final equity.
        2. Build the equity curve, trade log, and benchmark curve.
        3. Run PerformanceMetrics to generate all metrics.
        4. Populate self._results.
        """
        # ── Mark-to-market open positions ─────────────────────────────────
        # Unrealised PnL is included in portfolio.equity already (via the
        # equity property). We record a final equity snapshot here.
        final_equity = self.portfolio.equity
        final_dt     = self.data_handler.get_current_datetime()
        if final_dt is not None:
            # Override the last equity curve entry to include mark-to-market
            if self.portfolio._equity_curve:
                self.portfolio._equity_curve[-1]["equity"] = final_equity

        # ── Equity curve & trade log ──────────────────────────────────────
        equity_curve = self.portfolio.calculate_equity_curve()
        trade_log    = self.portfolio.get_trade_log()

        # ── Benchmark curve (normalised to portfolio's start capital) ─────
        benchmark_curve = None
        if self.data_handler.benchmark_data is not None:
            bc = self.data_handler.benchmark_data["close"].copy()
            if len(bc) > 0 and not bc.isna().all():
                benchmark_curve = bc / bc.iloc[0] * self._config.portfolio.initial_capital

        # ── Performance metrics ───────────────────────────────────────────
        from performance.metrics import PerformanceMetrics
        perf    = PerformanceMetrics(self._config)
        metrics = perf.generate_summary_report(equity_curve, trade_log, benchmark_curve)

        if self._config.verbose:
            _print_metrics(metrics)

        # ── Store results ─────────────────────────────────────────────────
        self._results = {
            "equity_curve":    equity_curve,
            "trade_log":       trade_log,
            "metrics":         metrics,
            "open_positions":  dict(self.portfolio.open_positions),
            "benchmark_curve": benchmark_curve,
        }

    # ── Convenience accessors (kept for existing code that calls these) ───

    def get_equity_curve(self) -> pd.DataFrame:
        return self.portfolio.get_equity_curve()

    def get_trade_log(self) -> pd.DataFrame:
        return self.portfolio.get_trade_log()


# ──────────────────────────────────────────────────────────────────────────────
# FACTORY — convenience function that builds and wires all components
# ──────────────────────────────────────────────────────────────────────────────

def build_backtest_engine(
    config: BacktestConfig,
    strategy_cls: Type[BaseStrategy],
) -> BacktestEngine:
    """
    Convenience factory: build all components and return a ready BacktestEngine.

    This is the one-liner entry-point used by run_backtest.py.
    Individual components can also be built manually for more control.
    """
    data_handler = DataHandler(config, None)   # queue injected by BacktestEngine
    portfolio    = PortfolioManager(config, data_handler, None)
    execution    = ExecutionHandler(config, data_handler, None)
    strategy     = strategy_cls(config, None)

    return BacktestEngine(config, data_handler, strategy, portfolio, execution)


def _print_metrics(metrics: dict) -> None:
    """Print a compact metrics summary to stdout."""
    FMT = {
        "total_return":          ("Total Return",       "{:.2%}"),
        "cagr":                  ("CAGR",               "{:.2%}"),
        "annualised_volatility": ("Annual Volatility",  "{:.2%}"),
        "sharpe_ratio":          ("Sharpe Ratio",       "{:.3f}"),
        "sortino_ratio":         ("Sortino Ratio",      "{:.3f}"),
        "max_drawdown":          ("Max Drawdown",       "{:.2%}"),
        "max_dd_duration_days":  ("Max DD Duration",    "{:.0f} days"),
        "win_rate":              ("Win Rate",           "{:.2%}"),
        "profit_factor":         ("Profit Factor",      "{:.3f}"),
        "num_trades":            ("# Trades",           "{:.0f}"),
        "alpha":                 ("Alpha (annual)",     "{:.2%}"),
        "beta":                  ("Beta",               "{:.3f}"),
    }
    import math
    print("\n" + "─" * 50)
    print("  PERFORMANCE SUMMARY")
    print("─" * 50)
    for key, (label, fmt) in FMT.items():
        val = metrics.get(key)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            print(f"  {label:<25} N/A")
        else:
            try:
                print(f"  {label:<25} {fmt.format(val)}")
            except Exception:
                print(f"  {label:<25} {val}")
    print("─" * 50 + "\n")
