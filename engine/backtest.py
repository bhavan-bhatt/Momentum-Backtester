# engine/backtest.py
# ============================================================
# MAIN EVENT LOOP — ORCHESTRATES ALL MODULES
# The Backtest class wires together DataHandler, Strategy,
# Portfolio, and Execution into a single simulation run.
# ============================================================

import logging
import time
from collections import deque
from datetime import datetime
from typing import List, Type

from engine.events import EventType, MarketEvent, SignalEvent, OrderEvent, FillEvent
from engine.data_handler import DataHandler
from engine.strategy import Strategy
from engine.portfolio import PortfolioManager
from engine.execution import ExecutionHandler
from config import BacktestConfig

logger = logging.getLogger(__name__)


class Backtest:
    """
    Orchestrates a full historical simulation using an event-driven loop.

    Event Flow
    ----------
    Each time step:
      1. DataHandler.update_bars()
         → Pushes N × MarketEvent (one per symbol) into the queue.

      2. For each MarketEvent:
         → Strategy.calculate_signals(event)
            → May push SignalEvent(s) into the queue.

      3. For each SignalEvent:
         → Portfolio.on_signal(event)
            → May push OrderEvent(s) into the queue.

      4. For each OrderEvent:
         → Execution.on_order(event)
            → Pushes FillEvent into the queue.

      5. For each FillEvent:
         → Portfolio.on_fill(event)
            → Updates cash and holdings.

      6. Portfolio.update_equity_curve(timestamp)
         → Snapshots portfolio value at end of time step.

    Parameters
    ----------
    config       : BacktestConfig
    strategy_cls : Strategy subclass (the class itself, not an instance)
    """

    def __init__(
        self,
        config: BacktestConfig,
        strategy_cls: Type[Strategy],
    ) -> None:
        self._config = config
        self._event_queue: deque = deque()

        # ── Wire up modules ───────────────────────────────────────────────
        self.data_handler = DataHandler(config, self._event_queue)
        self.portfolio    = PortfolioManager(config, self.data_handler, self._event_queue)
        self.execution    = ExecutionHandler(config, self.data_handler, self._event_queue)
        self.strategy     = strategy_cls(config, self.data_handler, self._event_queue)

        self._start_time: float = 0.0
        self._bars_processed: int = 0

    # ──────────────────────────────────────────────────────────────────────
    # MAIN RUN
    # ──────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Execute the full backtest simulation.

        Iterates bar-by-bar until the DataHandler is exhausted,
        draining the event queue at each step.
        """
        self._start_time = time.perf_counter()
        cfg = self._config

        if cfg.verbose:
            print(f"\n{'='*60}")
            print(f"  Strategy : {self.strategy.strategy_id}")
            print(f"  Symbols  : {', '.join(self.data_handler.get_symbols())}")
            print(f"  Capital  : ₹{cfg.portfolio.initial_capital:,.0f}")
            print(f"{'='*60}\n")

        while self.data_handler.has_more_bars():
            # Advance the data source by one bar
            self.data_handler.update_bars()
            self._bars_processed += 1

            # Drain the entire event queue for this time step
            while self._event_queue:
                event = self._event_queue.popleft()
                self._route_event(event)

            # Snapshot equity after all events for this bar are processed
            current_dt = self.data_handler.get_current_datetime()
            if current_dt is not None:
                self.portfolio.update_equity_curve(current_dt)

        elapsed = time.perf_counter() - self._start_time
        final_equity = self.portfolio.get_final_equity()
        ret_pct = (final_equity / cfg.portfolio.initial_capital - 1.0) * 100.0

        if cfg.verbose:
            print(f"\n{'='*60}")
            print(f"  Backtest complete in {elapsed:.2f}s")
            print(f"  Bars processed : {self._bars_processed}")
            print(f"  Final equity   : ₹{final_equity:,.2f}")
            print(f"  Total return   : {ret_pct:+.2f}%")
            print(f"{'='*60}\n")

        logger.info(
            "Backtest finished: bars=%d, equity=%.2f, return=%.2f%%, elapsed=%.2fs",
            self._bars_processed, final_equity, ret_pct, elapsed,
        )

    def _route_event(self, event) -> None:
        """Dispatch an event to the correct handler based on its type."""
        etype = event.event_type

        if etype == EventType.MARKET:
            self.strategy.calculate_signals(event)

        elif etype == EventType.SIGNAL:
            self.portfolio.on_signal(event)

        elif etype == EventType.ORDER:
            self.execution.on_order(event)

        elif etype == EventType.FILL:
            self.portfolio.on_fill(event)

        else:
            logger.warning("Unknown event type: %s", etype)

    # ──────────────────────────────────────────────────────────────────────
    # RESULT ACCESSORS
    # ──────────────────────────────────────────────────────────────────────

    def get_equity_curve(self):
        """Return the portfolio equity curve DataFrame."""
        return self.portfolio.get_equity_curve()

    def get_trade_log(self):
        """Return the full trade log DataFrame."""
        return self.portfolio.get_trade_log()
