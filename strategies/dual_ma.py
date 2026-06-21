# strategies/dual_ma.py
# ============================================================
# DUAL MOVING AVERAGE CROSSOVER STRATEGY
# Signals LONG when fast MA crosses above slow MA (golden cross).
# Signals EXIT_LONG when fast MA crosses below slow MA (death cross).
# ============================================================

import logging
from collections import deque

from engine.events import MarketEvent, SignalDirection
from engine.strategy import BaseStrategy
from config import BacktestConfig

logger = logging.getLogger(__name__)


class DualMAStrategy(BaseStrategy):
    """
    Dual Moving Average Crossover strategy.

    Logic
    -----
    LONG entry    : fast MA crosses ABOVE slow MA (golden cross) while flat.
    EXIT_LONG     : fast MA crosses BELOW slow MA (death cross) while long.
    SHORT entry   : fast MA crosses BELOW slow MA while flat (if allow_short=True).
    EXIT_SHORT    : fast MA crosses ABOVE slow MA while short (if allow_short=True).

    Only one signal is emitted per symbol per bar. EXIT is always checked before ENTRY
    so an exit on the same bar as a new cross is not double-counted.

    Parameters (from StrategyConfig)
    ----------------------------------
    fast_ma_window : int  — fast period (default 20).
    slow_ma_window : int  — slow period (default 50). Must be > fast.
    ma_type        : str  — "SMA" or "EMA".
    allow_short    : bool — enable SHORT signals.
    """

    def __init__(self, config: BacktestConfig, event_queue: deque) -> None:
        super().__init__(config, event_queue)

        scfg = config.strategy
        self.fast_window  = scfg.fast_ma_window
        self.slow_window  = scfg.slow_ma_window
        self.ma_type      = scfg.ma_type
        self.allow_short  = scfg.allow_short

        if self.fast_window >= self.slow_window:
            raise ValueError(
                f"DualMAStrategy: fast_ma_window ({self.fast_window}) must be "
                f"< slow_ma_window ({self.slow_window})."
            )

        self.strategy_id  = f"DualMA_{self.ma_type}_{self.fast_window}_{self.slow_window}"
        # +2: need current bar AND previous bar for crossover detection
        self.required_bars = self.slow_window + 2

    def calculate_signals(self, event: MarketEvent, data_handler) -> None:
        """
        Compute MAs and emit a signal if a crossover occurred on this bar.

        Parameters
        ----------
        event        : MarketEvent for one symbol.
        data_handler : DataHandler to fetch historical bars.
        """
        bars = data_handler.get_latest_bars(event.symbol, self.required_bars)
        if not self._has_enough_bars(bars, self.required_bars):
            return

        prices = bars["close"]

        if self.ma_type == "SMA":
            fast_ma = self._calculate_sma(prices, self.fast_window)
            slow_ma = self._calculate_sma(prices, self.slow_window)
        else:
            fast_ma = self._calculate_ema(prices, self.fast_window)
            slow_ma = self._calculate_ema(prices, self.slow_window)

        symbol = event.symbol

        # ── EXIT checks first (always before entry) ───────────────────────
        if self._is_long(symbol) and self._crossed_below(fast_ma, slow_ma):
            self._emit_signal(symbol, event.timestamp, SignalDirection.EXIT_LONG)
            return

        if self._is_short(symbol) and self._crossed_above(fast_ma, slow_ma):
            self._emit_signal(symbol, event.timestamp, SignalDirection.EXIT_SHORT)
            return

        # ── ENTRY checks ─────────────────────────────────────────────────
        if self._is_flat(symbol):
            if self._crossed_above(fast_ma, slow_ma):
                self._emit_signal(symbol, event.timestamp, SignalDirection.LONG)
                return

            if self.allow_short and self._crossed_below(fast_ma, slow_ma):
                self._emit_signal(symbol, event.timestamp, SignalDirection.SHORT)
