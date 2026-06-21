# strategies/dual_ma.py
# ============================================================
# DUAL MOVING AVERAGE CROSSOVER STRATEGY
#
# Logic
# -----
# LONG  signal: fast MA crosses ABOVE slow MA (golden cross).
# EXIT  signal: fast MA crosses BELOW slow MA (death cross).
#
# The strategy tracks the previous bar's MA relationship to detect
# only the bar on which the crossover occurs, avoiding repeated signals.
# ============================================================

import logging
from collections import deque

import numpy as np

from engine.events import MarketEvent, SignalDirection
from engine.strategy import Strategy, moving_average
from engine.data_handler import DataHandler
from config import BacktestConfig

logger = logging.getLogger(__name__)


class DualMAStrategy(Strategy):
    """
    Dual Moving Average Crossover strategy.

    Configurable parameters (from StrategyConfig):
      fast_ma_window  — short MA period.
      slow_ma_window  — long MA period.
      ma_type         — "SMA" or "EMA".
      allow_short     — if True, emits SHORT on death cross.

    State per symbol (stored in self._state[symbol]):
      prev_fast, prev_slow — MA values from the previous bar.
      invested             — whether we hold a position.
    """

    def __init__(
        self,
        config: BacktestConfig,
        data_handler: DataHandler,
        event_queue: deque,
    ) -> None:
        super().__init__(config, data_handler, event_queue)
        scfg = config.strategy
        self._fast = scfg.fast_ma_window
        self._slow = scfg.slow_ma_window
        self._ma_type = scfg.ma_type

        if self._fast >= self._slow:
            raise ValueError(
                f"DualMA: fast_ma_window ({self._fast}) must be < "
                f"slow_ma_window ({self._slow})."
            )

        # Lookback needed: slow_window + 1 (for previous bar comparison)
        self._lookback = self._slow + 1

    @property
    def strategy_id(self) -> str:
        return f"DualMA_{self._ma_type}_{self._fast}_{self._slow}"

    def calculate_signals(self, event: MarketEvent) -> None:
        symbol = event.symbol
        bars   = self._get_bars(symbol, self._lookback)
        if bars is None:
            return

        close  = bars["close"]
        fast_s = moving_average(close, self._fast, self._ma_type)
        slow_s = moving_average(close, self._slow, self._ma_type)

        if len(fast_s) < 2 or len(slow_s) < 2:
            return

        fast_now  = fast_s.iloc[-1]
        slow_now  = slow_s.iloc[-1]
        fast_prev = fast_s.iloc[-2]
        slow_prev = slow_s.iloc[-2]

        # Guard against NaN values (insufficient history)
        if any(np.isnan(v) for v in [fast_now, slow_now, fast_prev, slow_prev]):
            return

        invested = self._is_invested(symbol)

        # ── Golden cross: fast crosses above slow ────────────────────────
        if fast_prev <= slow_prev and fast_now > slow_now:
            if not invested:
                self._emit_signal(event, SignalDirection.LONG)
                self._set_invested(symbol, True)

        # ── Death cross: fast crosses below slow ─────────────────────────
        elif fast_prev >= slow_prev and fast_now < slow_now:
            if invested:
                self._emit_signal(event, SignalDirection.EXIT_LONG)
                self._set_invested(symbol, False)

            if self._config.strategy.allow_short and not invested:
                self._emit_signal(event, SignalDirection.SHORT)
                self._set_invested(symbol, True)
