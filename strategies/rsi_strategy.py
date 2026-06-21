# strategies/rsi_strategy.py
# ============================================================
# RSI MEAN-REVERSION STRATEGY
#
# Logic
# -----
# LONG  signal: RSI drops below oversold threshold
#               (or crosses back above it if rsi_entry_on_reversal=True).
# EXIT  signal: RSI rises above overbought threshold.
#
# rsi_entry_on_reversal=True (default):
#   Enter only when RSI crosses *back* above oversold (reversal confirmation),
#   not when it first dips below — avoids catching falling knives.
#
# rsi_entry_on_reversal=False:
#   Enter as soon as RSI dips below the oversold level.
# ============================================================

import logging
from collections import deque

import numpy as np

from engine.events import MarketEvent, SignalDirection
from engine.strategy import Strategy, rsi as compute_rsi
from engine.data_handler import DataHandler
from config import BacktestConfig

logger = logging.getLogger(__name__)

# State keys for per-symbol tracking
_KEY_INVESTED      = "invested"
_KEY_WAS_OVERSOLD  = "was_oversold"


class RSIStrategy(Strategy):
    """
    RSI Mean-Reversion strategy.

    Configurable parameters (from StrategyConfig):
      rsi_period            — RSI look-back window (default 14).
      rsi_oversold          — Oversold threshold (default 30).
      rsi_overbought        — Overbought threshold (default 70).
      rsi_entry_on_reversal — True → enter on reversal cross; False → enter on touch.
      allow_short           — If True, SHORT when RSI crosses above overbought.
    """

    def __init__(
        self,
        config: BacktestConfig,
        data_handler: DataHandler,
        event_queue: deque,
    ) -> None:
        super().__init__(config, data_handler, event_queue)
        scfg = config.strategy
        self._period     = scfg.rsi_period
        self._oversold   = scfg.rsi_oversold
        self._overbought = scfg.rsi_overbought
        self._reversal   = scfg.rsi_entry_on_reversal

        # Need enough bars for RSI + 1 previous bar for crossover detection
        self._lookback = self._period * 3 + 1  # generous window for EWM warm-up

    @property
    def strategy_id(self) -> str:
        mode = "rev" if self._reversal else "touch"
        return f"RSI_{self._period}_{int(self._oversold)}_{int(self._overbought)}_{mode}"

    def calculate_signals(self, event: MarketEvent) -> None:
        symbol = event.symbol
        bars   = self._get_bars(symbol, self._lookback)
        if bars is None:
            return

        rsi_series = compute_rsi(bars["close"], self._period)
        if len(rsi_series) < 2:
            return

        rsi_now  = rsi_series.iloc[-1]
        rsi_prev = rsi_series.iloc[-2]

        if np.isnan(rsi_now) or np.isnan(rsi_prev):
            return

        invested     = self._is_invested(symbol)
        was_oversold = self._state.get(symbol, {}).get(_KEY_WAS_OVERSOLD, False)

        # ── Track oversold state ─────────────────────────────────────────
        if rsi_now < self._oversold:
            self._state.setdefault(symbol, {})[_KEY_WAS_OVERSOLD] = True
        elif rsi_now >= self._oversold:
            self._state.setdefault(symbol, {})[_KEY_WAS_OVERSOLD] = False

        # ── LONG entry ───────────────────────────────────────────────────
        if not invested:
            if self._reversal:
                # Enter when RSI crosses back above oversold (reversal confirmation)
                if rsi_prev < self._oversold <= rsi_now:
                    self._emit_signal(event, SignalDirection.LONG)
                    self._set_invested(symbol, True)
            else:
                # Enter as soon as RSI touches or drops below oversold
                if rsi_now <= self._oversold:
                    self._emit_signal(event, SignalDirection.LONG)
                    self._set_invested(symbol, True)

        # ── EXIT long when overbought ────────────────────────────────────
        elif invested:
            if rsi_now >= self._overbought:
                self._emit_signal(event, SignalDirection.EXIT_LONG)
                self._set_invested(symbol, False)

            # Also exit if RSI stays below mid-line for too long (momentum loss)
            # This prevents indefinite holding when RSI doesn't recover.
            # We use a simple heuristic: exit if RSI < 50 for 3 consecutive bars.
            # Tracked via a streak counter.
            streak = self._state.get(symbol, {}).get("below50_streak", 0)
            if rsi_now < 50:
                streak += 1
                self._state.setdefault(symbol, {})["below50_streak"] = streak
                if streak >= 30:  # ~6 trading weeks below mid-line
                    self._emit_signal(event, SignalDirection.EXIT_LONG)
                    self._set_invested(symbol, False)
                    self._state[symbol]["below50_streak"] = 0
            else:
                self._state.setdefault(symbol, {})["below50_streak"] = 0

        # ── SHORT entry (optional) ───────────────────────────────────────
        if self._config.strategy.allow_short and not invested:
            if rsi_prev > self._overbought >= rsi_now:
                self._emit_signal(event, SignalDirection.SHORT)
                self._set_invested(symbol, True)
