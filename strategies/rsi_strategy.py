# strategies/rsi_strategy.py
# ============================================================
# RSI ENTRY / EXIT STRATEGY
# Buys when RSI crosses out of oversold. Sells when crosses into overbought.
# ============================================================

import logging
from collections import deque

from engine.events import MarketEvent, SignalDirection
from engine.strategy import BaseStrategy
from config import BacktestConfig

logger = logging.getLogger(__name__)


class RSIStrategy(BaseStrategy):
    """
    RSI mean-reversion strategy.

    Entry Modes (controlled by rsi_entry_on_reversal)
    --------------------------------------------------
    MODE A — Reversal (rsi_entry_on_reversal=True, RECOMMENDED):
      BUY       : RSI was below oversold on prev bar AND has now risen back above it.
                  Confirms the oversold condition is resolving before entering.
      EXIT_LONG : RSI rises above overbought threshold.

    MODE B — Threshold (rsi_entry_on_reversal=False, simpler):
      BUY       : RSI drops below the oversold level.
      EXIT_LONG : RSI rises above the overbought level.

    Parameters (from StrategyConfig)
    ---------------------------------
    rsi_period            : int   — lookback (default 14).
    rsi_oversold          : float — oversold level (default 30).
    rsi_overbought        : float — overbought level (default 70).
    rsi_entry_on_reversal : bool  — Mode A (True) or Mode B (False).
    """

    def __init__(self, config: BacktestConfig, event_queue: deque) -> None:
        super().__init__(config, event_queue)

        scfg = config.strategy
        self.rsi_period       = scfg.rsi_period
        self.oversold         = scfg.rsi_oversold
        self.overbought       = scfg.rsi_overbought
        self.entry_on_reversal = scfg.rsi_entry_on_reversal

        if not (self.oversold < 50 < self.overbought and self.oversold < self.overbought):
            raise ValueError(
                f"RSIStrategy: Invalid thresholds — oversold={self.oversold}, "
                f"overbought={self.overbought}. Need oversold < 50 < overbought."
            )

        self.strategy_id   = (
            f"RSI_{self.rsi_period}_{int(self.oversold)}_{int(self.overbought)}"
        )
        # +2: need rsi_now (iloc[-1]) and rsi_prev (iloc[-2])
        self.required_bars = self.rsi_period + 2

    def calculate_signals(self, event: MarketEvent, data_handler) -> None:
        """
        Compute RSI and check entry/exit conditions for this bar.

        Parameters
        ----------
        event        : MarketEvent for one symbol.
        data_handler : DataHandler to fetch historical bars.
        """
        bars = data_handler.get_latest_bars(event.symbol, self.required_bars)
        if not self._has_enough_bars(bars, self.required_bars):
            return

        rsi_series = self._calculate_rsi(bars["close"], self.rsi_period)

        rsi_now  = self._get_latest_indicator_value(rsi_series, lookback=1)
        rsi_prev = self._get_latest_indicator_value(rsi_series, lookback=2)

        if rsi_now is None or rsi_prev is None:
            return

        symbol = event.symbol

        # ── LONG ENTRY ────────────────────────────────────────────────────
        if self._is_flat(symbol):
            if self.entry_on_reversal:
                # Mode A: RSI was below oversold and just crossed back above it
                if rsi_prev < self.oversold <= rsi_now:
                    self._emit_signal(symbol, event.timestamp, SignalDirection.LONG)
            else:
                # Mode B: RSI is currently below oversold
                if rsi_now < self.oversold:
                    self._emit_signal(symbol, event.timestamp, SignalDirection.LONG)

        # ── EXIT LONG ─────────────────────────────────────────────────────
        elif self._is_long(symbol):
            if rsi_now > self.overbought:
                self._emit_signal(symbol, event.timestamp, SignalDirection.EXIT_LONG)
