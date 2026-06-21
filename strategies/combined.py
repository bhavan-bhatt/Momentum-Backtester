# strategies/combined.py
# ============================================================
# COMBINED STRATEGY — MA TREND FILTER + RSI ENTRY + VOL SIZING
#
# Logic
# -----
# 1. TREND FILTER (when trend_filter_enabled=True):
#    Only permit LONG entries when fast MA > slow MA (uptrend).
#
# 2. RSI ENTRY (mean-reversion within a trend):
#    Enter LONG when RSI crosses back above the oversold level
#    (same reversal logic as RSIStrategy, but gated by trend).
#
# 3. ATR EXIT (volatility-adjusted stop):
#    Exit when close falls below: entry_price - atr_stop_multiplier × ATR.
#    Also exit when RSI reaches overbought or MA death cross occurs.
#
# 4. SIGNAL STRENGTH:
#    Strength is set to (1 - RSI/100) so deeply oversold entries get
#    higher confidence weight (used for analysis; not sizing by default).
# ============================================================

import logging
from collections import deque
from typing import Optional

import numpy as np

from engine.events import MarketEvent, SignalDirection
from engine.strategy import Strategy, moving_average, rsi as compute_rsi, atr as compute_atr
from engine.data_handler import DataHandler
from config import BacktestConfig

logger = logging.getLogger(__name__)


class CombinedStrategy(Strategy):
    """
    Three-layer strategy: MA trend filter + RSI entry + ATR/RSI exit.

    State tracked per symbol:
      invested       : bool
      entry_price    : float — price at which we entered.
      entry_atr      : float — ATR at time of entry (for dynamic stop).
      was_oversold   : bool  — RSI was below oversold threshold.
    """

    def __init__(
        self,
        config: BacktestConfig,
        data_handler: DataHandler,
        event_queue: deque,
    ) -> None:
        super().__init__(config, data_handler, event_queue)
        scfg = config.strategy
        self._fast      = scfg.fast_ma_window
        self._slow      = scfg.slow_ma_window
        self._ma_type   = scfg.ma_type
        self._rsi_p     = scfg.rsi_period
        self._oversold  = scfg.rsi_oversold
        self._overbought = scfg.rsi_overbought
        self._atr_p     = scfg.atr_period
        self._atr_mult  = scfg.atr_stop_multiplier
        self._trend_on  = scfg.trend_filter_enabled

        # Lookback: enough for slow MA + ATR warm-up
        self._lookback  = max(self._slow, self._atr_p) * 3 + 2

    @property
    def strategy_id(self) -> str:
        return (
            f"Combined_{self._ma_type}{self._fast}_{self._slow}"
            f"_RSI{self._rsi_p}_ATR{self._atr_p}"
        )

    def calculate_signals(self, event: MarketEvent) -> None:
        symbol = event.symbol
        bars   = self._get_bars(symbol, self._lookback)
        if bars is None:
            return

        close  = bars["close"]
        has_hl = self._data.has_high_low.get(symbol, False)

        # ── Compute indicators ───────────────────────────────────────────
        fast_ma  = moving_average(close, self._fast, self._ma_type)
        slow_ma  = moving_average(close, self._slow, self._ma_type)
        rsi_s    = compute_rsi(close, self._rsi_p)

        atr_val: Optional[float] = None
        if has_hl:
            atr_s   = compute_atr(bars["high"], bars["low"], close, self._atr_p)
            atr_now = atr_s.iloc[-1]
            atr_val = None if np.isnan(atr_now) else float(atr_now)

        # Current values
        fast_now  = fast_ma.iloc[-1]
        slow_now  = slow_ma.iloc[-1]
        fast_prev = fast_ma.iloc[-2] if len(fast_ma) >= 2 else np.nan
        slow_prev = slow_ma.iloc[-2] if len(slow_ma) >= 2 else np.nan
        rsi_now   = rsi_s.iloc[-1]
        rsi_prev  = rsi_s.iloc[-2] if len(rsi_s) >= 2 else np.nan
        close_now = float(close.iloc[-1])

        # Guard NaN
        if any(np.isnan(v) for v in [fast_now, slow_now, rsi_now]):
            return

        invested = self._is_invested(symbol)
        sym_state = self._state.setdefault(symbol, {
            "invested": False,
            "entry_price": 0.0,
            "entry_atr": 0.0,
            "was_oversold": False,
        })

        # ── Update oversold tracker ──────────────────────────────────────
        if not np.isnan(rsi_now):
            sym_state["was_oversold"] = rsi_now < self._oversold

        # ── Trend filter: fast MA > slow MA means uptrend ────────────────
        in_uptrend = fast_now > slow_now

        # ── ENTRY LOGIC ──────────────────────────────────────────────────
        if not invested:
            trend_ok = (not self._trend_on) or in_uptrend

            if trend_ok:
                # RSI reversal entry: RSI was below oversold, now crosses above
                if (
                    not np.isnan(rsi_prev)
                    and rsi_prev < self._oversold
                    and rsi_now >= self._oversold
                ):
                    strength = max(0.0, min(1.0, 1.0 - rsi_prev / 100.0))
                    self._emit_signal(event, SignalDirection.LONG, strength)
                    sym_state["invested"]    = True
                    sym_state["entry_price"] = close_now
                    sym_state["entry_atr"]   = atr_val if atr_val is not None else close_now * 0.02
                    self._set_invested(symbol, True)

        # ── EXIT LOGIC ───────────────────────────────────────────────────
        else:
            entry_price = sym_state.get("entry_price", 0.0)
            entry_atr   = sym_state.get("entry_atr", 0.0)

            exit_triggered = False

            # 1. ATR stop-loss: price fell below entry - (multiplier × ATR)
            if entry_atr > 0:
                stop_level = entry_price - self._atr_mult * entry_atr
                if close_now < stop_level:
                    logger.debug(
                        "[%s] ATR stop hit for %s: close=%.2f < stop=%.2f",
                        self.strategy_id, symbol, close_now, stop_level,
                    )
                    exit_triggered = True

            # 2. RSI overbought exit
            if not exit_triggered and rsi_now >= self._overbought:
                exit_triggered = True

            # 3. Trend reversal exit: MA death cross (trend filter must be enabled)
            if not exit_triggered and self._trend_on:
                if (
                    not np.isnan(fast_prev) and not np.isnan(slow_prev)
                    and fast_prev >= slow_prev
                    and fast_now < slow_now
                ):
                    exit_triggered = True

            if exit_triggered:
                self._emit_signal(event, SignalDirection.EXIT_LONG)
                sym_state["invested"]    = False
                sym_state["entry_price"] = 0.0
                sym_state["entry_atr"]   = 0.0
                self._set_invested(symbol, False)
