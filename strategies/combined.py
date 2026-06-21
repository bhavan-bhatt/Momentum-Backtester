# strategies/combined.py
# ============================================================
# COMBINED STRATEGY: MA TREND FILTER + RSI ENTRY + ATR EXIT
# Only takes RSI buy signals when the MA trend is upward.
# Three-condition exit: RSI overbought, MA death cross, ATR stop.
# ============================================================

import logging
from collections import deque
from typing import Optional

from engine.events import MarketEvent, SignalDirection
from engine.strategy import BaseStrategy
from config import BacktestConfig

logger = logging.getLogger(__name__)


class CombinedStrategy(BaseStrategy):
    """
    Production-quality multi-signal strategy.

    Entry (LONG)
    ------------
    Both conditions must be true simultaneously:
      1. MA trend filter: fast_ma > slow_ma (we are in an uptrend).
         Skipped if trend_filter_enabled=False.
      2. RSI reversal: RSI was below oversold on prev bar AND has now
         crossed back above the oversold threshold (Mode A entry).

    Exit (EXIT_LONG) — first condition that fires wins
    ---------------------------------------------------
    A. RSI overbought : RSI > rsi_overbought (profit target).
    B. Trend reversal : fast_ma crosses BELOW slow_ma (stop out).
    C. ATR trailing   : close < entry_price - (atr_stop_multiplier × entry_ATR).

    Signal Strength
    ---------------
    Computed as a blend of RSI depth and MA spread (see _calculate_signal_strength).
    Logged and stored on the SignalEvent but not used for sizing by default.

    Parameters (from StrategyConfig)
    ----------------------------------
    fast_ma_window, slow_ma_window, ma_type
    rsi_period, rsi_oversold, rsi_overbought
    atr_period, atr_stop_multiplier
    trend_filter_enabled
    """

    def __init__(self, config: BacktestConfig, event_queue: deque) -> None:
        super().__init__(config, event_queue)

        scfg = config.strategy
        self.fast_window         = scfg.fast_ma_window
        self.slow_window         = scfg.slow_ma_window
        self.ma_type             = scfg.ma_type
        self.rsi_period          = scfg.rsi_period
        self.rsi_oversold        = scfg.rsi_oversold
        self.rsi_overbought      = scfg.rsi_overbought
        self.atr_period          = scfg.atr_period
        self.atr_stop_multiplier = scfg.atr_stop_multiplier
        self.trend_filter        = scfg.trend_filter_enabled

        # Per-symbol entry state for ATR trailing stop
        self.entry_prices: dict = {}   # {symbol: float entry price}
        self.entry_atr:    dict = {}   # {symbol: float ATR at entry}

        # Warm-up requires the slowest indicator + extra buffer
        self.required_bars = max(self.slow_window, self.rsi_period, self.atr_period) + 3
        self.strategy_id   = "Combined_MA_RSI_ATR"

    def calculate_signals(self, event: MarketEvent, data_handler) -> None:
        """
        Evaluate MA+RSI entry and three-condition exit for this bar.

        Parameters
        ----------
        event        : MarketEvent for one symbol.
        data_handler : DataHandler to fetch historical bars.
        """
        bars = data_handler.get_latest_bars(event.symbol, self.required_bars)
        if not self._has_enough_bars(bars, self.required_bars):
            return

        prices = bars["close"]

        # ── Indicators ────────────────────────────────────────────────────
        if self.ma_type == "SMA":
            fast_ma = self._calculate_sma(prices, self.fast_window)
            slow_ma = self._calculate_sma(prices, self.slow_window)
        else:
            fast_ma = self._calculate_ema(prices, self.fast_window)
            slow_ma = self._calculate_ema(prices, self.slow_window)

        rsi_series = self._calculate_rsi(prices, self.rsi_period)

        has_hl  = data_handler.has_high_low.get(event.symbol, False)
        atr_val: Optional[float] = None
        if has_hl:
            atr_series = self._calculate_atr(
                bars["high"], bars["low"], prices, self.atr_period
            )
            atr_val = self._get_latest_indicator_value(atr_series)

        # Current scalar values
        fast_now = self._get_latest_indicator_value(fast_ma)
        slow_now = self._get_latest_indicator_value(slow_ma)
        rsi_now  = self._get_latest_indicator_value(rsi_series, lookback=1)
        rsi_prev = self._get_latest_indicator_value(rsi_series, lookback=2)

        if None in (fast_now, slow_now, rsi_now, rsi_prev):
            return

        symbol    = event.symbol
        is_uptrend = fast_now > slow_now

        # ── ENTRY LOGIC ───────────────────────────────────────────────────
        if self._is_flat(symbol):
            rsi_reversal = (rsi_prev < self.rsi_oversold <= rsi_now)
            trend_ok     = (not self.trend_filter) or is_uptrend

            if rsi_reversal and trend_ok:
                strength = self._calculate_signal_strength(rsi_now, fast_now, slow_now)
                self.entry_prices[symbol] = event.close
                self.entry_atr[symbol]    = atr_val if atr_val is not None else event.close * 0.02
                self._emit_signal(symbol, event.timestamp, SignalDirection.LONG, strength)

        # ── EXIT LOGIC ────────────────────────────────────────────────────
        elif self._is_long(symbol):
            exit_triggered = False
            reason         = ""

            # Condition A — RSI overbought
            if rsi_now > self.rsi_overbought:
                exit_triggered = True
                reason = "RSI_OB"

            # Condition B — MA death cross (trend reversal)
            if not exit_triggered and self._crossed_below(fast_ma, slow_ma):
                exit_triggered = True
                reason = "MA_CROSS"

            # Condition C — ATR trailing stop
            if not exit_triggered and symbol in self.entry_prices:
                entry_atr_val = self.entry_atr.get(symbol, atr_val or event.close * 0.02)
                stop_price    = self.entry_prices[symbol] - (
                    self.atr_stop_multiplier * entry_atr_val
                )
                if event.close < stop_price:
                    exit_triggered = True
                    reason = "ATR_STOP"

            if exit_triggered:
                logger.debug(
                    "[%s] EXIT %s: reason=%s", self.strategy_id, symbol, reason
                )
                self.entry_prices.pop(symbol, None)
                self.entry_atr.pop(symbol, None)
                self._emit_signal(symbol, event.timestamp, SignalDirection.EXIT_LONG)

    def _calculate_signal_strength(
        self, rsi_now: float, fast_ma: float, slow_ma: float
    ) -> float:
        """
        Compute a confidence score ∈ [0, 1].

        RSI component  (weight 0.7):
          Depth below the oversold level at time of reversal.
          0 when RSI is exactly AT oversold; higher when deeper into oversold.

        Trend component (weight 0.3, only when trend filter is on):
          Normalised spread between fast and slow MA.
          5% spread → full strength (1.0).
        """
        rsi_depth      = max(0.0, self.rsi_oversold - rsi_now) / self.rsi_oversold
        trend_strength = 0.0
        if self.trend_filter and slow_ma != 0:
            ma_spread      = (fast_ma - slow_ma) / slow_ma
            trend_strength = min(1.0, max(0.0, ma_spread * 20.0))

        strength = 0.7 * rsi_depth + 0.3 * trend_strength
        return float(max(0.0, min(1.0, strength)))
