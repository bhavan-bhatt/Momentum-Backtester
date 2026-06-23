# strategies/volatility_breakout.py
# ============================================================
# DONCHIAN CHANNEL / ATR VOLATILITY BREAKOUT
# ============================================================

import logging
from collections import deque
from typing import Dict, Set

import pandas as pd

from config import BacktestConfig
from engine.events import MarketEvent, SignalDirection
from engine.strategy import BaseStrategy

logger = logging.getLogger(__name__)


class VolatilityBreakoutStrategy(BaseStrategy):
    """
    Donchian channel breakout with ATR-based noise filtering.

    Channel bounds use only prior bars (anti-lookahead).
    """

    def __init__(
        self,
        config: BacktestConfig,
        event_queue: deque,
        regime_filter=None,
    ) -> None:
        super().__init__(config, event_queue)

        bcfg = config.advanced.breakout
        self.donchian_window = bcfg.donchian_window
        self.atr_period = bcfg.atr_period
        self.atr_breakout_multiplier = bcfg.atr_breakout_multiplier
        self.exit_method = bcfg.exit_method
        self.atr_stop_multiplier = config.strategy.atr_stop_multiplier

        self.entry_prices: Dict[str, float] = {}
        self._warned_symbols: Set[str] = set()
        self.required_bars = max(self.donchian_window, self.atr_period) + 3
        self.strategy_id = f"VolBreakout_DC{self.donchian_window}"
        self.set_regime_filter(regime_filter, "VolatilityBreakout")

    def calculate_signals(self, event: MarketEvent, data_handler) -> None:
        """Compute Donchian channels and ATR, check breakout/exit conditions."""
        symbol = event.symbol
        bars = data_handler.get_latest_bars(symbol, self.required_bars + 1)
        if not self._has_enough_bars(bars, self.required_bars + 1):
            return

        if not data_handler.has_high_low.get(symbol, False):
            if symbol not in self._warned_symbols:
                logger.warning(
                    "[%s] High/low data unavailable for %s — strategy skipped.",
                    self.strategy_id,
                    symbol,
                )
                self._warned_symbols.add(symbol)
            return

        if bars["high"].isna().all() or bars["low"].isna().all():
            return

        history = bars.iloc[:-1]
        current = bars.iloc[-1]

        upper_channel = (
            history["high"]
            .rolling(self.donchian_window, min_periods=self.donchian_window)
            .max()
            .iloc[-1]
        )
        lower_channel = (
            history["low"]
            .rolling(self.donchian_window, min_periods=self.donchian_window)
            .min()
            .iloc[-1]
        )

        atr_series = self._calculate_atr(
            bars["high"], bars["low"], bars["close"], self.atr_period
        )
        atr_now = self._get_latest_indicator_value(atr_series)

        if (
            pd.isna(upper_channel)
            or pd.isna(lower_channel)
            or atr_now is None
        ):
            return

        close = float(current["close"])

        if self._is_flat(symbol):
            breakout_distance = close - upper_channel
            if (
                close > upper_channel
                and breakout_distance > self.atr_breakout_multiplier * atr_now
            ):
                self.entry_prices[symbol] = close
                self._emit_signal(symbol, event.timestamp, SignalDirection.LONG)
            return

        if self._is_long(symbol):
            if self.exit_method == "opposite_channel":
                if close < lower_channel:
                    self.entry_prices.pop(symbol, None)
                    self._emit_signal(
                        symbol, event.timestamp, SignalDirection.EXIT_LONG
                    )
            elif self.exit_method == "atr_trailing":
                entry_price = self.entry_prices.get(symbol)
                if entry_price is not None:
                    stop_price = entry_price - self.atr_stop_multiplier * atr_now
                    if close < stop_price:
                        self.entry_prices.pop(symbol, None)
                        self._emit_signal(
                            symbol, event.timestamp, SignalDirection.EXIT_LONG
                        )
