# engine/strategy.py
# ============================================================
# ABSTRACT BASE CLASS FOR ALL STRATEGIES
# Provides shared indicator utility functions used by all strategy implementations.
# ============================================================

import logging
from abc import ABC, abstractmethod
from collections import deque
from typing import Optional

import numpy as np
import pandas as pd

from engine.events import MarketEvent, SignalEvent, SignalDirection
from config import BacktestConfig

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """
    Abstract base class that all strategies must inherit from.

    Provides
    --------
    1. Abstract method calculate_signals() that every strategy must implement.
    2. Protected indicator utility methods (SMA, EMA, RSI, ATR, crossover detection).
    3. Position state tracking per symbol (strategy-level approximation only;
       Portfolio is the authoritative source of truth for actual positions).

    Internal State
    --------------
    self.current_positions : Dict[str, Optional[SignalDirection]]
        {symbol: SignalDirection.LONG | SignalDirection.SHORT | None}
        None means flat.
    """

    def __init__(self, config: BacktestConfig, event_queue: deque) -> None:
        self._config      = config
        self._event_queue = event_queue
        self.strategy_id  = "BaseStrategy"
        # Keyed by symbol; None = flat, SignalDirection = open direction
        self.current_positions: dict = {}
        self._regime_filter = None
        self._regime_name: Optional[str] = None
        self.audit_log = None

    def set_regime_filter(self, regime_filter, strategy_name: str) -> None:
        """Attach optional regime filter for entry gating."""
        self._regime_filter = regime_filter
        self._regime_name = strategy_name

    def set_audit_log(self, audit_log) -> None:
        """Attach optional audit log for structured decision recording."""
        self.audit_log = audit_log

    # ──────────────────────────────────────────────────────────────────────
    # ABSTRACT INTERFACE
    # ──────────────────────────────────────────────────────────────────────

    @abstractmethod
    def calculate_signals(self, event: MarketEvent, data_handler) -> None:
        """
        Core method — analyse the current bar and emit SignalEvents.

        Contract
        --------
        - Receives a MarketEvent for ONE symbol at ONE timestamp.
        - May call data_handler.get_latest_bars(event.symbol, N) to get history.
        - If a trade signal is detected, calls self._emit_signal().
        - MUST NOT look at future data.
        - MUST NOT modify data_handler's internal state.

        Parameters
        ----------
        event        : MarketEvent — the bar triggering this call.
        data_handler : DataHandler — for fetching historical bars.
        """
        ...

    # ──────────────────────────────────────────────────────────────────────
    # SIGNAL EMISSION
    # ──────────────────────────────────────────────────────────────────────

    def _emit_signal(
        self,
        symbol: str,
        timestamp,
        direction: SignalDirection,
        strength: float = 1.0,
        reason: str = "",
        values: Optional[dict] = None,
    ) -> None:
        """
        Build a SignalEvent, update position state, and push into the event queue.

        Parameters
        ----------
        symbol    : str
        timestamp : datetime
        direction : SignalDirection
        strength  : float ∈ [0, 1]
        reason    : Optional human-readable explanation for audit log
        values    : Optional numeric values dict for audit log
        """
        if direction in (SignalDirection.LONG, SignalDirection.SHORT):
            if self._regime_filter is not None and self._regime_name:
                if not self._regime_filter.is_active(self._regime_name):
                    logger.debug(
                        "[%s] Entry blocked — regime inactive for %s.",
                        self.strategy_id,
                        self._regime_name,
                    )
                    return

        signal = SignalEvent(
            timestamp=timestamp,
            symbol=symbol,
            strategy_id=self.strategy_id,
            direction=direction,
            strength=max(0.0, min(1.0, strength)),
        )
        # Position state is updated on fill (on_fill), not here, so failed
        # orders (qty=0, blocked by portfolio) do not desync strategy state.

        self._event_queue.append(signal)

        if self.audit_log is not None:
            self.audit_log.record(
                timestamp=timestamp,
                event_type="SIGNAL",
                symbol=symbol,
                strategy_id=self.strategy_id,
                reason=reason or f"{direction.value} signal",
                values=values or {},
            )

        logger.debug(
            "[%s] %s → %s @ %s (strength=%.2f)",
            self.strategy_id, symbol, direction.value,
            timestamp.date() if hasattr(timestamp, "date") else timestamp,
            strength,
        )

    def on_fill(self, fill: "FillEvent") -> None:
        """Update strategy-level position tracking after a confirmed fill."""
        from engine.events import FillEvent, OrderDirection

        if not isinstance(fill, FillEvent):
            return
        if fill.order_ref is None or fill.order_ref.signal_ref is None:
            return

        symbol = fill.symbol
        sig_dir = fill.order_ref.signal_ref.direction

        if fill.direction == OrderDirection.BUY:
            if sig_dir == SignalDirection.LONG:
                self.current_positions[symbol] = SignalDirection.LONG
            elif sig_dir == SignalDirection.EXIT_SHORT:
                self.current_positions[symbol] = None
        elif fill.direction == OrderDirection.SELL:
            if sig_dir == SignalDirection.EXIT_LONG:
                self.current_positions[symbol] = None
            elif sig_dir == SignalDirection.SHORT:
                self.current_positions[symbol] = SignalDirection.SHORT

    # ──────────────────────────────────────────────────────────────────────
    # POSITION STATE HELPERS
    # ──────────────────────────────────────────────────────────────────────

    def _is_flat(self, symbol: str) -> bool:
        """True if no open position exists for this symbol."""
        return self.current_positions.get(symbol) is None

    def _is_long(self, symbol: str) -> bool:
        """True if a LONG position is open for this symbol."""
        return self.current_positions.get(symbol) == SignalDirection.LONG

    def _is_short(self, symbol: str) -> bool:
        """True if a SHORT position is open for this symbol."""
        return self.current_positions.get(symbol) == SignalDirection.SHORT

    def _has_enough_bars(self, bars: Optional[pd.DataFrame], required: int) -> bool:
        """
        Return True if bars is non-None and has at least `required` rows.
        Always call this before computing indicators to avoid warm-up errors.
        """
        return bars is not None and len(bars) >= required

    # ──────────────────────────────────────────────────────────────────────
    # INDICATOR UTILITY FUNCTIONS
    # ──────────────────────────────────────────────────────────────────────

    def _calculate_sma(self, prices: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average. First (window-1) values are NaN."""
        return prices.rolling(window=window, min_periods=window).mean()

    def _calculate_ema(self, prices: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average (span=window, adjust=False)."""
        return prices.ewm(span=window, min_periods=window, adjust=False).mean()

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """
        RSI using Wilder's smoothing (EMA with alpha = 1/period).

        Returns a Series in [0, 100]. First `period` values are NaN.
        Handles the edge case of zero avg_loss (returns 100).
        """
        delta = prices.diff()
        gains = delta.clip(lower=0)
        losses = (-delta).clip(lower=0)

        avg_gain = gains.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = losses.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        # Avoid division by zero: where avg_loss is 0, RS is infinite → RSI = 100
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        # Fill the zero-avg_loss edge case
        rsi = rsi.where(avg_loss != 0, 100.0)
        return rsi

    def _calculate_atr(
        self,
        highs: pd.Series,
        lows: pd.Series,
        closes: pd.Series,
        period: int,
    ) -> pd.Series:
        """
        Average True Range using Wilder's smoothing (alpha = 1/period).

        If highs or lows are entirely NaN (data tier 3), returns a Series of NaN
        so callers can detect unavailability and fall back to other sizing.
        """
        if highs.isna().all() or lows.isna().all():
            return pd.Series(np.nan, index=closes.index)

        prev_close = closes.shift(1)
        tr = pd.concat(
            [
                highs - lows,
                (highs - prev_close).abs(),
                (lows  - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    def _get_latest_indicator_value(
        self, series: pd.Series, lookback: int = 1
    ) -> Optional[float]:
        """
        Safely return the most recent (or Nth-most-recent) scalar from a Series.

        Parameters
        ----------
        series   : pd.Series
        lookback : 1 = latest, 2 = one bar ago, etc.

        Returns
        -------
        float or None if the value is NaN or the series is too short.
        """
        if series is None or len(series) < lookback:
            return None
        val = series.iloc[-lookback]
        return None if pd.isna(val) else float(val)

    def _crossed_above(self, fast: pd.Series, slow: pd.Series) -> bool:
        """
        True if fast crossed ABOVE slow on the most recent bar (bullish crossover).

        Requires at least 2 rows in both series. Returns False on any NaN.
        """
        if len(fast) < 2 or len(slow) < 2:
            return False
        fast_now, fast_prev = fast.iloc[-1], fast.iloc[-2]
        slow_now, slow_prev = slow.iloc[-1], slow.iloc[-2]
        if any(pd.isna(v) for v in [fast_now, fast_prev, slow_now, slow_prev]):
            return False
        return fast_prev <= slow_prev and fast_now > slow_now

    def _crossed_below(self, fast: pd.Series, slow: pd.Series) -> bool:
        """
        True if fast crossed BELOW slow on the most recent bar (bearish crossover).

        Requires at least 2 rows in both series. Returns False on any NaN.
        """
        if len(fast) < 2 or len(slow) < 2:
            return False
        fast_now, fast_prev = fast.iloc[-1], fast.iloc[-2]
        slow_now, slow_prev = slow.iloc[-1], slow.iloc[-2]
        if any(pd.isna(v) for v in [fast_now, fast_prev, slow_now, slow_prev]):
            return False
        return fast_prev >= slow_prev and fast_now < slow_now


# ── Module-level indicator functions (kept for backward compatibility) ──────────
# These thin wrappers delegate to a throw-away strategy instance so that
# code importing from engine.strategy as free functions still works.

def _make_indicator_proxy():
    """Create a minimal concrete subclass just to expose indicator functions."""
    class _Proxy(BaseStrategy):
        def calculate_signals(self, event, data_handler):
            pass
    proxy = _Proxy.__new__(_Proxy)
    proxy._config = None
    proxy._event_queue = None
    proxy.strategy_id = ""
    proxy.current_positions = {}
    return proxy


_proxy = _make_indicator_proxy()


def sma(series: pd.Series, window: int) -> pd.Series:
    return _proxy._calculate_sma(series, window)


def ema(series: pd.Series, window: int) -> pd.Series:
    return _proxy._calculate_ema(series, window)


def moving_average(series: pd.Series, window: int, ma_type: str = "SMA") -> pd.Series:
    t = ma_type.upper()
    if t == "SMA":
        return sma(series, window)
    elif t == "EMA":
        return ema(series, window)
    raise ValueError(f"Unknown ma_type '{ma_type}'. Use 'SMA' or 'EMA'.")


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    return _proxy._calculate_rsi(series, period)


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    return _proxy._calculate_atr(high, low, close, period)


def bollinger_bands(
    series: pd.Series, window: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    middle = sma(series, window)
    std    = series.rolling(window=window, min_periods=window).std()
    return pd.DataFrame({
        "upper":  middle + num_std * std,
        "middle": middle,
        "lower":  middle - num_std * std,
    })


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    fast_ema   = ema(series, fast)
    slow_ema   = ema(series, slow)
    macd_line  = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    return pd.DataFrame({
        "macd":   macd_line,
        "signal": signal_line,
        "hist":   macd_line - signal_line,
    })
