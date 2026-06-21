# engine/strategy.py
# ============================================================
# ABSTRACT STRATEGY BASE CLASS + STATELESS INDICATOR UTILITIES
# All concrete strategies inherit from Strategy and implement
# calculate_signals(). Indicators are pure functions that
# accept pd.Series / pd.DataFrame and return pd.Series.
# ============================================================

import logging
from abc import ABC, abstractmethod
from collections import deque
from typing import Optional

import numpy as np
import pandas as pd

from engine.events import MarketEvent, SignalEvent, SignalDirection
from engine.data_handler import DataHandler
from config import BacktestConfig

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# STATELESS INDICATOR UTILITIES
# All functions accept a pd.Series of prices (or DataFrame for multi-column
# indicators) and return a pd.Series. No side-effects.
# ══════════════════════════════════════════════════════════════════════════════

def sma(series: pd.Series, window: int) -> pd.Series:
    """
    Simple Moving Average.

    Parameters
    ----------
    series : pd.Series  — price series (typically close).
    window : int        — look-back period.

    Returns
    -------
    pd.Series of the same length; first (window-1) values are NaN.
    """
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    """
    Exponential Moving Average using pandas' EWM (span=window).

    min_periods=window ensures the first value appears only after
    `window` data points are available (consistent with SMA behaviour).

    Parameters
    ----------
    series : pd.Series
    window : int

    Returns
    -------
    pd.Series
    """
    return series.ewm(span=window, min_periods=window, adjust=False).mean()


def moving_average(series: pd.Series, window: int, ma_type: str = "SMA") -> pd.Series:
    """
    Dispatch to sma() or ema() based on ma_type string.

    Parameters
    ----------
    series  : pd.Series
    window  : int
    ma_type : str — "SMA" or "EMA" (case-insensitive).

    Returns
    -------
    pd.Series

    Raises
    ------
    ValueError if ma_type is not recognised.
    """
    t = ma_type.upper()
    if t == "SMA":
        return sma(series, window)
    elif t == "EMA":
        return ema(series, window)
    else:
        raise ValueError(f"Unknown ma_type '{ma_type}'. Use 'SMA' or 'EMA'.")


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (Wilder's smoothing method).

    RSI = 100 - (100 / (1 + RS))
    RS  = avg_gain / avg_loss  (over `period` bars)

    Uses exponential Wilder smoothing (equivalent to EMA with alpha=1/period).

    Parameters
    ----------
    series : pd.Series — typically close prices.
    period : int       — look-back window (standard: 14).

    Returns
    -------
    pd.Series — values in [0, 100]; first (period) values are NaN.
    """
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)

    # Wilder smoothing: alpha = 1 / period
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100.0 - (100.0 / (1.0 + rs))
    return rsi_val


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Average True Range (Wilder smoothing).

    True Range = max(H-L, |H-Prev_C|, |L-Prev_C|)
    ATR = EWM mean of TR with alpha = 1/period.

    Parameters
    ----------
    high, low, close : pd.Series — aligned OHLC columns.
    period : int                  — look-back (standard: 14).

    Returns
    -------
    pd.Series — ATR values; first (period) values are NaN.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def bollinger_bands(
    series: pd.Series, window: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """
    Bollinger Bands.

    Returns a DataFrame with columns: ['upper', 'middle', 'lower'].

    Parameters
    ----------
    series  : pd.Series — price series.
    window  : int       — rolling window for mean and std.
    num_std : float     — number of standard deviations for the bands.
    """
    middle = sma(series, window)
    std    = series.rolling(window=window, min_periods=window).std()
    upper  = middle + num_std * std
    lower  = middle - num_std * std
    return pd.DataFrame({"upper": upper, "middle": middle, "lower": lower})


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    MACD (Moving Average Convergence Divergence).

    Returns a DataFrame with columns: ['macd', 'signal', 'hist'].
    """
    fast_ema   = ema(series, fast)
    slow_ema   = ema(series, slow)
    macd_line  = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist       = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": hist})


# ══════════════════════════════════════════════════════════════════════════════
# ABSTRACT STRATEGY BASE
# ══════════════════════════════════════════════════════════════════════════════

class Strategy(ABC):
    """
    Abstract base class for all trading strategies.

    Concrete strategies must implement:
      - calculate_signals(event: MarketEvent) → None
        Called once per MarketEvent. Emits zero or more SignalEvents
        into the shared event_queue.

    Provided helpers:
      - _get_bars(symbol, N)     → pd.DataFrame or None
      - _emit_signal(event, direction, strength)
      - _is_invested(symbol)     → bool
      - _current_price(symbol)   → float or None
    """

    def __init__(
        self,
        config: BacktestConfig,
        data_handler: DataHandler,
        event_queue: deque,
    ) -> None:
        self._config       = config
        self._data         = data_handler
        self._event_queue  = event_queue
        # Subclasses can store per-symbol state in _state dict
        self._state: dict  = {}

    # ── Abstract interface ────────────────────────────────────────────────

    @property
    @abstractmethod
    def strategy_id(self) -> str:
        """Unique human-readable identifier for this strategy instance."""
        ...

    @abstractmethod
    def calculate_signals(self, event: MarketEvent) -> None:
        """
        Core signal generation logic.

        Called by the event loop on every MarketEvent.
        Must call self._emit_signal(...) to place signals in the queue.

        Parameters
        ----------
        event : MarketEvent — the most recent bar just pushed by DataHandler.
        """
        ...

    # ── Protected helpers ─────────────────────────────────────────────────

    def _get_bars(self, symbol: str, N: int) -> Optional[pd.DataFrame]:
        """
        Return the last N bars for symbol from DataHandler.
        Returns None if symbol is unknown or insufficient history exists.
        """
        bars = self._data.get_latest_bars(symbol, N)
        if bars is None or len(bars) < N:
            return None
        return bars

    def _emit_signal(
        self,
        event: MarketEvent,
        direction: SignalDirection,
        strength: float = 1.0,
    ) -> None:
        """
        Construct a SignalEvent and push it into the event queue.

        Parameters
        ----------
        event     : MarketEvent — the triggering bar event.
        direction : SignalDirection
        strength  : float ∈ [0, 1] — confidence (default 1.0).
        """
        signal = SignalEvent(
            timestamp=event.timestamp,
            symbol=event.symbol,
            strategy_id=self.strategy_id,
            direction=direction,
            strength=max(0.0, min(1.0, strength)),
        )
        self._event_queue.append(signal)
        logger.debug(
            "[%s] %s signal on %s @ %s (strength=%.2f)",
            self.strategy_id,
            direction.value,
            event.symbol,
            event.timestamp.date() if event.timestamp else "N/A",
            strength,
        )

    def _is_invested(self, symbol: str) -> bool:
        """
        Check whether the portfolio currently holds a position in `symbol`.

        The Portfolio object is not directly accessible here to keep coupling
        loose. Instead, the strategy tracks its own notion of "invested" via
        a state dict. The Portfolio sends back a FillEvent which the strategy
        can observe — but for simplicity, strategies track signal state only.

        Subclasses update self._state[symbol]["invested"] on entry/exit signals.
        """
        return self._state.get(symbol, {}).get("invested", False)

    def _set_invested(self, symbol: str, invested: bool) -> None:
        """Update the invested flag for a symbol."""
        if symbol not in self._state:
            self._state[symbol] = {}
        self._state[symbol]["invested"] = invested

    def _current_price(self, symbol: str) -> Optional[float]:
        """Return the close price of the current bar for a symbol."""
        return self._data.get_current_price(symbol)
