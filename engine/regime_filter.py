# engine/regime_filter.py
# ============================================================
# MARKET REGIME DETECTION — TURNS STRATEGIES ON/OFF
# ============================================================

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import BacktestConfig

logger = logging.getLogger(__name__)


class RegimeFilter:
    """
    Classifies market state as 'trend' or 'range' from the benchmark index.

    Strategies query is_active() before opening new positions.
    """

    def __init__(self, config: BacktestConfig) -> None:
        self._config = config
        self.method = config.advanced.regime.method
        self.adx_threshold = config.advanced.regime.adx_threshold
        self.sma_window = config.advanced.regime.sma_window
        self.strategy_regime_map: Dict[str, str] = dict(
            config.advanced.regime.strategy_regime_map
        )
        self.current_regime = "trend"
        self.regime_history: List[Tuple[datetime, str]] = []

    def update(self, current_date: datetime, data_handler) -> None:
        """Recompute regime from latest benchmark data."""
        required = self.sma_window + 5 if self.method == "sma_200" else 50
        bench_bars = data_handler.get_benchmark_bars(N=required)

        if bench_bars is None or len(bench_bars) < required:
            return

        if self.method == "sma_200":
            sma = (
                bench_bars["close"]
                .rolling(self.sma_window, min_periods=self.sma_window)
                .mean()
                .iloc[-1]
            )
            current_close = bench_bars["close"].iloc[-1]
            if pd.isna(sma):
                return
            new_regime = "trend" if current_close > sma else "range"
        elif self.method == "adx":
            adx_value = self._calculate_adx(bench_bars, period=14)
            if adx_value is None or np.isnan(adx_value):
                return
            new_regime = "trend" if adx_value > self.adx_threshold else "range"
        else:
            logger.warning("Unknown regime method '%s' — keeping current regime.", self.method)
            return

        self.current_regime = new_regime
        self.regime_history.append((current_date, new_regime))

    def _calculate_adx(self, bars: pd.DataFrame, period: int = 14) -> Optional[float]:
        """Compute Wilder's ADX for the most recent bar."""
        for col in ("high", "low", "close"):
            if col not in bars.columns or bars[col].isna().all():
                return None

        high = bars["high"]
        low = bars["low"]
        close = bars["close"]

        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        alpha = 1.0 / period
        smoothed_tr = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        smoothed_plus = pd.Series(plus_dm, index=bars.index).ewm(
            alpha=alpha, min_periods=period, adjust=False
        ).mean()
        smoothed_minus = pd.Series(minus_dm, index=bars.index).ewm(
            alpha=alpha, min_periods=period, adjust=False
        ).mean()

        plus_di = 100.0 * smoothed_plus / smoothed_tr.replace(0, np.nan)
        minus_di = 100.0 * smoothed_minus / smoothed_tr.replace(0, np.nan)

        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

        val = adx.iloc[-1]
        return None if pd.isna(val) else float(val)

    def is_active(self, strategy_name: str) -> bool:
        """Return True if strategy may open new entries in the current regime."""
        required_regime = self.strategy_regime_map.get(strategy_name)
        if required_regime is None:
            return True
        return required_regime == self.current_regime

    def get_regime_series(self) -> pd.Series:
        """Return regime history as a time series for charting."""
        if not self.regime_history:
            return pd.Series(dtype=object, name="regime")
        dates, regimes = zip(*self.regime_history)
        return pd.Series(regimes, index=pd.DatetimeIndex(dates), name="regime")
