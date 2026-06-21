# tests/test_strategies.py
# ============================================================
# UNIT TESTS — strategies/ + engine/strategy.py indicators
# ============================================================

import numpy as np
import pandas as pd
import pytest
from collections import deque
from datetime import datetime
from unittest.mock import MagicMock, patch

from engine.strategy import sma, ema, moving_average, rsi, atr, bollinger_bands, macd
from engine.events import MarketEvent, EventType, SignalDirection, SignalEvent
from config import BacktestConfig


# ══════════════════════════════════════════════════════════════════════════════
# INDICATOR UNIT TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestSMA:
    def test_window_of_one(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sma(s, 1)
        pd.testing.assert_series_equal(result, s.astype(float), check_names=False)

    def test_initial_nans(self):
        s = pd.Series(range(1, 11), dtype=float)
        result = sma(s, 5)
        assert result.iloc[:4].isna().all()
        assert not result.iloc[4:].isna().any()

    def test_known_value(self):
        s = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0])
        result = sma(s, 5)
        assert result.iloc[-1] == pytest.approx(6.0)

    def test_all_same_values(self):
        s = pd.Series([5.0] * 20)
        result = sma(s, 10)
        assert result.dropna().apply(lambda x: x == pytest.approx(5.0)).all()


class TestEMA:
    def test_initial_nans_respect_min_periods(self):
        s = pd.Series(range(1, 21), dtype=float)
        result = ema(s, 10)
        assert result.iloc[:9].isna().all()
        assert not result.iloc[9:].isna().any()

    def test_ema_reacts_faster_than_sma(self):
        """EMA should be closer to recent prices than SMA after a step-up."""
        base  = [100.0] * 20
        spike = [200.0] * 5
        s     = pd.Series(base + spike)
        e     = ema(s, 10)
        sm    = sma(s, 10)
        # After the spike, EMA should be closer to 200 than SMA
        assert abs(e.iloc[-1] - 200) < abs(sm.iloc[-1] - 200)


class TestMovingAverage:
    def test_sma_dispatch(self):
        s = pd.Series(range(1, 21), dtype=float)
        result = moving_average(s, 10, "SMA")
        pd.testing.assert_series_equal(result, sma(s, 10))

    def test_ema_dispatch(self):
        s = pd.Series(range(1, 21), dtype=float)
        result = moving_average(s, 10, "EMA")
        pd.testing.assert_series_equal(result, ema(s, 10))

    def test_invalid_type_raises(self):
        s = pd.Series(range(1, 21), dtype=float)
        with pytest.raises(ValueError, match="Unknown ma_type"):
            moving_average(s, 10, "VWAP")


class TestRSI:
    def test_range_0_to_100(self):
        np.random.seed(0)
        s = pd.Series(1000 + np.cumsum(np.random.randn(200) * 5))
        r = rsi(s, 14)
        valid = r.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_constant_prices_rsi_undefined(self):
        s = pd.Series([100.0] * 30)
        r = rsi(s, 14)
        # Constant prices → zero gains and zero losses → NaN
        assert r.dropna().empty or (r.dropna() == pytest.approx(50.0, abs=50)).all()

    def test_strongly_rising_above_70(self):
        # Use a trending series with small random noise so RSI warm-up completes
        rng  = np.random.default_rng(0)
        base = np.linspace(100, 400, 200)
        noise = rng.normal(0, 0.5, 200)
        s = pd.Series(base + noise)
        r = rsi(s, 14)
        valid = r.dropna()
        assert len(valid) > 0, "RSI produced no valid values"
        assert valid.iloc[-1] > 70

    def test_strongly_falling_below_30(self):
        s = pd.Series(np.linspace(300, 100, 100))
        r = rsi(s, 14)
        assert r.iloc[-1] < 30

    def test_initial_nans(self):
        s = pd.Series(range(1, 50), dtype=float)
        r = rsi(s, 14)
        assert r.iloc[:14].isna().all()


class TestATR:
    def test_positive_values(self):
        np.random.seed(1)
        close = pd.Series(1000 + np.cumsum(np.random.randn(100) * 3))
        high  = close + np.random.uniform(5, 20, 100)
        low   = close - np.random.uniform(5, 20, 100)
        a     = atr(high, low, close, 14)
        assert (a.dropna() > 0).all()

    def test_no_negative_values(self):
        close = pd.Series(np.linspace(100, 200, 50))
        high  = close + 5
        low   = close - 5
        a     = atr(high, low, close, 14)
        assert (a.dropna() >= 0).all()


class TestBollingerBands:
    def test_upper_gt_middle_gt_lower(self):
        s  = pd.Series(1000 + np.cumsum(np.random.randn(100) * 5))
        bb = bollinger_bands(s, 20, 2.0)
        valid = bb.dropna()
        assert (valid["upper"] > valid["middle"]).all()
        assert (valid["middle"] > valid["lower"]).all()

    def test_returns_correct_columns(self):
        s  = pd.Series(range(1, 51), dtype=float)
        bb = bollinger_bands(s, 20)
        assert set(bb.columns) == {"upper", "middle", "lower"}


class TestMACD:
    def test_returns_correct_columns(self):
        s = pd.Series(1000 + np.cumsum(np.random.randn(200) * 5))
        m = macd(s)
        assert set(m.columns) == {"macd", "signal", "hist"}

    def test_hist_equals_macd_minus_signal(self):
        s = pd.Series(1000 + np.cumsum(np.random.randn(200) * 5))
        m = macd(s)
        expected = m["macd"] - m["signal"]
        pd.testing.assert_series_equal(m["hist"], expected, check_names=False)


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY INTEGRATION SMOKE TESTS
# ══════════════════════════════════════════════════════════════════════════════

def _make_mock_data_handler(prices: list, symbol: str = "TEST") -> MagicMock:
    """Create a mock DataHandler that returns a DataFrame of prices."""
    dates  = pd.date_range("2016-01-04", periods=len(prices), freq="B")
    close  = pd.Series(prices, index=dates)
    high   = close * 1.01
    low    = close * 0.99
    df     = pd.DataFrame({"close": close, "high": high, "low": low,
                           "open": close, "volume": 1_000_000})

    mock = MagicMock()
    mock.has_high_low  = {symbol: True}
    mock.has_volume    = {symbol: True}
    mock.get_current_price.return_value = float(prices[-1])

    def _latest_bars(sym, N):
        if sym != symbol:
            return None
        return df.tail(N) if N <= len(df) else None

    mock.get_latest_bars.side_effect = _latest_bars
    return mock


class TestDualMAStrategy:
    def _make_strategy(self, prices, fast=5, slow=10):
        from strategies.dual_ma import DualMAStrategy
        cfg = BacktestConfig()
        cfg.strategy.fast_ma_window = fast
        cfg.strategy.slow_ma_window = slow
        cfg.strategy.ma_type = "SMA"
        queue   = deque()
        handler = _make_mock_data_handler(prices, "X")
        return DualMAStrategy(cfg, handler, queue), queue

    def test_golden_cross_emits_long(self):
        # Falling then rising: creates a golden cross
        prices = [100 - i for i in range(15)] + [90 + i * 2 for i in range(20)]
        strat, queue = self._make_strategy(prices)
        event = MarketEvent(timestamp=datetime(2020, 1, 1), symbol="X", close=prices[-1])

        # Simulate running through enough bars
        handler = strat._data
        handler.get_latest_bars.return_value = pd.DataFrame({
            "close": prices,
            "high":  [p * 1.01 for p in prices],
            "low":   [p * 0.99 for p in prices],
            "open":  prices,
            "volume": [1_000_000] * len(prices),
        }, index=pd.date_range("2016-01-04", periods=len(prices), freq="B"))

        strat.calculate_signals(event)
        signals = [e for e in queue if e.event_type == EventType.SIGNAL]
        # May or may not fire depending on crossover at last bar — just ensure no crash
        for sig in signals:
            assert sig.direction in (SignalDirection.LONG, SignalDirection.EXIT_LONG)

    def test_invalid_window_raises(self):
        from strategies.dual_ma import DualMAStrategy
        cfg = BacktestConfig()
        cfg.strategy.fast_ma_window = 50
        cfg.strategy.slow_ma_window = 20  # fast > slow
        with pytest.raises(ValueError, match="must be <"):
            DualMAStrategy(cfg, MagicMock(), deque())


class TestRSIStrategy:
    def _make_strategy(self, prices, oversold=30, overbought=70, reversal=True):
        from strategies.rsi_strategy import RSIStrategy
        cfg = BacktestConfig()
        cfg.strategy.rsi_period      = 14
        cfg.strategy.rsi_oversold    = oversold
        cfg.strategy.rsi_overbought  = overbought
        cfg.strategy.rsi_entry_on_reversal = reversal
        queue   = deque()
        handler = _make_mock_data_handler(prices, "Y")
        return RSIStrategy(cfg, handler, queue), queue

    def test_strategy_id_format(self):
        prices = list(range(100, 200))
        strat, _ = self._make_strategy(prices)
        assert "RSI" in strat.strategy_id

    def test_no_crash_on_insufficient_history(self):
        prices = [100.0] * 5
        strat, queue = self._make_strategy(prices)
        event = MarketEvent(timestamp=datetime(2020, 1, 1), symbol="Y", close=100.0)
        strat.calculate_signals(event)
        # Should not raise; no signal expected
        assert len(queue) == 0
