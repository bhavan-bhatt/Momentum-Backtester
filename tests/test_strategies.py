# tests/test_strategies.py
# ============================================================
# UNIT TESTS — strategies/ + engine/strategy.py indicators
# ============================================================

import numpy as np
import pandas as pd
import pytest
from collections import deque
from datetime import datetime
from unittest.mock import MagicMock

from engine.strategy import (
    BaseStrategy, sma, ema, moving_average, rsi, atr, bollinger_bands, macd
)
from engine.events import MarketEvent, EventType, SignalDirection, SignalEvent
from config import BacktestConfig


# ══════════════════════════════════════════════════════════════════════════════
# INDICATOR UNIT TESTS (module-level functions)
# ══════════════════════════════════════════════════════════════════════════════

class TestSMA:
    def test_window_of_one(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        pd.testing.assert_series_equal(sma(s, 1), s.astype(float), check_names=False)

    def test_initial_nans(self):
        s = pd.Series(range(1, 11), dtype=float)
        result = sma(s, 5)
        assert result.iloc[:4].isna().all()
        assert not result.iloc[4:].isna().any()

    def test_known_value(self):
        s = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0])
        assert sma(s, 5).iloc[-1] == pytest.approx(6.0)

    def test_all_same_values(self):
        s = pd.Series([5.0] * 20)
        assert sma(s, 10).dropna().apply(lambda x: x == pytest.approx(5.0)).all()


class TestEMA:
    def test_initial_nans_respect_min_periods(self):
        s = pd.Series(range(1, 21), dtype=float)
        result = ema(s, 10)
        assert result.iloc[:9].isna().all()
        assert not result.iloc[9:].isna().any()

    def test_ema_reacts_faster_than_sma(self):
        base  = [100.0] * 20
        spike = [200.0] * 5
        s     = pd.Series(base + spike)
        assert abs(ema(s, 10).iloc[-1] - 200) < abs(sma(s, 10).iloc[-1] - 200)


class TestMovingAverage:
    def test_sma_dispatch(self):
        s = pd.Series(range(1, 21), dtype=float)
        pd.testing.assert_series_equal(moving_average(s, 10, "SMA"), sma(s, 10))

    def test_ema_dispatch(self):
        s = pd.Series(range(1, 21), dtype=float)
        pd.testing.assert_series_equal(moving_average(s, 10, "EMA"), ema(s, 10))

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

    def test_strongly_rising_above_70(self):
        rng   = np.random.default_rng(0)
        base  = np.linspace(100, 400, 200)
        noise = rng.normal(0, 0.5, 200)
        s     = pd.Series(base + noise)
        r     = rsi(s, 14)
        valid = r.dropna()
        assert len(valid) > 0
        assert valid.iloc[-1] > 70

    def test_strongly_falling_below_30(self):
        s = pd.Series(np.linspace(300, 100, 100))
        r = rsi(s, 14)
        assert r.dropna().iloc[-1] < 30

    def test_initial_nans(self):
        s = pd.Series(range(1, 50), dtype=float)
        assert rsi(s, 14).iloc[:14].isna().all()

    def test_constant_prices_no_crash(self):
        s = pd.Series([100.0] * 30)
        r = rsi(s, 14)
        # Should not raise; result may be NaN or 100 (zero loss path)
        assert isinstance(r, pd.Series)


class TestATR:
    def test_positive_values(self):
        np.random.seed(1)
        close = pd.Series(1000 + np.cumsum(np.random.randn(100) * 3))
        high  = close + np.random.uniform(5, 20, 100)
        low   = close - np.random.uniform(5, 20, 100)
        assert (atr(high, low, close, 14).dropna() > 0).all()

    def test_no_negative_values(self):
        close = pd.Series(np.linspace(100, 200, 50))
        high  = close + 5
        low   = close - 5
        assert (atr(high, low, close, 14).dropna() >= 0).all()

    def test_all_nan_when_hl_missing(self):
        close = pd.Series([100.0] * 30)
        h     = pd.Series([np.nan] * 30)
        l     = pd.Series([np.nan] * 30)
        result = atr(h, l, close, 14)
        assert result.isna().all()


class TestBollingerBands:
    def test_upper_gt_middle_gt_lower(self):
        s  = pd.Series(1000 + np.cumsum(np.random.randn(100) * 5))
        bb = bollinger_bands(s, 20, 2.0)
        valid = bb.dropna()
        assert (valid["upper"] > valid["middle"]).all()
        assert (valid["middle"] > valid["lower"]).all()

    def test_returns_correct_columns(self):
        bb = bollinger_bands(pd.Series(range(1, 51), dtype=float), 20)
        assert set(bb.columns) == {"upper", "middle", "lower"}


class TestMACD:
    def test_returns_correct_columns(self):
        s = pd.Series(1000 + np.cumsum(np.random.randn(200) * 5))
        m = macd(s)
        assert set(m.columns) == {"macd", "signal", "hist"}

    def test_hist_equals_macd_minus_signal(self):
        s = pd.Series(1000 + np.cumsum(np.random.randn(200) * 5))
        m = macd(s)
        pd.testing.assert_series_equal(m["hist"], m["macd"] - m["signal"], check_names=False)


# ══════════════════════════════════════════════════════════════════════════════
# BaseStrategy INSTANCE-METHOD INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

class ConcreteStrategy(BaseStrategy):
    """Minimal concrete subclass for testing BaseStrategy methods."""
    def calculate_signals(self, event, data_handler):
        pass


@pytest.fixture
def strategy():
    cfg = BacktestConfig()
    return ConcreteStrategy(cfg, deque())


class TestBaseStrategyIndicators:
    def test_calculate_sma(self, strategy):
        s      = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = strategy._calculate_sma(s, 3)
        assert result.iloc[-1] == pytest.approx(4.0)

    def test_calculate_ema(self, strategy):
        s = pd.Series(range(1, 21), dtype=float)
        assert not strategy._calculate_ema(s, 5).dropna().empty

    def test_calculate_rsi_range(self, strategy):
        np.random.seed(42)
        s = pd.Series(1000 + np.cumsum(np.random.randn(200) * 5))
        r = strategy._calculate_rsi(s, 14)
        assert (r.dropna() >= 0).all() and (r.dropna() <= 100).all()

    def test_calculate_atr_positive(self, strategy):
        close = pd.Series(np.linspace(100, 200, 50))
        high  = close + 5
        low   = close - 5
        a     = strategy._calculate_atr(high, low, close, 14)
        assert (a.dropna() > 0).all()

    def test_get_latest_indicator_value_none_on_nan(self, strategy):
        s = pd.Series([np.nan, np.nan, 3.0])
        assert strategy._get_latest_indicator_value(s, lookback=2) is None
        assert strategy._get_latest_indicator_value(s, lookback=1) == pytest.approx(3.0)

    def test_crossed_above_detection(self, strategy):
        fast = pd.Series([1.0, 2.0, 3.0, 5.0])
        slow = pd.Series([4.0, 4.0, 4.0, 4.0])
        assert strategy._crossed_above(fast, slow)

    def test_crossed_below_detection(self, strategy):
        fast = pd.Series([5.0, 4.0, 3.0, 1.0])
        slow = pd.Series([2.0, 2.0, 2.0, 2.0])
        assert strategy._crossed_below(fast, slow)

    def test_no_cross_returns_false(self, strategy):
        fast = pd.Series([1.0, 1.0, 1.0, 1.0])
        slow = pd.Series([5.0, 5.0, 5.0, 5.0])
        assert not strategy._crossed_above(fast, slow)
        assert not strategy._crossed_below(fast, slow)


class TestBaseStrategyPositionState:
    def test_flat_by_default(self, strategy):
        assert strategy._is_flat("RELIANCE.NS") is True
        assert strategy._is_long("RELIANCE.NS") is False

    def test_emit_signal_updates_position(self, strategy):
        event = MarketEvent(timestamp=datetime(2020, 1, 1), symbol="X", close=100.0)
        strategy._emit_signal("X", event.timestamp, SignalDirection.LONG)
        assert strategy._is_long("X") is True
        assert strategy._is_flat("X") is False

    def test_exit_clears_position(self, strategy):
        event = MarketEvent(timestamp=datetime(2020, 1, 1), symbol="X", close=100.0)
        strategy._emit_signal("X", event.timestamp, SignalDirection.LONG)
        strategy._emit_signal("X", event.timestamp, SignalDirection.EXIT_LONG)
        assert strategy._is_flat("X") is True

    def test_has_enough_bars_true(self, strategy):
        df = pd.DataFrame({"close": [1.0] * 10})
        assert strategy._has_enough_bars(df, 10) is True
        assert strategy._has_enough_bars(df, 11) is False

    def test_has_enough_bars_none(self, strategy):
        assert strategy._has_enough_bars(None, 1) is False


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY INTEGRATION SMOKE TESTS
# ══════════════════════════════════════════════════════════════════════════════

def _make_mock_handler(prices: list, symbol: str = "X") -> MagicMock:
    dates = pd.date_range("2016-01-04", periods=len(prices), freq="B")
    close = pd.Series(prices, index=dates)
    df    = pd.DataFrame({
        "close":  close,
        "high":   close * 1.01,
        "low":    close * 0.99,
        "open":   close,
        "volume": 1_000_000,
    })
    mock = MagicMock()
    mock.has_high_low  = {symbol: True}
    mock.has_volume    = {symbol: True}
    mock.get_current_price.return_value = float(prices[-1])

    def _latest(sym, N):
        return df.tail(N) if sym == symbol and N <= len(df) else None

    mock.get_latest_bars.side_effect = _latest
    return mock


class TestDualMAStrategy:
    def test_raises_on_invalid_windows(self):
        from strategies.dual_ma import DualMAStrategy
        cfg = BacktestConfig()
        cfg.strategy.fast_ma_window = 50
        cfg.strategy.slow_ma_window = 20
        with pytest.raises(ValueError, match="must be"):
            DualMAStrategy(cfg, deque())

    def test_strategy_id_format(self):
        from strategies.dual_ma import DualMAStrategy
        cfg = BacktestConfig()
        s   = DualMAStrategy(cfg, deque())
        assert "DualMA" in s.strategy_id

    def test_no_signal_on_insufficient_bars(self):
        from strategies.dual_ma import DualMAStrategy
        cfg   = BacktestConfig()
        queue = deque()
        s     = DualMAStrategy(cfg, queue)
        mock  = _make_mock_handler([100.0] * 5)
        event = MarketEvent(timestamp=datetime(2020, 1, 1), symbol="X", close=100.0)
        s.calculate_signals(event, mock)
        assert len(queue) == 0

    def test_no_crash_on_flat_series(self):
        from strategies.dual_ma import DualMAStrategy
        cfg   = BacktestConfig()
        queue = deque()
        s     = DualMAStrategy(cfg, queue)
        prices = [100.0] * 60
        mock   = _make_mock_handler(prices)
        event  = MarketEvent(timestamp=datetime(2020, 1, 1), symbol="X", close=100.0)
        s.calculate_signals(event, mock)  # should not raise


class TestRSIStrategy:
    def test_strategy_id_contains_rsi(self):
        from strategies.rsi_strategy import RSIStrategy
        cfg = BacktestConfig()
        s   = RSIStrategy(cfg, deque())
        assert "RSI" in s.strategy_id

    def test_invalid_thresholds_raise(self):
        from strategies.rsi_strategy import RSIStrategy
        cfg = BacktestConfig()
        cfg.strategy.rsi_oversold  = 60
        cfg.strategy.rsi_overbought = 40
        with pytest.raises(ValueError):
            RSIStrategy(cfg, deque())

    def test_no_signal_on_insufficient_bars(self):
        from strategies.rsi_strategy import RSIStrategy
        cfg   = BacktestConfig()
        queue = deque()
        s     = RSIStrategy(cfg, queue)
        mock  = _make_mock_handler([100.0] * 5)
        event = MarketEvent(timestamp=datetime(2020, 1, 1), symbol="X", close=100.0)
        s.calculate_signals(event, mock)
        assert len(queue) == 0


class TestCombinedStrategy:
    def test_strategy_id(self):
        from strategies.combined import CombinedStrategy
        s = CombinedStrategy(BacktestConfig(), deque())
        assert "Combined" in s.strategy_id

    def test_signal_strength_in_range(self):
        from strategies.combined import CombinedStrategy
        s = CombinedStrategy(BacktestConfig(), deque())
        strength = s._calculate_signal_strength(25.0, 110.0, 100.0)
        assert 0.0 <= strength <= 1.0

    def test_signal_strength_higher_when_deeper_oversold(self):
        from strategies.combined import CombinedStrategy
        s = CombinedStrategy(BacktestConfig(), deque())
        s1 = s._calculate_signal_strength(28.0, 110.0, 100.0)  # RSI near oversold
        s2 = s._calculate_signal_strength(10.0, 110.0, 100.0)  # RSI deeply oversold
        assert s2 > s1  # deeper → higher strength
