# tests/test_phase2_modules.py
# ============================================================
# UNIT TESTS — Phase 2 modules C–G
# ============================================================

from collections import deque
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from config import BacktestConfig
from engine.events import MarketEvent, OrderDirection, OrderEvent, SignalDirection, SignalEvent
from engine.regime_filter import RegimeFilter
from portfolio.ensemble import EnsembleAllocator


def _trending_prices(n: int, start: float = 100.0, drift: float = 0.5) -> np.ndarray:
    return start + np.arange(n) * drift


def _make_ohlcv(prices: np.ndarray, dates=None) -> pd.DataFrame:
    if dates is None:
        dates = pd.date_range("2018-01-01", periods=len(prices), freq="B")
    close = pd.Series(prices, index=dates)
    return pd.DataFrame({
        "close": close,
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "volume": np.full(len(prices), 1_000_000),
    })


class TestRegimeFilter:
    def test_sma_200_trend_regime(self):
        cfg = BacktestConfig()
        cfg.advanced.regime.method = "sma_200"
        cfg.advanced.regime.sma_window = 20
        rf = RegimeFilter(cfg)

        prices = _trending_prices(60, start=100, drift=2.0)
        bench = _make_ohlcv(prices)
        mock = MagicMock()
        mock.get_benchmark_bars.return_value = bench

        rf.update(datetime(2018, 3, 1), mock)
        assert rf.current_regime == "trend"
        assert rf.is_active("DualMA") is True
        assert rf.is_active("PairsStatArb") is False

    def test_is_active_when_unmapped(self):
        cfg = BacktestConfig()
        rf = RegimeFilter(cfg)
        rf.current_regime = "range"
        assert rf.is_active("UnknownStrategy") is True

    def test_get_regime_series(self):
        cfg = BacktestConfig()
        rf = RegimeFilter(cfg)
        rf.regime_history = [(datetime(2020, 1, 1), "trend"), (datetime(2020, 2, 1), "range")]
        s = rf.get_regime_series()
        assert len(s) == 2


class TestEnsembleAllocator:
    def test_equal_weight_filter_signal(self):
        cfg = BacktestConfig()
        alloc = EnsembleAllocator(cfg, ["A", "B"])
        sig = SignalEvent(
            timestamp=datetime(2020, 1, 1),
            symbol="X",
            strategy_id="A",
            direction=SignalDirection.LONG,
            strength=1.0,
        )
        out = alloc.filter_signal(sig)
        assert out.strength == pytest.approx(0.5)

    def test_equal_vol_weights(self):
        cfg = BacktestConfig()
        alloc = EnsembleAllocator(cfg, ["A", "B"])
        alloc.sleeve_returns["A"] = [0.01] * 30
        alloc.sleeve_returns["B"] = [0.05] * 30
        weights = alloc._compute_equal_vol_weights()
        assert weights["A"] > weights["B"]

    def test_rebalance_monthly(self):
        cfg = BacktestConfig()
        alloc = EnsembleAllocator(cfg, ["A"])
        assert alloc._is_rebalance_date(datetime(2020, 1, 15)) is True
        alloc.last_rebalance_date = datetime(2020, 1, 15)
        assert alloc._is_rebalance_date(datetime(2020, 1, 20)) is False
        assert alloc._is_rebalance_date(datetime(2020, 2, 1)) is True


class TestVolatilityBreakoutStrategy:
    def test_strategy_id(self):
        from strategies.volatility_breakout import VolatilityBreakoutStrategy
        s = VolatilityBreakoutStrategy(BacktestConfig(), deque())
        assert "VolBreakout" in s.strategy_id

    def test_long_entry_on_breakout(self):
        from strategies.volatility_breakout import VolatilityBreakoutStrategy
        cfg = BacktestConfig()
        cfg.advanced.breakout.donchian_window = 5
        cfg.advanced.breakout.atr_breakout_multiplier = 0.1
        queue = deque()
        s = VolatilityBreakoutStrategy(cfg, queue)

        base = np.linspace(100, 110, 30)
        spike = np.concatenate([base, [130.0]])
        mock = MagicMock()
        mock.has_high_low = {"X": True}
        df = _make_ohlcv(spike)
        mock.get_latest_bars.return_value = df

        event = MarketEvent(timestamp=datetime(2020, 2, 1), symbol="X", close=130.0)
        s.calculate_signals(event, mock)
        assert any(e.direction == SignalDirection.LONG for e in queue)


class TestPairsStatArbStrategy:
    def test_zscore_computation(self):
        from strategies.pairs_stat_arb import PairsStatArbStrategy
        s = PairsStatArbStrategy(BacktestConfig(), deque())
        state = {
            "hedge_ratio": 1.0,
            "spread_mean": 0.0,
            "spread_std": 2.0,
        }
        mock = MagicMock()
        mock.get_current_price.side_effect = lambda sym: 104.0 if sym == "A" else 100.0
        z = s._compute_zscore("A", "B", state, mock)
        assert z == pytest.approx(2.0)

    @patch("strategies.pairs_stat_arb.coint")
    def test_cointegration_active(self, mock_coint):
        from strategies.pairs_stat_arb import PairsStatArbStrategy
        mock_coint.return_value = (0.0, 0.01, [0.0, 0.0, 0.0])
        cfg = BacktestConfig()
        cfg.advanced.pairs.candidate_pairs = [("A", "B")]
        cfg.advanced.pairs.lookback_window = 20
        s = PairsStatArbStrategy(cfg, deque())

        prices = _trending_prices(25)
        df = _make_ohlcv(prices)
        mock = MagicMock()
        mock.get_latest_bars.return_value = df

        result = s._test_cointegration("A", "B", datetime(2020, 1, 1), mock)
        assert result["is_cointegrated"] is True
        assert result["hedge_ratio"] is not None


class TestAdvancedExecutionHandler:
    def test_impact_slippage_increases_with_size(self):
        from engine.execution_advanced import AdvancedExecutionHandler
        cfg = BacktestConfig()
        ex = AdvancedExecutionHandler(cfg, deque())
        small = ex._calculate_impact_slippage(100, 1_000_000)
        large = ex._calculate_impact_slippage(100_000, 1_000_000)
        assert large > small

    def test_capacity_blocks_oversized_order(self):
        from engine.execution_advanced import AdvancedExecutionHandler
        cfg = BacktestConfig()
        cfg.advanced.advanced_execution.participation_rate_limit = 0.10
        queue = deque()
        ex = AdvancedExecutionHandler(cfg, queue)

        order = OrderEvent(
            timestamp=datetime(2020, 1, 1),
            symbol="X",
            direction=OrderDirection.BUY,
            quantity=200_000,
        )
        mock = MagicMock()
        mock.get_current_datetime.return_value = datetime(2020, 1, 1)
        mock.get_latest_bars.return_value = _make_ohlcv(np.full(20, 100.0))
        mock.get_next_open.return_value = 100.0

        ex.execute_order(order, mock)
        assert order.quantity == 100_000
        assert len(queue) == 1
        assert len(ex.capacity_constrained_orders) == 1
