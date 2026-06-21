# tests/test_data_handler.py
# ============================================================
# UNIT TESTS — engine/data_handler.py
# ============================================================

import os
import tempfile
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from engine.data_handler import DataHandler
from engine.events import EventType, MarketEvent
from config import BacktestConfig, DataConfig


def _make_csv(path: str, rows: int = 100, include_hl: bool = True,
              include_vol: bool = True, include_adj: bool = True) -> None:
    """Write a synthetic OHLCV CSV."""
    dates  = pd.date_range("2016-01-04", periods=rows, freq="B")  # business days
    close  = 1000 + np.cumsum(np.random.randn(rows) * 5)
    data = {"Date": dates.strftime("%Y-%m-%d"), "Close": close.round(2)}
    if include_hl:
        data["High"] = (close * 1.01).round(2)
        data["Low"]  = (close * 0.99).round(2)
        data["Open"] = close.round(2)
    if include_vol:
        data["Volume"] = np.random.randint(100_000, 1_000_000, rows)
    if include_adj:
        data["Adj Close"] = close.round(2)
    pd.DataFrame(data).to_csv(path, index=False)


@pytest.fixture
def tmp_csv_dir(tmp_path):
    """Create a temporary CSV directory with two synthetic symbols."""
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    _make_csv(str(csv_dir / "SYM_A.csv"), rows=500)
    _make_csv(str(csv_dir / "SYM_B.csv"), rows=500)
    _make_csv(str(csv_dir / "BENCH.csv"), rows=500)
    return str(csv_dir)


@pytest.fixture
def config(tmp_csv_dir):
    cfg = BacktestConfig()
    cfg.data.csv_dir          = tmp_csv_dir
    cfg.data.symbols          = ["SYM_A", "SYM_B"]
    cfg.data.benchmark_symbol = "BENCH"
    cfg.data.start_date       = "2016-01-01"
    cfg.data.end_date         = "2017-12-31"
    cfg.verbose               = False
    return cfg


@pytest.fixture
def handler(config):
    queue = deque()
    return DataHandler(config, queue), queue


class TestDataHandlerInit:
    def test_loads_symbols(self, handler):
        dh, _ = handler
        assert set(dh.get_symbols()) == {"SYM_A", "SYM_B"}

    def test_benchmark_loaded(self, handler):
        dh, _ = handler
        assert dh.benchmark_data is not None

    def test_has_high_low_true(self, handler):
        dh, _ = handler
        assert dh.has_high_low["SYM_A"] is True

    def test_raises_if_no_symbols(self, tmp_csv_dir):
        cfg = BacktestConfig()
        cfg.data.csv_dir  = tmp_csv_dir
        cfg.data.symbols  = ["NONEXISTENT"]
        cfg.data.benchmark_symbol = "BENCH"
        cfg.verbose = False
        with pytest.raises(RuntimeError, match="Zero symbols"):
            DataHandler(cfg, deque())


class TestUpdateBars:
    def test_pushes_market_events(self, handler):
        dh, queue = handler
        assert len(queue) == 0
        dh.update_bars()
        # One MarketEvent per symbol
        assert len(queue) == 2
        for ev in queue:
            assert ev.event_type == EventType.MARKET
            assert ev.symbol in {"SYM_A", "SYM_B"}

    def test_has_more_bars_decreases(self, handler):
        dh, queue = handler
        count = 0
        while dh.has_more_bars():
            dh.update_bars()
            queue.clear()
            count += 1
        assert count == len(dh._all_dates)
        assert not dh.has_more_bars()

    def test_is_last_bar_flag(self, handler):
        dh, queue = handler
        n_dates = len(dh._all_dates)
        for _ in range(n_dates):
            queue.clear()
            dh.update_bars()
        # All events in final batch should have is_last_bar=True
        assert all(ev.is_last_bar for ev in queue)


class TestGetLatestBars:
    def test_returns_none_before_any_bars(self, handler):
        dh, _ = handler
        result = dh.get_latest_bars("SYM_A", 10)
        # bar_index=0 → slice [0:0] → empty
        assert result is not None
        assert len(result) == 0

    def test_returns_correct_n_bars(self, handler):
        dh, queue = handler
        for _ in range(30):
            dh.update_bars()
            queue.clear()
        bars = dh.get_latest_bars("SYM_A", 20)
        assert bars is not None
        assert len(bars) == 20

    def test_no_look_ahead(self, handler):
        dh, queue = handler
        dh.update_bars(); queue.clear()  # advance once
        bars = dh.get_latest_bars("SYM_A", 1000)
        # Only 1 bar should be available
        assert len(bars) == 1


class TestGetCurrentPrice:
    def test_returns_float_after_one_bar(self, handler):
        dh, queue = handler
        dh.update_bars(); queue.clear()
        price = dh.get_current_price("SYM_A")
        assert isinstance(price, float)
        assert price > 0

    def test_returns_none_for_unknown_symbol(self, handler):
        dh, _ = handler
        assert dh.get_current_price("UNKNOWN") is None


class TestGetNextOpen:
    def test_returns_float(self, handler):
        dh, queue = handler
        dh.update_bars(); queue.clear()
        nxt = dh.get_next_open("SYM_A")
        assert isinstance(nxt, float)
        assert nxt > 0

    def test_at_end_returns_last_price(self, handler):
        dh, queue = handler
        n = len(dh._all_dates)
        for _ in range(n):
            dh.update_bars()
            queue.clear()
        # After exhaustion, should still return a float
        nxt = dh.get_next_open("SYM_A")
        assert isinstance(nxt, float)


class TestGetAllData:
    def test_returns_copy(self, handler):
        dh, _ = handler
        data = dh.get_all_data()
        assert "SYM_A" in data
        # Mutating the copy should not affect the original
        orig_len = len(dh.symbol_data["SYM_A"])
        data["SYM_A"].drop(data["SYM_A"].index[:10], inplace=True)
        assert len(dh.symbol_data["SYM_A"]) == orig_len


class TestColumnNormalisation:
    def test_tier2_no_adj_close(self, tmp_path):
        """Tier-2: no adj_close, should fall back to close."""
        csv_dir = tmp_path / "csv2"
        csv_dir.mkdir()
        _make_csv(str(csv_dir / "SYM_C.csv"), include_adj=False)

        cfg = BacktestConfig()
        cfg.data.csv_dir          = str(csv_dir)
        cfg.data.symbols          = ["SYM_C"]
        cfg.data.benchmark_symbol = "SYM_C"
        cfg.data.start_date       = "2016-01-01"
        cfg.data.end_date         = "2017-12-31"
        cfg.verbose               = False
        dh = DataHandler(cfg, deque())
        assert len(dh.get_symbols()) == 1

    def test_tier3_close_only(self, tmp_path):
        """Tier-3: only date + close."""
        csv_dir = tmp_path / "csv3"
        csv_dir.mkdir()
        dates  = pd.date_range("2016-01-04", periods=300, freq="B")
        close  = 500 + np.cumsum(np.random.randn(300) * 2)
        pd.DataFrame({"Date": dates.strftime("%Y-%m-%d"), "Close": close}).to_csv(
            str(csv_dir / "SYM_D.csv"), index=False
        )
        cfg = BacktestConfig()
        cfg.data.csv_dir          = str(csv_dir)
        cfg.data.symbols          = ["SYM_D"]
        cfg.data.benchmark_symbol = "SYM_D"
        cfg.data.start_date       = "2016-01-01"
        cfg.data.end_date         = "2017-12-31"
        cfg.verbose               = False
        dh = DataHandler(cfg, deque())
        assert dh.has_high_low["SYM_D"] is False
