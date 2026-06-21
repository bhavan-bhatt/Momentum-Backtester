# tests/test_portfolio.py
# ============================================================
# UNIT TESTS — engine/portfolio.py
# ============================================================

import pytest
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime
from unittest.mock import MagicMock

from engine.events import (
    SignalEvent, SignalDirection,
    OrderEvent, OrderDirection,
    FillEvent, EventType,
)
from engine.portfolio import PortfolioManager, Position
from config import BacktestConfig, PortfolioConfig


def _make_portfolio(
    initial_capital: float = 1_000_000.0,
    sizing_method: str = "equal_weight",
    max_open: int = 5,
    allow_short: bool = False,
) -> tuple:
    cfg = BacktestConfig()
    cfg.portfolio.initial_capital  = initial_capital
    cfg.portfolio.sizing_method    = sizing_method
    cfg.portfolio.max_open_positions = max_open
    cfg.strategy.allow_short       = allow_short
    cfg.portfolio.max_position_pct = 0.30
    cfg.portfolio.risk_per_trade_pct = 0.02

    queue   = deque()
    handler = MagicMock()
    handler.get_current_price.return_value = 1000.0
    handler.has_high_low = {"SYM": True, "SYM2": True}

    port = PortfolioManager(cfg, handler, queue)
    return port, queue, handler, cfg


def _signal(symbol: str = "SYM", direction: SignalDirection = SignalDirection.LONG) -> SignalEvent:
    return SignalEvent(
        timestamp=datetime(2020, 3, 1),
        symbol=symbol,
        strategy_id="test",
        direction=direction,
    )


def _fill(
    symbol: str = "SYM",
    direction: OrderDirection = OrderDirection.BUY,
    qty: int = 100,
    price: float = 1000.0,
    order: OrderEvent = None,
) -> FillEvent:
    order_ref = order or OrderEvent(symbol=symbol, direction=direction, quantity=qty)
    return FillEvent(
        timestamp=datetime(2020, 3, 2),
        symbol=symbol,
        direction=direction,
        quantity=qty,
        fill_price=price,
        commission=50.0,
        slippage=5.0,
        order_ref=order_ref,
    )


class TestPosition:
    def test_market_value(self):
        pos = Position("SYM", 100, 1000.0, datetime(2020, 1, 1))
        assert pos.market_value(1100.0) == pytest.approx(110_000.0)

    def test_unrealised_pnl(self):
        pos = Position("SYM", 100, 1000.0, datetime(2020, 1, 1))
        assert pos.unrealised_pnl(1200.0) == pytest.approx(20_000.0)
        assert pos.unrealised_pnl(800.0)  == pytest.approx(-20_000.0)


class TestPortfolioEquity:
    def test_initial_equity_equals_cash(self):
        port, _, handler, _ = _make_portfolio()
        handler.get_current_price.return_value = None
        assert port.equity == pytest.approx(1_000_000.0)

    def test_equity_includes_positions(self):
        port, queue, handler, _ = _make_portfolio()
        # Simulate a BUY fill
        f = _fill(qty=100, price=1000.0)
        port.on_fill(f)
        handler.get_current_price.return_value = 1100.0  # price up 10%
        # equity = cash + 100 * 1100
        expected = port.cash + 100 * 1100.0
        assert port.equity == pytest.approx(expected, rel=1e-3)


class TestOnSignal:
    def test_long_signal_emits_order(self):
        port, queue, handler, _ = _make_portfolio()
        handler.get_current_price.return_value = 1000.0
        port.on_signal(_signal(direction=SignalDirection.LONG))
        orders = [e for e in queue if e.event_type == EventType.ORDER]
        assert len(orders) == 1
        assert orders[0].direction == OrderDirection.BUY

    def test_duplicate_long_ignored(self):
        port, queue, handler, _ = _make_portfolio()
        # First signal → enters
        port.on_signal(_signal(direction=SignalDirection.LONG))
        # Fill it
        order = queue.popleft()
        fill  = _fill(qty=order.quantity, price=1000.0, order=order)
        port.on_fill(fill)
        queue.clear()
        # Second signal → should be ignored (already invested)
        port.on_signal(_signal(direction=SignalDirection.LONG))
        assert len(queue) == 0

    def test_max_positions_respected(self):
        port, queue, handler, _ = _make_portfolio(max_open=1)
        handler.get_current_price.return_value = 1000.0

        # First signal → accepted
        port.on_signal(_signal("SYM", SignalDirection.LONG))
        order1 = queue.popleft()
        port.on_fill(_fill("SYM", qty=order1.quantity, order=order1))
        queue.clear()

        # Second signal on different symbol → rejected (max positions = 1)
        port.on_signal(_signal("SYM2", SignalDirection.LONG))
        orders = [e for e in queue if e.event_type == EventType.ORDER]
        assert len(orders) == 0

    def test_exit_long_emits_sell(self):
        port, queue, handler, _ = _make_portfolio()

        # Enter first
        port.on_signal(_signal(direction=SignalDirection.LONG))
        order = queue.popleft()
        port.on_fill(_fill(qty=order.quantity, order=order))
        queue.clear()

        # Exit
        port.on_signal(_signal(direction=SignalDirection.EXIT_LONG))
        orders = [e for e in queue if e.event_type == EventType.ORDER]
        assert len(orders) == 1
        assert orders[0].direction == OrderDirection.SELL

    def test_short_ignored_when_allow_short_false(self):
        port, queue, handler, _ = _make_portfolio(allow_short=False)
        port.on_signal(_signal(direction=SignalDirection.SHORT))
        orders = [e for e in queue if e.event_type == EventType.ORDER]
        assert len(orders) == 0

    def test_exit_without_position_does_nothing(self):
        port, queue, _, _ = _make_portfolio()
        port.on_signal(_signal(direction=SignalDirection.EXIT_LONG))
        assert len(queue) == 0


class TestOnFill:
    def test_buy_reduces_cash(self):
        port, queue, _, _ = _make_portfolio()
        initial_cash = port.cash
        f = _fill(qty=100, price=1000.0)
        port.on_fill(f)
        # cost = 100 * 1000 + 50 (commission)
        expected_cash = initial_cash - (100 * 1000.0 + 50.0)
        assert port.cash == pytest.approx(expected_cash)

    def test_buy_creates_position(self):
        port, queue, _, _ = _make_portfolio()
        f = _fill(qty=50, price=2000.0)
        port.on_fill(f)
        assert "SYM" in port.open_positions
        assert port.open_positions["SYM"].quantity == 50

    def test_sell_increases_cash(self):
        port, queue, handler, _ = _make_portfolio()
        # First buy
        f_buy = _fill(direction=OrderDirection.BUY, qty=100, price=1000.0)
        port.on_fill(f_buy)
        cash_after_buy = port.cash

        # Then sell
        f_sell = _fill(direction=OrderDirection.SELL, qty=100, price=1100.0)
        port.on_fill(f_sell)
        # proceeds = 100 * 1100 - 50 commission
        expected_cash = cash_after_buy + (100 * 1100.0 - 50.0)
        assert port.cash == pytest.approx(expected_cash)

    def test_sell_removes_position(self):
        port, queue, _, _ = _make_portfolio()
        f_buy  = _fill(qty=100, price=1000.0)
        port.on_fill(f_buy)
        f_sell = _fill(direction=OrderDirection.SELL, qty=100, price=1050.0)
        port.on_fill(f_sell)
        assert "SYM" not in port.open_positions

    def test_partial_sell_leaves_remaining(self):
        port, queue, _, _ = _make_portfolio()
        port.on_fill(_fill(qty=100, price=1000.0))
        port.on_fill(_fill(direction=OrderDirection.SELL, qty=40, price=1050.0))
        assert port.open_positions["SYM"].quantity == 60


class TestEquityCurveAndTradelog:
    def test_equity_curve_recorded(self):
        port, queue, handler, _ = _make_portfolio()
        port.update_equity_curve(datetime(2020, 1, 1))
        port.update_equity_curve(datetime(2020, 1, 2))
        curve = port.get_equity_curve()
        assert len(curve) == 2
        assert "equity" in curve.columns

    def test_trade_log_records_entry_and_exit(self):
        port, queue, handler, _ = _make_portfolio()
        f_buy  = _fill(qty=100, price=1000.0)
        f_sell = _fill(direction=OrderDirection.SELL, qty=100, price=1100.0)
        port.on_fill(f_buy)
        port.on_fill(f_sell)
        tlog = port.get_trade_log()
        assert "ENTRY" in tlog["type"].values
        assert "EXIT"  in tlog["type"].values

    def test_realised_pnl_correct(self):
        port, _, _, _ = _make_portfolio()
        port.on_fill(_fill(qty=100, price=1000.0))
        port.on_fill(_fill(direction=OrderDirection.SELL, qty=100, price=1100.0))
        tlog   = port.get_trade_log()
        exit_r = tlog[tlog["type"] == "EXIT"]["realised_pnl"].values[0]
        # pnl = 100 * (1100 - 1000) - 50 commission
        assert exit_r == pytest.approx(100 * 100.0 - 50.0)
