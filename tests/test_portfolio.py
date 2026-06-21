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
from engine.portfolio import PortfolioManager
from config import BacktestConfig


# ──────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ──────────────────────────────────────────────────────────────────────────────

def _make_portfolio(
    initial_capital: float = 1_000_000.0,
    sizing_method: str = "equal_weight",
    max_open: int = 5,
    allow_short: bool = False,
) -> tuple:
    cfg = BacktestConfig()
    cfg.portfolio.initial_capital    = initial_capital
    cfg.portfolio.sizing_method      = sizing_method
    cfg.portfolio.max_open_positions = max_open
    cfg.strategy.allow_short         = allow_short
    cfg.portfolio.max_position_pct   = 0.30
    cfg.portfolio.risk_per_trade_pct = 0.02

    queue   = deque()
    handler = MagicMock()
    handler.get_current_price.return_value = 1000.0
    handler.has_high_low = {"SYM": True, "SYM2": True}

    port = PortfolioManager(cfg, queue)
    return port, queue, handler, cfg


def _signal(
    symbol: str = "SYM",
    direction: SignalDirection = SignalDirection.LONG,
) -> SignalEvent:
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
    commission: float = 50.0,
    order: "OrderEvent | None" = None,
) -> FillEvent:
    order_ref = order or OrderEvent(symbol=symbol, direction=direction, quantity=qty)
    return FillEvent(
        timestamp=datetime(2020, 3, 2),
        symbol=symbol,
        direction=direction,
        quantity=qty,
        fill_price=price,
        commission=commission,
        slippage=5.0,
        order_ref=order_ref,
    )


# ──────────────────────────────────────────────────────────────────────────────
# PORTFOLIO EQUITY / INITIALISATION
# ──────────────────────────────────────────────────────────────────────────────

class TestPortfolioInit:
    def test_initial_cash_equals_capital(self):
        port, _, handler, _ = _make_portfolio()
        assert port.cash == pytest.approx(1_000_000.0)

    def test_no_holdings_at_start(self):
        port, _, _, _ = _make_portfolio()
        assert port.holdings == {}

    def test_empty_equity_curve_at_start(self):
        port, _, _, _ = _make_portfolio()
        assert port.equity_curve == []


class TestPortfolioEquity:
    def test_initial_equity_equals_cash(self):
        port, _, handler, _ = _make_portfolio()
        handler.get_current_price.return_value = None
        total = port.get_total_equity(handler)
        assert total == pytest.approx(1_000_000.0)

    def test_equity_includes_mark_to_market(self):
        port, queue, handler, _ = _make_portfolio()
        f = _fill(qty=100, price=1000.0)
        port.update_portfolio_on_fill(f)
        handler.get_current_price.return_value = 1100.0
        expected = port.cash + 100 * 1100.0
        assert port.get_total_equity(handler) == pytest.approx(expected, rel=1e-3)

    def test_mark_to_market_uses_avg_cost_when_price_none(self):
        port, queue, handler, _ = _make_portfolio()
        port.update_portfolio_on_fill(_fill(qty=50, price=2000.0))
        handler.get_current_price.return_value = None
        # Should fall back to avg_cost
        mtm = port._mark_to_market_value(handler)
        assert mtm == pytest.approx(50 * 2000.0)


# ──────────────────────────────────────────────────────────────────────────────
# PROCESS SIGNAL → ORDER EMISSION
# ──────────────────────────────────────────────────────────────────────────────

class TestProcessSignal:
    def test_long_signal_emits_buy_order(self):
        port, queue, handler, _ = _make_portfolio()
        port.process_signal(_signal(direction=SignalDirection.LONG), handler)
        orders = [e for e in queue if e.event_type == EventType.ORDER]
        assert len(orders) == 1
        assert orders[0].direction == OrderDirection.BUY

    def test_duplicate_long_ignored_after_fill(self):
        port, queue, handler, _ = _make_portfolio()
        port.process_signal(_signal(direction=SignalDirection.LONG), handler)
        order = queue.popleft()
        port.update_portfolio_on_fill(_fill(qty=order.quantity, order=order))
        queue.clear()
        # Second signal for same symbol → blocked by holdings check
        port.process_signal(_signal(direction=SignalDirection.LONG), handler)
        assert len(queue) == 0

    def test_pending_order_prevents_duplicate(self):
        port, queue, handler, _ = _make_portfolio()
        port.process_signal(_signal(direction=SignalDirection.LONG), handler)
        queue.clear()  # do NOT fill — leave order pending
        # Second signal before fill → blocked by _pending_orders
        port.process_signal(_signal(direction=SignalDirection.LONG), handler)
        assert len(queue) == 0

    def test_max_positions_respected(self):
        port, queue, handler, _ = _make_portfolio(max_open=1)
        handler.get_current_price.return_value = 1000.0

        port.process_signal(_signal("SYM", SignalDirection.LONG), handler)
        order1 = queue.popleft()
        port.update_portfolio_on_fill(_fill("SYM", qty=order1.quantity, order=order1))
        queue.clear()

        # Different symbol — max positions already reached
        port.process_signal(_signal("SYM2", SignalDirection.LONG), handler)
        assert len([e for e in queue if e.event_type == EventType.ORDER]) == 0

    def test_exit_long_emits_sell_order(self):
        port, queue, handler, _ = _make_portfolio()
        port.process_signal(_signal(direction=SignalDirection.LONG), handler)
        order = queue.popleft()
        port.update_portfolio_on_fill(_fill(qty=order.quantity, order=order))
        queue.clear()

        port.process_signal(_signal(direction=SignalDirection.EXIT_LONG), handler)
        orders = [e for e in queue if e.event_type == EventType.ORDER]
        assert len(orders) == 1
        assert orders[0].direction == OrderDirection.SELL

    def test_short_ignored_when_allow_short_false(self):
        port, queue, handler, _ = _make_portfolio(allow_short=False)
        port.process_signal(_signal(direction=SignalDirection.SHORT), handler)
        assert len([e for e in queue if e.event_type == EventType.ORDER]) == 0

    def test_exit_without_position_does_nothing(self):
        port, queue, handler, _ = _make_portfolio()
        port.process_signal(_signal(direction=SignalDirection.EXIT_LONG), handler)
        assert len(queue) == 0

    def test_no_price_skips_entry(self):
        port, queue, handler, _ = _make_portfolio()
        handler.get_current_price.return_value = None
        port.process_signal(_signal(direction=SignalDirection.LONG), handler)
        assert len(queue) == 0


# ──────────────────────────────────────────────────────────────────────────────
# UPDATE ON FILL — CASH / HOLDINGS
# ──────────────────────────────────────────────────────────────────────────────

class TestUpdateOnFill:
    def test_buy_reduces_cash(self):
        port, _, _, _ = _make_portfolio()
        initial_cash = port.cash
        port.update_portfolio_on_fill(_fill(qty=100, price=1000.0, commission=50.0))
        expected = initial_cash - (100 * 1000.0 + 50.0)
        assert port.cash == pytest.approx(expected)

    def test_buy_creates_holding(self):
        port, _, _, _ = _make_portfolio()
        port.update_portfolio_on_fill(_fill(qty=50, price=2000.0))
        assert "SYM" in port.holdings
        assert port.holdings["SYM"]["quantity"] == 50

    def test_buy_records_avg_cost(self):
        port, _, _, _ = _make_portfolio()
        port.update_portfolio_on_fill(_fill(qty=100, price=1500.0))
        assert port.holdings["SYM"]["avg_cost"] == pytest.approx(1500.0)

    def test_sell_increases_cash(self):
        port, _, _, _ = _make_portfolio()
        port.update_portfolio_on_fill(_fill(qty=100, price=1000.0, commission=50.0))
        cash_after_buy = port.cash
        port.update_portfolio_on_fill(
            _fill(direction=OrderDirection.SELL, qty=100, price=1100.0, commission=50.0)
        )
        expected = cash_after_buy + (100 * 1100.0 - 50.0)
        assert port.cash == pytest.approx(expected)

    def test_sell_removes_holding(self):
        port, _, _, _ = _make_portfolio()
        port.update_portfolio_on_fill(_fill(qty=100, price=1000.0))
        port.update_portfolio_on_fill(
            _fill(direction=OrderDirection.SELL, qty=100, price=1050.0)
        )
        assert "SYM" not in port.holdings

    def test_sell_appends_trade_log(self):
        port, _, _, _ = _make_portfolio()
        port.update_portfolio_on_fill(_fill(qty=100, price=1000.0))
        port.update_portfolio_on_fill(
            _fill(direction=OrderDirection.SELL, qty=100, price=1100.0)
        )
        assert len(port.trade_log) == 1

    def test_sell_without_position_is_ignored(self):
        port, _, _, _ = _make_portfolio()
        port.update_portfolio_on_fill(
            _fill(direction=OrderDirection.SELL, qty=100, price=1000.0)
        )
        assert len(port.trade_log) == 0

    def test_pending_order_cleared_on_buy_fill(self):
        port, queue, handler, _ = _make_portfolio()
        port.process_signal(_signal(), handler)
        order = queue.popleft()
        assert "SYM" in port._pending_orders
        port.update_portfolio_on_fill(_fill(qty=order.quantity, order=order))
        assert "SYM" not in port._pending_orders


# ──────────────────────────────────────────────────────────────────────────────
# TRADE LOG CONTENTS
# ──────────────────────────────────────────────────────────────────────────────

class TestTradeLog:
    def _round_trip(self):
        port, _, _, _ = _make_portfolio()
        port.update_portfolio_on_fill(_fill(qty=100, price=1000.0, commission=50.0))
        port.update_portfolio_on_fill(
            _fill(direction=OrderDirection.SELL, qty=100, price=1100.0, commission=50.0)
        )
        return port

    def test_trade_log_has_one_entry(self):
        port = self._round_trip()
        assert len(port.get_trade_log()) == 1

    def test_trade_log_required_keys(self):
        port = self._round_trip()
        t = port.get_trade_log()[0]
        for key in ("symbol", "entry_date", "exit_date", "net_pnl",
                    "gross_pnl", "commission", "return_pct", "quantity"):
            assert key in t, f"Missing key: {key}"

    def test_trade_net_pnl_correct(self):
        port = self._round_trip()
        t = port.get_trade_log()[0]
        # gross = (1100 - 1000) * 100 = 10_000; net = 10_000 - 50 = 9_950
        assert t["net_pnl"] == pytest.approx(9_950.0)

    def test_trade_return_pct_positive(self):
        port = self._round_trip()
        t = port.get_trade_log()[0]
        assert t["return_pct"] > 0


# ──────────────────────────────────────────────────────────────────────────────
# EQUITY CURVE RECORDING
# ──────────────────────────────────────────────────────────────────────────────

class TestEquityCurve:
    def test_record_equity_appends(self):
        port, _, handler, _ = _make_portfolio()
        handler.get_current_price.return_value = None
        port.record_equity(datetime(2020, 1, 1), handler)
        port.record_equity(datetime(2020, 1, 2), handler)
        assert len(port.equity_curve) == 2

    def test_calculate_equity_curve_returns_series(self):
        port, _, handler, _ = _make_portfolio()
        handler.get_current_price.return_value = None
        port.record_equity(datetime(2020, 1, 1), handler)
        port.record_equity(datetime(2020, 1, 2), handler)
        curve = port.calculate_equity_curve()
        assert isinstance(curve, pd.Series)
        assert len(curve) == 2

    def test_equity_value_equals_cash_with_no_positions(self):
        port, _, handler, _ = _make_portfolio()
        handler.get_current_price.return_value = None
        port.record_equity(datetime(2020, 1, 1), handler)
        curve = port.calculate_equity_curve()
        assert curve.iloc[0] == pytest.approx(1_000_000.0)


# ──────────────────────────────────────────────────────────────────────────────
# PORTFOLIO SUMMARY
# ──────────────────────────────────────────────────────────────────────────────

class TestPortfolioSummary:
    def test_summary_keys_present(self):
        port, _, handler, _ = _make_portfolio()
        summary = port.get_portfolio_summary(handler)
        for key in ("cash", "invested_value", "total_equity",
                    "num_positions", "open_positions", "total_trades"):
            assert key in summary

    def test_summary_no_positions(self):
        port, _, handler, _ = _make_portfolio()
        handler.get_current_price.return_value = None
        summary = port.get_portfolio_summary(handler)
        assert summary["num_positions"] == 0
        assert summary["invested_value"] == pytest.approx(0.0)

    def test_get_open_positions_returns_copy(self):
        port, _, _, _ = _make_portfolio()
        port.update_portfolio_on_fill(_fill(qty=100, price=1000.0))
        positions = port.get_open_positions()
        assert "SYM" in positions
        positions.pop("SYM")  # modifying copy should not affect original
        assert "SYM" in port.holdings
