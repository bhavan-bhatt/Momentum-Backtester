# tests/test_events.py
# ============================================================
# UNIT TESTS — engine/events.py
# ============================================================

import pytest
from datetime import datetime

from engine.events import (
    EventType,
    SignalDirection,
    OrderDirection,
    MarketEvent,
    SignalEvent,
    OrderEvent,
    FillEvent,
)


class TestEventType:
    def test_all_members_exist(self):
        assert EventType.MARKET
        assert EventType.SIGNAL
        assert EventType.ORDER
        assert EventType.FILL

    def test_values_are_strings(self):
        for member in EventType:
            assert isinstance(member.value, str)


class TestMarketEvent:
    def test_default_event_type(self):
        e = MarketEvent()
        assert e.event_type == EventType.MARKET

    def test_cannot_override_event_type(self):
        """event_type is a field with init=False; it cannot be set via constructor."""
        e = MarketEvent(timestamp=datetime(2020, 1, 1), symbol="RELIANCE.NS", close=100.0)
        assert e.event_type == EventType.MARKET

    def test_field_assignment(self):
        ts = datetime(2021, 6, 15)
        e = MarketEvent(
            timestamp=ts,
            symbol="TCS.NS",
            open=3100.0,
            high=3200.0,
            low=3050.0,
            close=3150.0,
            volume=1_000_000.0,
            is_last_bar=True,
        )
        assert e.timestamp == ts
        assert e.symbol == "TCS.NS"
        assert e.open == 3100.0
        assert e.close == 3150.0
        assert e.is_last_bar is True

    def test_optional_fields_default_to_none(self):
        e = MarketEvent(timestamp=datetime(2020, 1, 1), symbol="X", close=10.0)
        assert e.open is None
        assert e.high is None
        assert e.low is None
        assert e.volume is None


class TestSignalEvent:
    def test_default_event_type(self):
        s = SignalEvent()
        assert s.event_type == EventType.SIGNAL

    def test_strength_clamp_not_enforced_at_dataclass_level(self):
        # The dataclass itself doesn't clamp; Strategy._emit_signal does.
        s = SignalEvent(strength=2.0)
        assert s.strength == 2.0

    def test_direction_default(self):
        s = SignalEvent()
        assert s.direction == SignalDirection.LONG


class TestOrderEvent:
    def test_default_order_type(self):
        o = OrderEvent()
        assert o.order_type == "MARKET"

    def test_signal_ref_linkage(self):
        signal = SignalEvent(symbol="INFY.NS", direction=SignalDirection.EXIT_LONG)
        order  = OrderEvent(symbol="INFY.NS", quantity=50, signal_ref=signal)
        assert order.signal_ref is signal
        assert order.signal_ref.symbol == "INFY.NS"


class TestFillEvent:
    def test_default_event_type(self):
        f = FillEvent()
        assert f.event_type == EventType.FILL

    def test_order_ref_linkage(self):
        order  = OrderEvent(symbol="WIPRO.NS", quantity=100)
        fill   = FillEvent(symbol="WIPRO.NS", quantity=100, fill_price=450.0, order_ref=order)
        assert fill.order_ref is order

    def test_commission_and_slippage_defaults(self):
        f = FillEvent()
        assert f.commission == 0.0
        assert f.slippage == 0.0


class TestSignalDirection:
    def test_all_directions(self):
        for d in [SignalDirection.LONG, SignalDirection.SHORT,
                  SignalDirection.EXIT_LONG, SignalDirection.EXIT_SHORT]:
            assert isinstance(d.value, str)


class TestOrderDirection:
    def test_buy_sell(self):
        assert OrderDirection.BUY.value == "BUY"
        assert OrderDirection.SELL.value == "SELL"
