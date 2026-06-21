# engine/execution.py
# ============================================================
# EXECUTION HANDLER — SIMULATES NSE BROKER FILLS
# Receives OrderEvents, simulates slippage and costs, emits FillEvents.
# ============================================================

import logging
from collections import deque
from typing import Optional

from engine.events import OrderEvent, FillEvent, OrderDirection
from config import BacktestConfig

logger = logging.getLogger(__name__)


class ExecutionHandler:
    """
    Simulates a brokerage execution for NSE equity delivery trading.

    Fill Model
    ----------
    Orders fill at the NEXT bar's open price (fill_at = "next_open").
      BUY : next_open × (1 + slippage_pct)  — we pay a touch more.
      SELL: next_open × (1 - slippage_pct)  — we receive a touch less.

    Cost Model (NSE delivery equity)
    ---------------------------------
    Brokerage       : commission_pct  × trade_value  (both sides)
    Exchange charges: exchange_charges_pct × trade_value (both sides)
    Stamp duty      : stamp_duty_pct  × trade_value  (BUY only)
    STT             : stt_pct         × trade_value  (SELL only)
    """

    def __init__(self, config: BacktestConfig, event_queue: deque) -> None:
        self._config      = config
        self._event_queue = event_queue  # may be replaced by BacktestEngine
        self._fill_count  = 0

    def execute_order(self, order_event: OrderEvent, data_handler) -> None:
        """
        Simulate order execution and push a FillEvent into the queue.

        Parameters
        ----------
        order_event  : OrderEvent to execute.
        data_handler : DataHandler — used to fetch next-open price.
        """
        symbol   = order_event.symbol
        qty      = order_event.quantity
        direction = order_event.direction

        if qty <= 0:
            logger.warning("Order for %s has qty=%d — skipped.", symbol, qty)
            return

        # ── Fill price ────────────────────────────────────────────────────
        raw_price = self._get_fill_price(order_event, data_handler)
        if raw_price is None:
            logger.warning("Cannot fill order for %s — price unavailable.", symbol)
            return

        fill_price      = self._apply_slippage(direction, raw_price)
        commission      = self._calculate_total_cost(direction, qty, fill_price)
        slippage_amount = abs(fill_price - raw_price) * qty

        # ── Fill timestamp ────────────────────────────────────────────────
        fill_ts = data_handler.get_current_datetime()
        if fill_ts is None:
            fill_ts = order_event.timestamp

        fill = FillEvent(
            timestamp=fill_ts,
            symbol=symbol,
            direction=direction,
            quantity=qty,
            fill_price=round(fill_price, 4),
            commission=round(commission, 4),
            slippage=round(slippage_amount, 4),
            order_ref=order_event,
        )
        self._event_queue.append(fill)
        self._fill_count += 1

        logger.debug(
            "FILL %s %d × %s @ ₹%.2f  cost=₹%.2f",
            direction.value, qty, symbol, fill_price, commission,
        )

    # ──────────────────────────────────────────────────────────────────────
    # DECOMPOSED HELPERS
    # ──────────────────────────────────────────────────────────────────────

    def _get_fill_price(
        self, order_event: OrderEvent, data_handler
    ) -> Optional[float]:
        """
        Determine the raw (pre-slippage) fill price.

        "next_open"  — fill at the next bar's open (eliminates lookahead bias).
        "same_close" — fill at the current bar's close (for debugging only).
        """
        symbol = order_event.symbol
        if self._config.execution.fill_at == "next_open":
            price = data_handler.get_next_open(symbol)
            if price is None:
                price = data_handler.get_current_price(symbol)
        else:
            price = data_handler.get_current_price(symbol)
        return price

    def _apply_slippage(self, direction: OrderDirection, raw_price: float) -> float:
        """Adjust raw price for market impact (slippage)."""
        sp = self._config.execution.slippage_pct
        if direction == OrderDirection.BUY:
            return raw_price * (1.0 + sp)
        return raw_price * (1.0 - sp)

    def _calculate_total_cost(
        self, direction: OrderDirection, quantity: int, fill_price: float
    ) -> float:
        """
        Calculate total transaction cost (INR).

        BUY  side: brokerage + exchange + stamp duty
        SELL side: brokerage + exchange + STT
        """
        ecfg        = self._config.execution
        trade_value = quantity * fill_price
        brokerage   = trade_value * ecfg.commission_pct
        exchange    = trade_value * ecfg.exchange_charges_pct

        if direction == OrderDirection.BUY:
            stamp = trade_value * ecfg.stamp_duty_pct
            stt   = 0.0
        else:
            stt   = trade_value * ecfg.stt_pct
            stamp = 0.0

        return brokerage + exchange + stt + stamp
