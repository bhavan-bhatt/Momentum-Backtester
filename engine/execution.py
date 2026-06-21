# engine/execution.py
# ============================================================
# SIMULATED BROKER + SLIPPAGE + TRANSACTION COST MODEL
# Receives OrderEvents and emits FillEvents.
# ============================================================

import logging
from collections import deque

from engine.events import OrderEvent, OrderDirection, FillEvent
from engine.data_handler import DataHandler
from config import BacktestConfig

logger = logging.getLogger(__name__)


class ExecutionHandler:
    """
    Simulates order execution with realistic Indian market transaction costs.

    Execution Model
    ---------------
    - Orders are filled at the NEXT bar's open price (fill_at="next_open").
    - Slippage is applied symmetrically: BUY fills higher, SELL fills lower.
    - Transaction costs:
        BUY  side: commission + exchange charges + stamp duty
        SELL side: commission + exchange charges + STT

    Cost Breakdown (per NSE delivery segment)
    ------------------------------------------
    commission        : 0.03% of trade value (both sides)
    exchange charges  : 0.00345% of trade value (both sides)
    stamp_duty        : 0.015% of trade value (BUY side only)
    STT               : 0.1% of trade value (SELL side only)
    """

    def __init__(
        self,
        config: BacktestConfig,
        data_handler: DataHandler,
        event_queue: deque,
    ) -> None:
        self._config      = config
        self._data        = data_handler
        self._event_queue = event_queue

    def on_order(self, order: OrderEvent) -> None:
        """
        Process an OrderEvent and emit a FillEvent.

        Steps
        -----
        1. Determine the fill price (next-open + slippage).
        2. Compute transaction costs.
        3. Build and queue a FillEvent.
        """
        symbol    = order.symbol
        direction = order.direction
        quantity  = order.quantity
        ecfg      = self._config.execution

        if quantity <= 0:
            logger.warning("Order for %s has quantity %d — skipped.", symbol, quantity)
            return

        # ── Fill price ───────────────────────────────────────────────────
        if ecfg.fill_at == "next_open":
            base_price = self._data.get_next_open(symbol)
        else:
            # "same_close" — unrealistic but supported for debugging
            base_price = self._data.get_current_price(symbol)

        if base_price is None or base_price <= 0:
            logger.warning(
                "Cannot fill order for %s — price unavailable. Order dropped.", symbol
            )
            return

        # Slippage: BUY pays more, SELL receives less
        if direction == OrderDirection.BUY:
            fill_price = base_price * (1.0 + ecfg.slippage_pct)
        else:
            fill_price = base_price * (1.0 - ecfg.slippage_pct)

        slippage_cost = abs(fill_price - base_price) * quantity

        # ── Transaction costs ────────────────────────────────────────────
        trade_value = fill_price * quantity
        commission  = trade_value * ecfg.commission_pct
        exchange    = trade_value * ecfg.exchange_charges_pct

        if direction == OrderDirection.BUY:
            stamp_duty  = trade_value * ecfg.stamp_duty_pct
            total_cost  = commission + exchange + stamp_duty
        else:
            stt         = trade_value * ecfg.stt_pct
            total_cost  = commission + exchange + stt

        # ── Determine fill timestamp ─────────────────────────────────────
        # If filling at next open, the timestamp advances by one bar.
        # We use the current datetime from DataHandler (already incremented).
        fill_ts = self._data.get_current_datetime()
        if fill_ts is None:
            fill_ts = order.timestamp

        fill = FillEvent(
            timestamp=fill_ts,
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            fill_price=round(fill_price, 4),
            commission=round(total_cost, 4),
            slippage=round(slippage_cost, 4),
            order_ref=order,
        )
        self._event_queue.append(fill)

        logger.debug(
            "FILL: %s %s %d @ %.2f (slip=%.2f, cost=%.2f)",
            direction.value, symbol, quantity,
            fill_price, slippage_cost, total_cost,
        )
