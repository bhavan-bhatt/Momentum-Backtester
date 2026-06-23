# engine/execution_advanced.py
# ============================================================
# LIQUIDITY-AWARE EXECUTION — SUPERSEDES PHASE 1's FLAT SLIPPAGE MODEL
# ============================================================

import logging
from collections import deque
from typing import List, Optional

import pandas as pd

from config import BacktestConfig
from engine.events import FillEvent, OrderDirection, OrderEvent
from engine.execution import ExecutionHandler

logger = logging.getLogger(__name__)


class AdvancedExecutionHandler(ExecutionHandler):
    """
    Extends ExecutionHandler with square-root market impact slippage and
    participation-rate capacity constraints.
    """

    def __init__(self, config: BacktestConfig, event_queue: deque) -> None:
        super().__init__(config, event_queue)
        aecfg = config.advanced.advanced_execution
        self.use_liquidity_aware = aecfg.use_liquidity_aware_slippage
        self.participation_rate_limit = aecfg.participation_rate_limit
        self.impact_coefficient = aecfg.impact_coefficient
        self.min_slippage_pct = aecfg.min_slippage_pct
        self.capacity_constrained_orders: List[dict] = []

    def execute_order(self, order_event: OrderEvent, data_handler) -> None:
        """Execute with liquidity-aware slippage and capacity constraints."""
        avg_volume = self._get_avg_daily_volume(order_event.symbol, data_handler, window=20)

        if avg_volume is None or not self.use_liquidity_aware:
            super().execute_order(order_event, data_handler)
            return

        original_qty = order_event.quantity
        participation = original_qty / avg_volume

        if participation > self.participation_rate_limit:
            capped_quantity = int(avg_volume * self.participation_rate_limit)
            logger.warning(
                "Order for %s reduced from %d to %d shares — exceeds %.0f%% of avg daily volume.",
                order_event.symbol,
                original_qty,
                capped_quantity,
                self.participation_rate_limit * 100,
            )
            self.capacity_constrained_orders.append({
                "date": data_handler.get_current_datetime(),
                "symbol": order_event.symbol,
                "requested_qty": original_qty,
                "filled_qty": capped_quantity,
                "avg_daily_volume": avg_volume,
                "participation_pct": participation,
            })
            order_event.quantity = max(capped_quantity, 0)
            if order_event.quantity == 0:
                logger.warning(
                    "Order for %s fully capacity-blocked — skipping.",
                    order_event.symbol,
                )
                return

        raw_price = self._get_fill_price(order_event, data_handler)
        if raw_price is None:
            logger.warning("Cannot fill order for %s — price unavailable.", order_event.symbol)
            return

        slippage_pct = self._calculate_impact_slippage(order_event.quantity, avg_volume)
        if order_event.direction == OrderDirection.BUY:
            fill_price = raw_price * (1.0 + slippage_pct)
        else:
            fill_price = raw_price * (1.0 - slippage_pct)

        commission = self._calculate_total_cost(
            order_event.direction, order_event.quantity, fill_price
        )
        slippage_amount = abs(fill_price - raw_price) * order_event.quantity

        fill_ts = data_handler.get_current_datetime()
        if fill_ts is None:
            fill_ts = order_event.timestamp

        fill = FillEvent(
            timestamp=fill_ts,
            symbol=order_event.symbol,
            direction=order_event.direction,
            quantity=order_event.quantity,
            fill_price=round(fill_price, 4),
            commission=round(commission, 4),
            slippage=round(slippage_amount, 4),
            order_ref=order_event,
        )
        self._event_queue.append(fill)
        self._fill_count += 1

        logger.debug(
            "FILL %s qty=%d participation=%.2%% slippage=%.3%%",
            order_event.symbol,
            order_event.quantity,
            (order_event.quantity / avg_volume) * 100,
            slippage_pct * 100,
        )

    def _get_avg_daily_volume(
        self, symbol: str, data_handler, window: int = 20
    ) -> Optional[float]:
        """Trailing average daily volume for a symbol."""
        bars = data_handler.get_latest_bars(symbol, window)
        if bars is None or "volume" not in bars.columns or bars["volume"].isna().all():
            return None
        avg_volume = float(bars["volume"].dropna().mean())
        if avg_volume <= 0:
            return None
        return avg_volume

    def _calculate_impact_slippage(self, quantity: int, avg_volume: float) -> float:
        """Square-root market impact slippage."""
        participation = quantity / avg_volume
        raw_impact = self.impact_coefficient * (participation ** 0.5)
        return max(raw_impact, self.min_slippage_pct)

    def get_capacity_report(self) -> pd.DataFrame:
        """Summarise orders reduced due to participation rate limits."""
        if not self.capacity_constrained_orders:
            return pd.DataFrame(
                columns=[
                    "date",
                    "symbol",
                    "requested_qty",
                    "filled_qty",
                    "avg_daily_volume",
                    "participation_pct",
                ]
            )
        return pd.DataFrame(self.capacity_constrained_orders)
