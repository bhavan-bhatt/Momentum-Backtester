# engine/portfolio.py
# ============================================================
# PORTFOLIO MANAGER — POSITION SIZING, CASH, AND HOLDINGS
# Receives SignalEvents, computes order size, emits OrderEvents.
# Receives FillEvents, updates cash and holdings.
# ============================================================

import logging
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from engine.events import (
    SignalEvent, OrderEvent, FillEvent,
    OrderDirection, SignalDirection,
)
from config import BacktestConfig

logger = logging.getLogger(__name__)


class PortfolioManager:
    """
    Manages the portfolio's capital, positions, and risk controls.

    Ensemble mode gives each strategy sleeve its own cash pool and holdings
    so sleeves do not block one another on the same symbol.
    """

    def __init__(self, config: BacktestConfig, event_queue: deque) -> None:
        self._config      = config
        self._event_queue = event_queue

        self.cash             = config.portfolio.initial_capital
        self.holdings:         Dict[str, dict]         = {}
        self.equity_curve:     List[tuple]             = []
        self.trade_log:        List[dict]              = []
        self._pending_orders:  Dict[str, OrderEvent]   = {}
        self._ensemble_enabled: bool = False
        self._sleeve_ids:      List[str]               = []
        self._sleeve_cash:     Dict[str, float]        = {}
        self._sleeve_holdings: Dict[str, Dict[str, dict]] = {}

    def enable_ensemble_mode(self, strategy_ids: Optional[List[str]] = None) -> None:
        """
        Split capital across strategy sleeves so each can trade independently.

        Each sleeve receives an equal cash allocation. signal.strength is NOT
        applied again to quantity (capital is already split).
        """
        self._ensemble_enabled = True
        if not strategy_ids:
            logger.warning("Ensemble mode enabled without strategy_ids — using shared pool.")
            return

        self._sleeve_ids = list(strategy_ids)
        n = len(strategy_ids)
        per_sleeve = self._config.portfolio.initial_capital / n
        self._sleeve_cash = {sid: per_sleeve for sid in strategy_ids}
        self._sleeve_holdings = {sid: {} for sid in strategy_ids}
        self.cash = 0.0
        self.holdings = {}
        logger.info(
            "Ensemble split-capital mode: %d sleeves × ₹%.0f each.",
            n, per_sleeve,
        )

    def _resolve_sleeve_id(self, strategy_id: str) -> Optional[str]:
        """Map a signal's strategy_id to a registered sleeve id."""
        if strategy_id in self._sleeve_holdings:
            return strategy_id
        for sid in self._sleeve_ids:
            prefix = sid.split("_")[0]
            if strategy_id.startswith(prefix) or sid in strategy_id:
                return sid
            if prefix.lower() in strategy_id.lower():
                return sid
        return strategy_id if strategy_id in self._sleeve_holdings else None

    def _order_key(self, sleeve_id: Optional[str], symbol: str) -> str:
        if self._ensemble_enabled and sleeve_id:
            return f"{sleeve_id}:{symbol}"
        return symbol

    def _get_book(
        self, signal_event: SignalEvent
    ) -> Tuple[float, Dict[str, dict], Optional[str]]:
        """Return (cash, holdings, sleeve_id) for this signal."""
        if not self._ensemble_enabled:
            return self.cash, self.holdings, None

        sleeve_id = self._resolve_sleeve_id(signal_event.strategy_id)
        if sleeve_id is None or sleeve_id not in self._sleeve_holdings:
            logger.debug(
                "Unknown sleeve for strategy_id=%s — signal skipped.",
                signal_event.strategy_id,
            )
            return 0.0, {}, None

        return (
            self._sleeve_cash.get(sleeve_id, 0.0),
            self._sleeve_holdings[sleeve_id],
            sleeve_id,
        )

    def _sleeve_mtm(self, holdings: Dict[str, dict], data_handler) -> float:
        total = 0.0
        for symbol, pos in holdings.items():
            price = data_handler.get_current_price(symbol)
            if price is not None:
                total += pos["quantity"] * price
            else:
                total += pos["quantity"] * pos["avg_cost"]
        return total

    # ──────────────────────────────────────────────────────────────────────
    # SIGNAL → ORDER
    # ──────────────────────────────────────────────────────────────────────

    def process_signal(self, signal_event: SignalEvent, data_handler) -> None:
        """Convert a SignalEvent into an OrderEvent after checking risk limits."""
        symbol    = signal_event.symbol
        direction = signal_event.direction
        pcfg      = self._config.portfolio
        scfg      = self._config.strategy

        cash, holdings, sleeve_id = self._get_book(signal_event)
        if self._ensemble_enabled and sleeve_id is None:
            return

        order_key = self._order_key(sleeve_id, symbol)

        if direction in (SignalDirection.LONG, SignalDirection.SHORT):
            if direction == SignalDirection.SHORT and not scfg.allow_short:
                return

            price = data_handler.get_current_price(symbol)
            if price is None:
                return

            if symbol in holdings or order_key in self._pending_orders:
                return

            open_slots = len(holdings)
            if sleeve_id:
                open_slots += sum(
                    1 for k in self._pending_orders if k.startswith(f"{sleeve_id}:")
                )
            else:
                open_slots += len(self._pending_orders)
            if open_slots >= pcfg.max_open_positions:
                return

            qty = self._compute_position_size(
                signal_event, data_handler, price, cash, holdings
            )
            if qty <= 0:
                return

            order_dir = (
                OrderDirection.BUY if direction == SignalDirection.LONG
                else OrderDirection.SELL
            )
            order = OrderEvent(
                timestamp=signal_event.timestamp,
                symbol=symbol,
                direction=order_dir,
                quantity=qty,
                signal_ref=signal_event,
            )
            self._event_queue.append(order)
            self._pending_orders[order_key] = order

        elif direction == SignalDirection.EXIT_LONG:
            if symbol not in holdings:
                return
            qty = holdings[symbol]["quantity"]
            self._event_queue.append(OrderEvent(
                timestamp=signal_event.timestamp,
                symbol=symbol,
                direction=OrderDirection.SELL,
                quantity=qty,
                signal_ref=signal_event,
            ))

        elif direction == SignalDirection.EXIT_SHORT:
            if symbol not in holdings:
                return
            qty = holdings[symbol]["quantity"]
            self._event_queue.append(OrderEvent(
                timestamp=signal_event.timestamp,
                symbol=symbol,
                direction=OrderDirection.BUY,
                quantity=qty,
                signal_ref=signal_event,
            ))

    # ──────────────────────────────────────────────────────────────────────
    # POSITION SIZING
    # ──────────────────────────────────────────────────────────────────────

    def _compute_position_size(
        self,
        signal: SignalEvent,
        data_handler,
        current_price: float,
        cash: Optional[float] = None,
        holdings: Optional[Dict[str, dict]] = None,
    ) -> int:
        """Determine share quantity based on the configured sizing method."""
        pcfg   = self._config.portfolio
        scfg   = self._config.strategy
        method = pcfg.sizing_method

        if cash is None:
            cash = self.cash
        if holdings is None:
            holdings = self.holdings

        invested = self._sleeve_mtm(holdings, data_handler)
        total = cash + invested

        if method == "fixed":
            qty = pcfg.fixed_quantity
            return qty if qty * current_price <= cash else 0

        if method == "equal_weight":
            qty = int((total * pcfg.max_position_pct) / current_price)
        elif method == "atr":
            qty = self._atr_quantity(signal.symbol, data_handler, total, current_price)
        else:
            qty = int((total * pcfg.max_position_pct) / current_price)

        max_by_cap = int((total * pcfg.max_position_pct) / current_price)
        qty = min(qty, max_by_cap)

        cost_available = (total * pcfg.max_total_exposure_pct) - invested
        max_by_exposure = int(cost_available / current_price)
        qty = min(qty, max_by_exposure)

        if qty <= 0:
            return 0

        # Shared-pool ensemble only: scale by sleeve weight. Split-capital skips this.
        if self._ensemble_enabled and not self._sleeve_ids:
            qty = int(qty * max(0.0, min(1.0, signal.strength)))
            if qty <= 0:
                return 0

        if qty * current_price > cash:
            return 0

        return qty

    def _atr_quantity(
        self, symbol: str, data_handler, total: float, price: float
    ) -> int:
        """ATR-based sizing with equal-weight fallback."""
        scfg  = self._config.strategy
        pcfg  = self._config.portfolio
        bars  = data_handler.get_latest_bars(symbol, 20)

        atr_val = None
        if bars is not None and len(bars) >= scfg.atr_period + 1:
            has_hl = data_handler.has_high_low.get(symbol, False)
            if has_hl:
                from engine.strategy import atr as compute_atr
                atr_series = compute_atr(
                    bars["high"], bars["low"], bars["close"], scfg.atr_period
                )
                last = atr_series.iloc[-1]
                if not np.isnan(last) and last > 0:
                    atr_val = float(last)

        if atr_val is None:
            return int((total * pcfg.max_position_pct) / price)

        risk_amount   = total * pcfg.risk_per_trade_pct
        stop_distance = atr_val * scfg.atr_stop_multiplier
        return int(risk_amount / stop_distance)

    # ──────────────────────────────────────────────────────────────────────
    # FILL → UPDATE HOLDINGS
    # ──────────────────────────────────────────────────────────────────────

    def update_portfolio_on_fill(self, fill_event: FillEvent) -> None:
        """Update cash, holdings, and trade log after a fill is received."""
        symbol     = fill_event.symbol
        qty        = fill_event.quantity
        price      = fill_event.fill_price
        commission = fill_event.commission
        direction  = fill_event.direction

        strategy_id = ""
        if fill_event.order_ref and fill_event.order_ref.signal_ref:
            strategy_id = fill_event.order_ref.signal_ref.strategy_id

        if self._ensemble_enabled and self._sleeve_ids:
            sleeve_id = self._resolve_sleeve_id(strategy_id)
            if sleeve_id is None:
                logger.warning("Fill for unknown sleeve %s — ignored.", strategy_id)
                return
            holdings = self._sleeve_holdings[sleeve_id]
            order_key = self._order_key(sleeve_id, symbol)
        else:
            sleeve_id = None
            holdings = self.holdings
            order_key = symbol

        if direction == OrderDirection.BUY:
            trade_cost = qty * price + commission
            if sleeve_id:
                self._sleeve_cash[sleeve_id] -= trade_cost
            else:
                self.cash -= trade_cost

            self._pending_orders.pop(order_key, None)

            if symbol not in holdings:
                holdings[symbol] = {
                    "quantity":    qty,
                    "avg_cost":    price,
                    "entry_date":  fill_event.timestamp,
                    "entry_price": price,
                    "strategy_id": strategy_id,
                }
            else:
                pos = holdings[symbol]
                total_qty = pos["quantity"] + qty
                pos["avg_cost"] = (pos["quantity"] * pos["avg_cost"] + qty * price) / total_qty
                pos["quantity"] = total_qty

        elif direction == OrderDirection.SELL:
            self._pending_orders.pop(order_key, None)

            if symbol not in holdings:
                logger.warning("SELL fill for %s but no open position — ignored.", symbol)
                return

            pos = holdings.pop(symbol)
            proceeds = qty * price - commission
            if sleeve_id:
                self._sleeve_cash[sleeve_id] += proceeds
            else:
                self.cash += proceeds

            gross_pnl  = (price - pos["avg_cost"]) * qty
            net_pnl    = gross_pnl - commission
            cost_basis = pos["avg_cost"] * pos["quantity"]
            return_pct = net_pnl / cost_basis if cost_basis != 0 else 0.0

            self.trade_log.append({
                "symbol":       symbol,
                "entry_date":   pos["entry_date"],
                "exit_date":    fill_event.timestamp,
                "direction":    "LONG",
                "quantity":     qty,
                "entry_price":  pos["entry_price"],
                "exit_price":   price,
                "gross_pnl":    round(gross_pnl, 4),
                "commission":   round(commission, 4),
                "net_pnl":      round(net_pnl, 4),
                "strategy_id":  pos.get("strategy_id", ""),
                "return_pct":   round(return_pct, 6),
            })

    # ──────────────────────────────────────────────────────────────────────
    # EQUITY SNAPSHOT
    # ──────────────────────────────────────────────────────────────────────

    def record_equity(self, timestamp: datetime, data_handler) -> None:
        """Record the current portfolio value at this timestamp."""
        self.equity_curve.append((timestamp, self.get_total_equity(data_handler)))

    def _mark_to_market_value(self, data_handler) -> float:
        """Total current market value of all open positions."""
        if self._ensemble_enabled and self._sleeve_ids:
            return sum(
                self._sleeve_mtm(h, data_handler)
                for h in self._sleeve_holdings.values()
            )
        return self._sleeve_mtm(self.holdings, data_handler)

    def _total_invested(self, data_handler) -> float:
        return self._mark_to_market_value(data_handler)

    def get_total_equity(self, data_handler) -> float:
        """Return cash + mark-to-market of all open positions."""
        if self._ensemble_enabled and self._sleeve_ids:
            cash_total = sum(self._sleeve_cash.values())
            return cash_total + self._mark_to_market_value(data_handler)
        return self.cash + self._mark_to_market_value(data_handler)

    # ──────────────────────────────────────────────────────────────────────
    # REPORTING ACCESSORS
    # ──────────────────────────────────────────────────────────────────────

    def calculate_equity_curve(self) -> pd.Series:
        if not self.equity_curve:
            return pd.Series(dtype=float, name="portfolio_value")
        dates, values = zip(*self.equity_curve)
        return pd.Series(values, index=pd.DatetimeIndex(dates), name="portfolio_value")

    def get_trade_log(self) -> List[dict]:
        return list(self.trade_log)

    def get_open_positions(self) -> Dict[str, dict]:
        if self._ensemble_enabled and self._sleeve_ids:
            merged = {}
            for sleeve_id, holdings in self._sleeve_holdings.items():
                for sym, pos in holdings.items():
                    merged[f"{sleeve_id}:{sym}"] = dict(pos)
            return merged
        return {sym: dict(pos) for sym, pos in self.holdings.items()}

    def get_portfolio_summary(self, data_handler) -> dict:
        invested = self._mark_to_market_value(data_handler)
        if self._ensemble_enabled and self._sleeve_ids:
            cash = sum(self._sleeve_cash.values())
        else:
            cash = self.cash
        return {
            "cash":           cash,
            "invested_value": invested,
            "total_equity":   cash + invested,
            "num_positions":  len(self.get_open_positions()),
            "open_positions": self.get_open_positions(),
            "total_trades":   len(self.trade_log),
        }

    def get_final_equity(self) -> float:
        if self.equity_curve:
            return self.equity_curve[-1][1]
        if self._ensemble_enabled and self._sleeve_ids:
            return sum(self._sleeve_cash.values())
        return self.cash
