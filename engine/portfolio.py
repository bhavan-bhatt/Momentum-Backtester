# engine/portfolio.py
# ============================================================
# PORTFOLIO MANAGER — POSITION SIZING, CASH, AND HOLDINGS
# Receives SignalEvents, computes order size, emits OrderEvents.
# Receives FillEvents, updates cash and holdings.
# ============================================================

import logging
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional

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

    Internal State
    --------------
    cash            : float                   — Uninvested cash in INR.
    holdings        : Dict[str, dict]          — Open positions.
                      Each value: {quantity, avg_cost, entry_date, entry_price, strategy_id}
    equity_curve    : list[(datetime, float)]  — (date, total_portfolio_value) snapshots.
    trade_log       : list[dict]               — One dict per CLOSED trade.
    _pending_orders : Dict[str, OrderEvent]    — Orders awaiting fills (dedup guard).
    """

    def __init__(self, config: BacktestConfig, event_queue: deque) -> None:
        self._config      = config
        self._event_queue = event_queue  # may be replaced by BacktestEngine

        self.cash             = config.portfolio.initial_capital
        self.holdings:         Dict[str, dict]         = {}
        self.equity_curve:     List[tuple]             = []
        self.trade_log:        List[dict]              = []
        self._pending_orders:  Dict[str, "OrderEvent"] = {}

    # ──────────────────────────────────────────────────────────────────────
    # SIGNAL → ORDER
    # ──────────────────────────────────────────────────────────────────────

    def process_signal(self, signal_event: SignalEvent, data_handler) -> None:
        """
        Convert a SignalEvent into an OrderEvent after checking risk limits.

        Entry signals (LONG/SHORT):
          Skipped if already invested, order pending, or max positions reached.
          Size is computed via _compute_position_size().

        Exit signals (EXIT_LONG/EXIT_SHORT):
          Immediately emit a SELL/BUY for the full current quantity.
        """
        symbol    = signal_event.symbol
        direction = signal_event.direction
        pcfg      = self._config.portfolio
        scfg      = self._config.strategy

        if direction in (SignalDirection.LONG, SignalDirection.SHORT):
            if direction == SignalDirection.SHORT and not scfg.allow_short:
                logger.debug("SHORT ignored for %s — allow_short=False.", symbol)
                return

            price = data_handler.get_current_price(symbol)
            if price is None:
                logger.warning("No price for %s — signal skipped.", symbol)
                return

            if symbol in self.holdings:
                return  # already invested

            if symbol in self._pending_orders:
                return  # order already pending fill

            open_slots = len(self.holdings) + len(self._pending_orders)
            if open_slots >= pcfg.max_open_positions:
                logger.debug(
                    "Max positions (%d) reached — skipping %s.",
                    pcfg.max_open_positions, symbol,
                )
                return

            qty = self._compute_position_size(signal_event, data_handler, price)
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
            self._pending_orders[symbol] = order

        elif direction == SignalDirection.EXIT_LONG:
            if symbol not in self.holdings:
                return
            qty   = self.holdings[symbol]["quantity"]
            order = OrderEvent(
                timestamp=signal_event.timestamp,
                symbol=symbol,
                direction=OrderDirection.SELL,
                quantity=qty,
                signal_ref=signal_event,
            )
            self._event_queue.append(order)

        elif direction == SignalDirection.EXIT_SHORT:
            if symbol not in self.holdings:
                return
            qty   = self.holdings[symbol]["quantity"]
            order = OrderEvent(
                timestamp=signal_event.timestamp,
                symbol=symbol,
                direction=OrderDirection.BUY,
                quantity=qty,
                signal_ref=signal_event,
            )
            self._event_queue.append(order)

    # ──────────────────────────────────────────────────────────────────────
    # POSITION SIZING
    # ──────────────────────────────────────────────────────────────────────

    def _compute_position_size(
        self,
        signal: SignalEvent,
        data_handler,
        current_price: float,
    ) -> int:
        """
        Determine share quantity based on the configured sizing method.

        Returns 0 if any constraint is violated (no trade).
        """
        pcfg   = self._config.portfolio
        scfg   = self._config.strategy
        method = pcfg.sizing_method

        if method == "fixed":
            qty = pcfg.fixed_quantity
            if qty * current_price > self.cash:
                return 0
            return qty

        total = self.cash + self._mark_to_market_value(data_handler)

        if method == "equal_weight":
            per_position = total * pcfg.max_position_pct
            qty          = int(per_position / current_price)

        elif method == "atr":
            qty = self._atr_quantity(signal.symbol, data_handler, total, current_price)
        else:
            logger.warning("Unknown sizing_method '%s' — using equal_weight.", method)
            per_position = total * pcfg.max_position_pct
            qty          = int(per_position / current_price)

        # ── Constraints ───────────────────────────────────────────────────
        # Max position cap
        max_by_cap = int((total * pcfg.max_position_pct) / current_price)
        qty        = min(qty, max_by_cap)

        # Total exposure cap
        cost_available = (total * pcfg.max_total_exposure_pct) - self._total_invested(data_handler)
        max_by_exposure = int(cost_available / current_price)
        qty             = min(qty, max_by_exposure)

        if qty <= 0:
            return 0

        # Cash sufficiency
        if qty * current_price > self.cash:
            return 0

        return qty

    def _atr_quantity(
        self, symbol: str, data_handler, total: float, price: float
    ) -> int:
        """
        ATR-based sizing:
          risk_amount = total × risk_per_trade_pct
          stop        = ATR × atr_stop_multiplier
          qty         = risk_amount / stop
        Falls back to equal-weight if ATR is unavailable.
        """
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
            # Equal-weight fallback
            return int((total * pcfg.max_position_pct) / price)

        risk_amount   = total * pcfg.risk_per_trade_pct
        stop_distance = atr_val * scfg.atr_stop_multiplier
        return int(risk_amount / stop_distance)

    # ──────────────────────────────────────────────────────────────────────
    # FILL → UPDATE HOLDINGS
    # ──────────────────────────────────────────────────────────────────────

    def update_portfolio_on_fill(self, fill_event: FillEvent) -> None:
        """
        Update cash, holdings, and trade log after a fill is received.
        """
        symbol     = fill_event.symbol
        qty        = fill_event.quantity
        price      = fill_event.fill_price
        commission = fill_event.commission
        direction  = fill_event.direction

        if direction == OrderDirection.BUY:
            trade_cost = qty * price + commission
            self.cash -= trade_cost

            # Remove from pending
            self._pending_orders.pop(symbol, None)

            if symbol not in self.holdings:
                strategy_id = ""
                if fill_event.order_ref and fill_event.order_ref.signal_ref:
                    strategy_id = fill_event.order_ref.signal_ref.strategy_id
                self.holdings[symbol] = {
                    "quantity":    qty,
                    "avg_cost":    price,
                    "entry_date":  fill_event.timestamp,
                    "entry_price": price,
                    "strategy_id": strategy_id,
                }
            else:
                # Averaging up (rare but handled gracefully)
                pos       = self.holdings[symbol]
                total_qty = pos["quantity"] + qty
                avg_cost  = (pos["quantity"] * pos["avg_cost"] + qty * price) / total_qty
                pos["quantity"] = total_qty
                pos["avg_cost"] = avg_cost

        elif direction == OrderDirection.SELL:
            self._pending_orders.pop(symbol, None)

            if symbol not in self.holdings:
                logger.warning("SELL fill for %s but no open position — ignored.", symbol)
                return

            pos   = self.holdings.pop(symbol)
            proceeds = qty * price - commission
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
        """
        Record the current portfolio value at this timestamp.
        Called at the end of each bar by BacktestEngine.
        """
        total = self.cash + self._mark_to_market_value(data_handler)
        self.equity_curve.append((timestamp, total))

    def _mark_to_market_value(self, data_handler) -> float:
        """Total current market value of all open positions."""
        total = 0.0
        for symbol, pos in self.holdings.items():
            price = data_handler.get_current_price(symbol)
            if price is not None:
                total += pos["quantity"] * price
            else:
                total += pos["quantity"] * pos["avg_cost"]
        return total

    def _total_invested(self, data_handler) -> float:
        """Mark-to-market value invested (alias for exposure calculation)."""
        return self._mark_to_market_value(data_handler)

    def get_total_equity(self, data_handler) -> float:
        """Return cash + mark-to-market of all open positions."""
        return self.cash + self._mark_to_market_value(data_handler)

    # ──────────────────────────────────────────────────────────────────────
    # REPORTING ACCESSORS
    # ──────────────────────────────────────────────────────────────────────

    def calculate_equity_curve(self) -> pd.Series:
        """Convert the equity_curve list into a pd.Series indexed by datetime."""
        if not self.equity_curve:
            return pd.Series(dtype=float, name="portfolio_value")
        dates, values = zip(*self.equity_curve)
        return pd.Series(values, index=pd.DatetimeIndex(dates), name="portfolio_value")

    def get_trade_log(self) -> List[dict]:
        """Return the complete trade log as a list of dicts."""
        return list(self.trade_log)

    def get_open_positions(self) -> Dict[str, dict]:
        """Return a copy of all currently open positions."""
        return {sym: dict(pos) for sym, pos in self.holdings.items()}

    def get_portfolio_summary(self, data_handler) -> dict:
        """Return a human-readable snapshot of the current portfolio state."""
        invested  = self._mark_to_market_value(data_handler)
        positions = {}
        for sym, pos in self.holdings.items():
            price = data_handler.get_current_price(sym) or pos["avg_cost"]
            positions[sym] = {
                **pos,
                "current_price": price,
                "market_value":  pos["quantity"] * price,
                "unrealised_pnl": (price - pos["avg_cost"]) * pos["quantity"],
            }
        return {
            "cash":           self.cash,
            "invested_value": invested,
            "total_equity":   self.cash + invested,
            "num_positions":  len(self.holdings),
            "open_positions": positions,
            "total_trades":   len(self.trade_log),
        }

    def get_final_equity(self) -> float:
        """Return the last snapshotted portfolio value."""
        if self.equity_curve:
            return self.equity_curve[-1][1]
        return self.cash
