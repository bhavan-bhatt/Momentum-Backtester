# engine/portfolio.py
# ============================================================
# POSITION SIZING + CASH MANAGEMENT + PORTFOLIO STATE
# Receives SignalEvents, computes position size, emits OrderEvents.
# Receives FillEvents, updates holdings and cash.
# ============================================================

import logging
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from engine.events import (
    SignalEvent,
    SignalDirection,
    OrderEvent,
    OrderDirection,
    FillEvent,
)
from engine.data_handler import DataHandler
from config import BacktestConfig

logger = logging.getLogger(__name__)


class Position:
    """Represents a single open position."""

    __slots__ = ("symbol", "quantity", "avg_cost", "entry_date", "strategy_id")

    def __init__(
        self,
        symbol: str,
        quantity: int,
        avg_cost: float,
        entry_date: datetime,
        strategy_id: str = "",
    ) -> None:
        self.symbol      = symbol
        self.quantity    = quantity
        self.avg_cost    = avg_cost
        self.entry_date  = entry_date
        self.strategy_id = strategy_id

    def market_value(self, current_price: float) -> float:
        return self.quantity * current_price

    def unrealised_pnl(self, current_price: float) -> float:
        return self.quantity * (current_price - self.avg_cost)

    def __repr__(self) -> str:
        return (
            f"Position({self.symbol}, qty={self.quantity}, "
            f"avg_cost={self.avg_cost:.2f})"
        )


class PortfolioManager:
    """
    Manages portfolio state: cash, open positions, equity curve, and trade log.

    Signal → size computation → OrderEvent
    Fill   → update holdings, cash, equity curve

    Position Sizing Methods
    -----------------------
    "atr"          — Volatility-adjusted sizing:
                     qty = (equity × risk_pct) / (ATR × atr_stop_multiplier)
                     Capped at max_position_pct × equity / price.
    "equal_weight" — Allocate equity/max_open_positions to each signal.
    "fixed"        — Fixed lot from config.portfolio.fixed_quantity.
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

        pcfg = config.portfolio
        self._cash         = pcfg.initial_capital
        self._initial_cap  = pcfg.initial_capital
        self._positions: Dict[str, Position] = {}
        self._equity_curve: List[Dict] = []
        self._trade_log:    List[Dict] = []

    # ──────────────────────────────────────────────────────────────────────
    # PROPERTIES
    # ──────────────────────────────────────────────────────────────────────

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def open_positions(self) -> Dict[str, Position]:
        return self._positions

    @property
    def equity(self) -> float:
        """Total portfolio value: cash + mark-to-market positions."""
        total = self._cash
        for sym, pos in self._positions.items():
            price = self._data.get_current_price(sym)
            if price is not None:
                total += pos.market_value(price)
        return total

    @property
    def num_open_positions(self) -> int:
        return len(self._positions)

    # ──────────────────────────────────────────────────────────────────────
    # SIGNAL PROCESSING
    # ──────────────────────────────────────────────────────────────────────

    def on_signal(self, signal: SignalEvent) -> None:
        """
        React to a SignalEvent:
          - LONG / SHORT → compute position size and emit OrderEvent (if allowed).
          - EXIT_LONG / EXIT_SHORT → emit OrderEvent to close position.
        """
        symbol    = signal.symbol
        direction = signal.direction
        pcfg      = self._config.portfolio
        scfg      = self._config.strategy

        if direction in (SignalDirection.LONG, SignalDirection.SHORT):
            if direction == SignalDirection.SHORT and not scfg.allow_short:
                logger.debug("SHORT signal ignored for %s — allow_short=False.", symbol)
                return

            if symbol in self._positions:
                logger.debug("Signal ignored for %s — position already open.", symbol)
                return

            if self.num_open_positions >= pcfg.max_open_positions:
                logger.debug(
                    "Signal ignored for %s — max open positions (%d) reached.",
                    symbol, pcfg.max_open_positions,
                )
                return

            price = self._data.get_current_price(symbol)
            if price is None or price <= 0:
                logger.warning("Cannot size position for %s — invalid price.", symbol)
                return

            qty = self._compute_quantity(symbol, price, signal)
            if qty <= 0:
                logger.debug("Position size 0 for %s — skipping order.", symbol)
                return

            order_dir = OrderDirection.BUY if direction == SignalDirection.LONG else OrderDirection.SELL
            self._emit_order(signal, order_dir, qty)

        elif direction == SignalDirection.EXIT_LONG:
            if symbol not in self._positions:
                return
            qty = self._positions[symbol].quantity
            self._emit_order(signal, OrderDirection.SELL, qty)

        elif direction == SignalDirection.EXIT_SHORT:
            if symbol not in self._positions:
                return
            qty = self._positions[symbol].quantity
            self._emit_order(signal, OrderDirection.BUY, qty)

    def on_fill(self, fill: FillEvent) -> None:
        """
        React to a FillEvent: update holdings and cash.
        """
        symbol     = fill.symbol
        qty        = fill.quantity
        price      = fill.fill_price
        commission = fill.commission
        direction  = fill.direction

        if direction == OrderDirection.BUY:
            cost = qty * price + commission
            if cost > self._cash:
                logger.warning(
                    "Insufficient cash for fill: need %.2f, have %.2f. "
                    "Order may have been partially filled.",
                    cost, self._cash,
                )
            self._cash -= cost

            if symbol in self._positions:
                pos = self._positions[symbol]
                total_qty  = pos.quantity + qty
                avg_cost   = (pos.quantity * pos.avg_cost + qty * price) / total_qty
                pos.quantity = total_qty
                pos.avg_cost = avg_cost
            else:
                strategy_id = fill.order_ref.signal_ref.strategy_id if (
                    fill.order_ref and fill.order_ref.signal_ref
                ) else ""
                self._positions[symbol] = Position(
                    symbol=symbol,
                    quantity=qty,
                    avg_cost=price,
                    entry_date=fill.timestamp,
                    strategy_id=strategy_id,
                )

            self._log_trade(fill, "ENTRY")

        elif direction == OrderDirection.SELL:
            pos = self._positions.get(symbol)
            proceeds = qty * price - commission
            self._cash += proceeds

            realised_pnl = 0.0
            if pos is not None:
                realised_pnl = qty * (price - pos.avg_cost) - commission
                remaining = pos.quantity - qty
                if remaining <= 0:
                    del self._positions[symbol]
                else:
                    pos.quantity = remaining

            self._log_trade(fill, "EXIT", realised_pnl=realised_pnl)

    def update_equity_curve(self, timestamp: datetime) -> None:
        """Record a snapshot of current portfolio value."""
        eq = self.equity
        self._equity_curve.append({
            "date":            timestamp,
            "equity":          eq,
            "cash":            self._cash,
            "open_positions":  self.num_open_positions,
        })

    # ──────────────────────────────────────────────────────────────────────
    # POSITION SIZING
    # ──────────────────────────────────────────────────────────────────────

    def _compute_quantity(
        self, symbol: str, price: float, signal: SignalEvent
    ) -> int:
        """Compute integer share quantity using the configured sizing method."""
        pcfg   = self._config.portfolio
        scfg   = self._config.strategy
        method = pcfg.sizing_method
        equity = self.equity

        # Exposure cap: don't allocate more than (max_total_exposure_pct - current_exposure)
        current_exposure = equity - self._cash
        max_new_exposure = equity * pcfg.max_total_exposure_pct - current_exposure
        if max_new_exposure <= 0:
            return 0

        if method == "fixed":
            qty = pcfg.fixed_quantity
        elif method == "equal_weight":
            allocation = equity / max(1, pcfg.max_open_positions)
            qty        = int(allocation / price)
        elif method == "atr":
            qty = self._atr_size(symbol, price, equity, scfg, pcfg)
        else:
            logger.warning("Unknown sizing_method '%s' — using equal_weight.", method)
            allocation = equity / max(1, pcfg.max_open_positions)
            qty        = int(allocation / price)

        # Position cap: no single position > max_position_pct of equity
        max_qty_by_cap = int(equity * pcfg.max_position_pct / price)
        # Exposure cap: don't exceed remaining exposure budget
        max_qty_by_exposure = int(max_new_exposure / price)

        qty = min(qty, max_qty_by_cap, max_qty_by_exposure)
        return max(0, qty)

    def _atr_size(
        self, symbol: str, price: float, equity: float, scfg, pcfg
    ) -> int:
        """
        ATR-based position sizing:
          risk_amount = equity × risk_per_trade_pct
          stop_distance = ATR × atr_stop_multiplier
          qty = risk_amount / stop_distance
        """
        bars = self._data.get_latest_bars(symbol, scfg.atr_period + 1)
        if bars is None or len(bars) < scfg.atr_period + 1:
            # Fall back to equal-weight if not enough history
            allocation = equity / max(1, pcfg.max_open_positions)
            return int(allocation / price)

        has_hl = self._data.has_high_low.get(symbol, False)
        if has_hl:
            from engine.strategy import atr as compute_atr
            atr_series = compute_atr(bars["high"], bars["low"], bars["close"], scfg.atr_period)
            atr_val    = atr_series.iloc[-1]
        else:
            # Price-percentage proxy when H/L unavailable
            atr_val = price * 0.02  # 2% proxy

        if np.isnan(atr_val) or atr_val <= 0:
            allocation = equity / max(1, pcfg.max_open_positions)
            return int(allocation / price)

        risk_amount    = equity * pcfg.risk_per_trade_pct
        stop_distance  = atr_val * scfg.atr_stop_multiplier
        qty            = int(risk_amount / stop_distance)
        return qty

    # ──────────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────────

    def _emit_order(
        self, signal: SignalEvent, direction: OrderDirection, quantity: int
    ) -> None:
        order = OrderEvent(
            timestamp=signal.timestamp,
            symbol=signal.symbol,
            direction=direction,
            quantity=quantity,
            signal_ref=signal,
        )
        self._event_queue.append(order)

    def _log_trade(
        self, fill: FillEvent, trade_type: str, realised_pnl: float = 0.0
    ) -> None:
        strategy_id = ""
        if fill.order_ref and fill.order_ref.signal_ref:
            strategy_id = fill.order_ref.signal_ref.strategy_id

        self._trade_log.append({
            "date":          fill.timestamp,
            "symbol":        fill.symbol,
            "type":          trade_type,
            "direction":     fill.direction.value,
            "quantity":      fill.quantity,
            "fill_price":    fill.fill_price,
            "commission":    fill.commission,
            "slippage":      fill.slippage,
            "realised_pnl":  realised_pnl,
            "strategy_id":   strategy_id,
            "equity_after":  self.equity,
        })

    # ──────────────────────────────────────────────────────────────────────
    # REPORTING ACCESSORS
    # ──────────────────────────────────────────────────────────────────────

    def get_equity_curve(self) -> pd.DataFrame:
        """Return the equity curve as a DataFrame indexed by date."""
        if not self._equity_curve:
            return pd.DataFrame()
        df = pd.DataFrame(self._equity_curve).set_index("date")
        df.index = pd.DatetimeIndex(df.index)
        return df

    def get_trade_log(self) -> pd.DataFrame:
        """Return the full trade log as a DataFrame."""
        if not self._trade_log:
            return pd.DataFrame()
        return pd.DataFrame(self._trade_log)

    def get_final_equity(self) -> float:
        """Return the final portfolio equity."""
        if self._equity_curve:
            return self._equity_curve[-1]["equity"]
        return self.equity
