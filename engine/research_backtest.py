# engine/research_backtest.py
# ============================================================
# MULTI-STRATEGY RESEARCH BACKTEST ENGINE (Phase 2)
# ============================================================

import logging
from typing import Dict, List, Optional

import pandas as pd

from engine.audit_log import AuditLog
from engine.backtest import BacktestEngine
from engine.events import EventType, FillEvent
from engine.regime_filter import RegimeFilter
from engine.strategy import BaseStrategy
from portfolio.ensemble import EnsembleAllocator

logger = logging.getLogger(__name__)


class ResearchBacktestEngine(BacktestEngine):
    """
    Extends BacktestEngine for multi-strategy ensemble research runs.

    - Updates regime filter once per bar
    - Routes signals through ensemble allocator
    - Tracks per-sleeve P&L for weight rebalancing
    """

    def __init__(
        self,
        config,
        data_handler,
        strategies: List[BaseStrategy],
        portfolio,
        execution,
        regime_filter: RegimeFilter,
        ensemble_allocator: EnsembleAllocator,
        audit_log: Optional[AuditLog] = None,
    ) -> None:
        primary = strategies[0] if strategies else None
        super().__init__(
            config,
            data_handler,
            primary,
            portfolio,
            execution,
            regime_filter=regime_filter,
            ensemble_allocator=ensemble_allocator,
        )
        self.strategies = strategies
        self.audit_log = audit_log
        # Wire every sleeve to the engine queue (super() only connects primary).
        for s in strategies:
            s._event_queue = self.event_queue
        self._prev_equity: Optional[float] = None
        self._sleeve_prev_equity: Dict[str, float] = {
            sid: config.portfolio.initial_capital / max(len(strategies), 1)
            for sid in ensemble_allocator.strategy_ids
        }

    def run(self) -> Dict:
        """Run backtest with all strategy sleeves."""
        if len(self.strategies) > 1:
            self.strategy = _MultiStrategyProxy(self.strategies)
        return super().run()

    def _process_event(self, event) -> None:
        etype = event.event_type

        if etype == EventType.MARKET:
            for strategy in self.strategies:
                strategy.calculate_signals(event, self.data_handler)
            return

        if etype == EventType.FILL:
            super()._process_event(event)
            self._update_sleeve_pnl(event)
            return

        super()._process_event(event)

    def _update_sleeve_pnl(self, fill: FillEvent) -> None:
        """Attribute fill P&L to strategy sleeve for ensemble weighting."""
        if self.ensemble_allocator is None:
            return

        current_dt = self.data_handler.get_current_datetime()
        if current_dt is None:
            return

        sleeve_pnls: Dict[str, float] = {}
        for sid in self.ensemble_allocator.strategy_ids:
            sleeve_pnls[sid] = 0.0

        strategy_id = ""
        if fill.order_ref and fill.order_ref.signal_ref:
            strategy_id = fill.order_ref.signal_ref.strategy_id

        if strategy_id:
            matched = None
            for sid in self.ensemble_allocator.strategy_ids:
                if strategy_id.startswith(sid.split("_")[0]) or sid in strategy_id:
                    matched = sid
                    break
            if matched is None:
                for sid in self.ensemble_allocator.strategy_ids:
                    if sid.lower() in strategy_id.lower():
                        matched = sid
                        break
            if matched:
                sleeve_pnls[matched] = 0.0

        total_eq = self.portfolio.get_total_equity(self.data_handler)
        if self._prev_equity is not None:
            daily_pnl = total_eq - self._prev_equity
            n = max(len(self.ensemble_allocator.strategy_ids), 1)
            per_sleeve = daily_pnl / n
            for sid in sleeve_pnls:
                sleeve_pnls[sid] = per_sleeve

        self.ensemble_allocator.update_sleeve_returns(current_dt, sleeve_pnls)
        self._prev_equity = total_eq


class _MultiStrategyProxy:
    """Placeholder strategy id for logging when running ensemble."""

    def __init__(self, strategies: List[BaseStrategy]) -> None:
        self.strategy_id = "Ensemble"
        self._strategies = strategies
