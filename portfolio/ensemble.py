# portfolio/ensemble.py
# ============================================================
# MULTI-STRATEGY ENSEMBLE PORTFOLIO CONSTRUCTION
# ============================================================

import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import BacktestConfig
from engine.events import SignalEvent

logger = logging.getLogger(__name__)


class EnsembleAllocator:
    """
    Scales position sizes per strategy sleeve using risk-aware capital weights.

    Repurposes SignalEvent.strength as the sleeve capital weight when ensemble
    mode is active (PortfolioManager must multiply sizing by this value).
    """

    def __init__(self, config: BacktestConfig, strategy_ids: List[str]) -> None:
        if not strategy_ids:
            raise ValueError("EnsembleAllocator requires at least one strategy_id.")

        self._config = config
        self.strategy_ids = list(strategy_ids)
        n = len(strategy_ids)

        self.sleeve_weights: Dict[str, float] = {
            sid: 1.0 / n for sid in strategy_ids
        }
        self.sleeve_returns: Dict[str, List[float]] = {sid: [] for sid in strategy_ids}
        self.sleeve_equity: Dict[str, float] = {
            sid: config.portfolio.initial_capital / n for sid in strategy_ids
        }
        self.last_rebalance_date: Optional[datetime] = None
        self.method = config.advanced.ensemble.weighting_method
        self.vol_lookback = config.advanced.ensemble.vol_lookback_days
        self.rebalance_freq = config.advanced.ensemble.rebalance_freq
        self.max_sleeve_weight = config.advanced.ensemble.max_sleeve_weight
        self._weight_log: List[tuple] = []

    def filter_signal(self, signal: SignalEvent) -> Optional[SignalEvent]:
        """Attach the current sizing multiplier for this signal's strategy sleeve."""
        if signal.strategy_id not in self.sleeve_weights:
            logger.warning(
                "Unregistered strategy_id %s in ensemble — using equal weight fallback.",
                signal.strategy_id,
            )
            weight = 1.0 / len(self.strategy_ids)
        else:
            weight = self.sleeve_weights[signal.strategy_id]

        signal.strength = weight
        return signal

    def update_sleeve_returns(self, date: datetime, sleeve_pnls: Dict[str, float]) -> None:
        """Record each sleeve's daily P&L for trailing volatility estimation."""
        max_len = self.vol_lookback * 3
        for sid, pnl in sleeve_pnls.items():
            if sid not in self.sleeve_equity:
                continue
            prior = self.sleeve_equity[sid]
            self.sleeve_equity[sid] += pnl
            daily_return = pnl / max(prior, 1.0)
            self.sleeve_returns[sid].append(daily_return)
            if len(self.sleeve_returns[sid]) > max_len:
                self.sleeve_returns[sid] = self.sleeve_returns[sid][-max_len:]

    def maybe_rebalance(self, date: datetime) -> None:
        """Recompute sleeve weights on schedule."""
        if not self._is_rebalance_date(date):
            return

        if self.method == "equal_weight":
            new_weights = {sid: 1.0 / len(self.strategy_ids) for sid in self.strategy_ids}
        elif self.method == "equal_vol":
            new_weights = self._compute_equal_vol_weights()
        elif self.method == "risk_parity":
            new_weights = self._compute_risk_parity_weights()
        else:
            logger.warning("Unknown weighting method '%s' — using equal_weight.", self.method)
            new_weights = {sid: 1.0 / len(self.strategy_ids) for sid in self.strategy_ids}

        new_weights = self._apply_max_weight_cap(new_weights)
        self.sleeve_weights = new_weights
        self.last_rebalance_date = date
        self._weight_log.append((date, dict(new_weights)))

        logger.info(
            "Ensemble rebalanced: %s",
            ", ".join(f"{k}={v:.1%}" for k, v in new_weights.items()),
        )

    def _apply_max_weight_cap(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Clip any sleeve above max_sleeve_weight and renormalise."""
        capped = dict(weights)
        while True:
            excess = 0.0
            uncapped_total = 0.0
            for sid, w in capped.items():
                if w > self.max_sleeve_weight:
                    excess += w - self.max_sleeve_weight
                    capped[sid] = self.max_sleeve_weight
                else:
                    uncapped_total += capped[sid]

            if excess <= 1e-12:
                break

            recipients = [sid for sid, w in capped.items() if w < self.max_sleeve_weight]
            if not recipients:
                break
            share = excess / len(recipients)
            for sid in recipients:
                capped[sid] = min(self.max_sleeve_weight, capped[sid] + share)

        total = sum(capped.values())
        if total <= 0:
            return {sid: 1.0 / len(capped) for sid in capped}
        return {sid: w / total for sid, w in capped.items()}

    def _compute_equal_vol_weights(self) -> Dict[str, float]:
        """Inverse-volatility weights across sleeves."""
        vols: Dict[str, float] = {}
        for sid in self.strategy_ids:
            returns = self.sleeve_returns[sid][-self.vol_lookback :]
            if len(returns) < 10:
                vols[sid] = 1.0
            else:
                vols[sid] = float(np.std(returns)) or 1e-8

        inv_vols = {sid: 1.0 / v for sid, v in vols.items()}
        total = sum(inv_vols.values())
        return {sid: iv / total for sid, iv in inv_vols.items()}

    def _compute_risk_parity_weights(self) -> Dict[str, float]:
        """Iterative risk-parity weights using sleeve return covariance."""
        series_list = []
        for sid in self.strategy_ids:
            s = self.sleeve_returns[sid][-self.vol_lookback :]
            if len(s) < 10:
                return self._compute_equal_vol_weights()
            series_list.append(s)

        min_len = min(len(s) for s in series_list)
        if min_len < 10:
            return self._compute_equal_vol_weights()

        aligned = [s[-min_len:] for s in series_list]
        r = pd.DataFrame({sid: aligned[i] for i, sid in enumerate(self.strategy_ids)})
        if r.shape[1] < 2:
            return self._compute_equal_vol_weights()

        cov = r.cov().values
        n = len(self.strategy_ids)
        w = np.ones(n) / n

        for _ in range(100):
            portfolio_vol = float(np.sqrt(w @ cov @ w))
            if portfolio_vol <= 1e-12:
                break
            marginal = cov @ w / portfolio_vol
            risk_contrib = w * marginal
            target = portfolio_vol / n
            w_new = w * np.sqrt(np.maximum(target / np.maximum(risk_contrib, 1e-12), 1e-12))
            w_new = w_new / w_new.sum()
            if np.max(np.abs(w_new - w)) < 1e-6:
                w = w_new
                break
            w = w_new

        return {sid: float(w[i]) for i, sid in enumerate(self.strategy_ids)}

    def _is_rebalance_date(self, current_date: datetime) -> bool:
        """Return True if sleeve weights should be recomputed today."""
        if self.last_rebalance_date is None:
            return True

        if self.rebalance_freq == "monthly":
            return (
                current_date.month != self.last_rebalance_date.month
                or current_date.year != self.last_rebalance_date.year
            )

        if self.rebalance_freq == "weekly":
            return (
                current_date.isocalendar()[1]
                != self.last_rebalance_date.isocalendar()[1]
            )

        return False

    def get_weight_history(self) -> pd.DataFrame:
        """Return sleeve weights over time for reporting."""
        if not self._weight_log:
            return pd.DataFrame(columns=self.strategy_ids)

        rows = []
        index = []
        for date, weights in self._weight_log:
            index.append(date)
            rows.append(weights)

        return pd.DataFrame(rows, index=pd.DatetimeIndex(index)).sort_index()
