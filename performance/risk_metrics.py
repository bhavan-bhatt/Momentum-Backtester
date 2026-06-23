# performance/risk_metrics.py
# ============================================================
# EXTENDED RISK METRICS — TAIL RATIO, RECOVERY, TURNOVER, CAPACITY
# ============================================================

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import BacktestConfig


class ExtendedRiskMetrics:
    """Risk-shape metrics beyond Sharpe and max drawdown."""

    def __init__(self, config: BacktestConfig) -> None:
        self.trading_days = config.report.trading_days_per_year
        self._config = config

    def calculate_tail_ratio(self, returns: pd.Series, percentile: float = 0.05) -> float:
        """Ratio of right-tail to left-tail magnitude at given percentile."""
        clean = returns.dropna()
        if clean.empty:
            return 0.0
        right_tail = abs(float(np.percentile(clean, 100 * (1 - percentile))))
        left_tail = abs(float(np.percentile(clean, 100 * percentile)))
        if left_tail == 0:
            return float("inf")
        return round(right_tail / left_tail, 4)

    def calculate_time_to_recovery(
        self, equity_curve: pd.Series
    ) -> Tuple[Optional[int], pd.DataFrame]:
        """Longest drawdown recovery time and full episode table."""
        if equity_curve is None or len(equity_curve) < 2:
            return None, pd.DataFrame(
                columns=[
                    "start_date",
                    "trough_date",
                    "end_date",
                    "drawdown_pct",
                    "recovery_days",
                    "still_underwater",
                ]
            )

        running_max = equity_curve.cummax()
        in_dd = equity_curve < running_max

        episodes: List[dict] = []
        i = 0
        idx = equity_curve.index

        while i < len(equity_curve):
            if not in_dd.iloc[i]:
                i += 1
                continue

            start_i = i
            peak_value = running_max.iloc[i]
            trough_i = i
            trough_value = equity_curve.iloc[i]

            while i < len(equity_curve) and in_dd.iloc[i]:
                if equity_curve.iloc[i] < trough_value:
                    trough_value = equity_curve.iloc[i]
                    trough_i = i
                i += 1

            still_underwater = i >= len(equity_curve)
            end_date = None if still_underwater else idx[i - 1]
            recovery_days = np.nan
            if not still_underwater:
                recovery_days = i - 1 - start_i

            dd_pct = (trough_value - peak_value) / peak_value if peak_value else 0.0
            episodes.append({
                "start_date": idx[start_i],
                "trough_date": idx[trough_i],
                "end_date": end_date,
                "drawdown_pct": round(float(dd_pct), 6),
                "recovery_days": recovery_days,
                "still_underwater": still_underwater,
            })

        df = pd.DataFrame(episodes)
        completed = df[~df["still_underwater"]]["recovery_days"].dropna()
        worst = int(completed.max()) if len(completed) else None
        return worst, df

    def calculate_turnover(
        self, trade_log: List[dict], equity_curve: pd.Series
    ) -> Dict[str, float]:
        """Portfolio turnover relative to average equity."""
        if not trade_log or equity_curve is None or len(equity_curve) < 2:
            return {
                "annual_turnover": 0.0,
                "avg_holding_period_days": 0.0,
                "total_round_trips": len(trade_log),
                "total_traded_value": 0.0,
            }

        total_traded_value = sum(
            t["quantity"] * t["entry_price"] + t["quantity"] * t["exit_price"]
            for t in trade_log
        )
        avg_equity = float(equity_curve.mean())
        num_years = max(
            (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25,
            0.01,
        )
        annual_turnover = (total_traded_value / avg_equity) / num_years

        holding_days = []
        for t in trade_log:
            entry = pd.Timestamp(t["entry_date"])
            exit_ = pd.Timestamp(t["exit_date"])
            holding_days.append((exit_ - entry).days)
        avg_holding = float(np.mean(holding_days)) if holding_days else 0.0

        return {
            "annual_turnover": round(annual_turnover, 2),
            "avg_holding_period_days": round(avg_holding, 1),
            "total_round_trips": len(trade_log),
            "total_traded_value": round(total_traded_value, 0),
        }

    def estimate_capacity(
        self,
        trade_log: List[dict],
        capacity_report: Optional[pd.DataFrame],
        avg_daily_volumes: Dict[str, float],
    ) -> Dict:
        """Rough capacity estimate from participation limits and liquidity."""
        n_constrained = len(capacity_report) if capacity_report is not None else 0
        pct_constrained = n_constrained / max(len(trade_log), 1)

        if not avg_daily_volumes:
            return {
                "pct_orders_capacity_constrained": round(pct_constrained, 4),
                "binding_symbol": None,
                "estimated_capacity_inr": 0.0,
                "caveat": (
                    "No volume data available — capacity estimate not computed."
                ),
            }

        binding_symbol = min(avg_daily_volumes, key=avg_daily_volumes.get)
        min_volume = avg_daily_volumes[binding_symbol]
        participation = self._config.advanced.advanced_execution.participation_rate_limit
        max_position_pct = self._config.portfolio.max_position_pct

        symbol_trades = [t for t in trade_log if t["symbol"] == binding_symbol]
        if symbol_trades:
            rep_price = float(np.mean([t["entry_price"] for t in symbol_trades]))
        else:
            rep_price = 1.0

        max_shares = min_volume * participation
        max_position_value = max_shares * rep_price
        implied_capacity = (
            max_position_value / max_position_pct if max_position_pct > 0 else 0.0
        )

        return {
            "pct_orders_capacity_constrained": round(pct_constrained, 4),
            "binding_symbol": binding_symbol,
            "estimated_capacity_inr": round(implied_capacity, 0),
            "caveat": (
                "This is a rough single-symbol estimate, not a full multi-asset "
                "capacity model. Treat as an order-of-magnitude figure, not a "
                "precise ceiling."
            ),
        }

    def calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0.0) -> float:
        """Omega ratio — probability-weighted gains vs losses above threshold."""
        excess = returns.dropna() - threshold
        gains = excess[excess > 0].sum()
        losses = abs(excess[excess < 0].sum())
        if losses == 0:
            return float("inf")
        return round(float(gains / losses), 4)
