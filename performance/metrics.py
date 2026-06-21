# performance/metrics.py
# ============================================================
# ALL PERFORMANCE METRICS — COMPUTED FROM EQUITY CURVE AND TRADE LOG
# ============================================================

import math
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import BacktestConfig

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """
    Computes all standard quantitative performance metrics.

    Usage
    -----
    perf    = PerformanceMetrics(config)
    metrics = perf.generate_summary_report(equity_curve, trade_log, benchmark_curve)

    All methods accept pd.Series for equity curves and List[dict] for trade logs.
    """

    def __init__(self, config: BacktestConfig) -> None:
        self.risk_free_rate = config.report.risk_free_rate
        self.trading_days   = config.report.trading_days_per_year
        self.daily_rf       = (1 + self.risk_free_rate) ** (1 / self.trading_days) - 1

    # ──────────────────────────────────────────────────────────────────────
    # RETURN / VOLATILITY
    # ──────────────────────────────────────────────────────────────────────

    def calculate_returns(self, equity_curve: pd.Series) -> pd.Series:
        """Convert portfolio equity values to daily percentage returns."""
        return equity_curve.pct_change().dropna()

    def calculate_sharpe_ratio(self, equity_curve: pd.Series) -> float:
        """Annualised Sharpe Ratio. >1 is good, >1.5 very good, >2 excellent."""
        returns        = self.calculate_returns(equity_curve)
        excess_returns = returns - self.daily_rf
        std            = excess_returns.std()
        if std == 0 or len(excess_returns) < 2:
            return 0.0
        sharpe = excess_returns.mean() / std * math.sqrt(self.trading_days)
        return round(sharpe, 4)

    def calculate_cagr(self, equity_curve: pd.Series) -> float:
        """Compound Annual Growth Rate as a decimal (0.15 = 15% p.a.)."""
        if len(equity_curve) < 2:
            return 0.0
        start_val = equity_curve.iloc[0]
        end_val   = equity_curve.iloc[-1]
        num_days  = (equity_curve.index[-1] - equity_curve.index[0]).days
        num_years = num_days / 365.25
        if num_years <= 0 or start_val <= 0:
            return 0.0
        cagr = (end_val / start_val) ** (1 / num_years) - 1
        return round(cagr, 4)

    def calculate_sortino_ratio(self, equity_curve: pd.Series) -> float:
        """Annualised Sortino Ratio (penalises only downside volatility)."""
        returns          = self.calculate_returns(equity_curve)
        excess_returns   = returns - self.daily_rf
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return 0.0
        down_std = downside_returns.std()
        if down_std == 0:
            return 0.0
        sortino = excess_returns.mean() / down_std * math.sqrt(self.trading_days)
        return round(sortino, 4)

    def calculate_max_drawdown(
        self, equity_curve: pd.Series
    ) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
        """
        Maximum peak-to-trough drawdown.

        Returns
        -------
        (max_drawdown_pct, peak_date, trough_date)
        max_drawdown_pct is ≤ 0 (e.g. -0.25 = 25% drawdown).
        """
        if len(equity_curve) < 2:
            return 0.0, equity_curve.index[0], equity_curve.index[0]
        running_max  = equity_curve.cummax()
        drawdown     = (equity_curve - running_max) / running_max
        max_dd       = drawdown.min()
        trough_date  = drawdown.idxmin()
        peak_date    = equity_curve.loc[:trough_date].idxmax()
        return round(max_dd, 6), peak_date, trough_date

    def calculate_calmar_ratio(self, cagr: float, max_drawdown: float) -> float:
        """Calmar Ratio = CAGR / |Max Drawdown|. >1 is acceptable, >2 is strong."""
        if max_drawdown == 0:
            return 0.0
        return round(cagr / abs(max_drawdown), 4)

    # ──────────────────────────────────────────────────────────────────────
    # TRADE-LEVEL METRICS
    # ──────────────────────────────────────────────────────────────────────

    def calculate_win_rate(self, trade_log: List[dict]) -> float:
        """Fraction of completed trades that were profitable."""
        if not trade_log:
            return 0.0
        wins = sum(1 for t in trade_log if t["net_pnl"] > 0)
        return round(wins / len(trade_log), 4)

    def calculate_profit_factor(self, trade_log: List[dict]) -> float:
        """
        Total gross profit / total gross loss.
        >1 = net positive, >1.5 = strong, >2 = excellent.
        """
        gross_profit = sum(t["net_pnl"] for t in trade_log if t["net_pnl"] > 0)
        gross_loss   = abs(sum(t["net_pnl"] for t in trade_log if t["net_pnl"] < 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 1.0
        return round(gross_profit / gross_loss, 4)

    def calculate_avg_win_loss_ratio(self, trade_log: List[dict]) -> float:
        """Average winning trade P&L / average losing trade |P&L|."""
        wins   = [t["net_pnl"] for t in trade_log if t["net_pnl"] > 0]
        losses = [abs(t["net_pnl"]) for t in trade_log if t["net_pnl"] < 0]
        if not wins or not losses:
            return 0.0
        return round(float(np.mean(wins)) / float(np.mean(losses)), 4)

    # ──────────────────────────────────────────────────────────────────────
    # BENCHMARK COMPARISON
    # ──────────────────────────────────────────────────────────────────────

    def calculate_alpha_beta(
        self,
        equity_curve: pd.Series,
        benchmark_curve: pd.Series,
    ) -> Tuple[float, float]:
        """
        Jensen's Alpha and Beta vs the benchmark (^NSEI).

        Returns (alpha, beta). Alpha > 0 → outperforming on risk-adjusted basis.
        """
        eq_returns = equity_curve.pct_change().dropna()
        bm_returns = benchmark_curve.pct_change().dropna()

        # Align to common dates
        eq_returns, bm_returns = eq_returns.align(bm_returns, join="inner")
        if len(eq_returns) < 10:
            return 0.0, 1.0

        cov  = np.cov(eq_returns, bm_returns)
        var_bm = cov[1][1]
        beta   = cov[0][1] / var_bm if var_bm != 0 else 0.0

        port_annual = (1 + eq_returns.mean()) ** self.trading_days - 1
        bm_annual   = (1 + bm_returns.mean()) ** self.trading_days - 1
        alpha = port_annual - (
            self.risk_free_rate + beta * (bm_annual - self.risk_free_rate)
        )
        return round(alpha, 4), round(beta, 4)

    # ──────────────────────────────────────────────────────────────────────
    # TIME-SERIES DERIVED DATA
    # ──────────────────────────────────────────────────────────────────────

    def calculate_monthly_returns(self, equity_curve: pd.Series) -> pd.DataFrame:
        """
        Monthly returns pivot table.

        Returns
        -------
        pd.DataFrame — rows = years, columns = month abbreviations (Jan … Dec).
        """
        returns = self.calculate_returns(equity_curve)
        try:
            monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        except ValueError:
            monthly = returns.resample("M").apply(lambda x: (1 + x).prod() - 1)

        df          = monthly.to_frame("return")
        df["year"]  = df.index.year
        df["month"] = df.index.month
        pivot       = df.pivot(index="year", columns="month", values="return")

        month_names = {
            1: "Jan", 2: "Feb",  3: "Mar", 4: "Apr",
            5: "May", 6: "Jun",  7: "Jul", 8: "Aug",
            9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
        }
        pivot.columns = [month_names.get(m, str(m)) for m in pivot.columns]
        return pivot

    def calculate_drawdown_series(self, equity_curve: pd.Series) -> pd.Series:
        """Drawdown percentage at each point in time. All values ≤ 0."""
        running_max = equity_curve.cummax()
        return (equity_curve - running_max) / running_max

    def calculate_rolling_sharpe(
        self, equity_curve: pd.Series, window: int = 252
    ) -> pd.Series:
        """Rolling Sharpe ratio over a configurable window (default 1 year = 252 days)."""
        returns      = self.calculate_returns(equity_curve)
        excess       = returns - self.daily_rf
        rolling_mean = excess.rolling(window=window).mean()
        rolling_std  = excess.rolling(window=window).std()
        rolling_sharpe = (rolling_mean / rolling_std) * math.sqrt(self.trading_days)
        return rolling_sharpe

    # ──────────────────────────────────────────────────────────────────────
    # MASTER REPORT
    # ──────────────────────────────────────────────────────────────────────

    def generate_summary_report(
        self,
        equity_curve: pd.Series,
        trade_log: List[dict],
        benchmark_curve: Optional[pd.Series] = None,
    ) -> Dict:
        """
        Compute every metric and return a single flat dictionary.

        Parameters
        ----------
        equity_curve    : pd.Series (date → portfolio value)
        trade_log       : List[dict] — completed round-trip trades
        benchmark_curve : pd.Series or None — benchmark equity values

        Returns
        -------
        dict with all standard performance keys.
        """
        if equity_curve is None or len(equity_curve) < 2:
            logger.warning("Equity curve too short — returning empty metrics.")
            return _empty_metrics()

        # ── Core metrics ──────────────────────────────────────────────────
        cagr            = self.calculate_cagr(equity_curve)
        sharpe          = self.calculate_sharpe_ratio(equity_curve)
        sortino         = self.calculate_sortino_ratio(equity_curve)
        max_dd, pk, tr  = self.calculate_max_drawdown(equity_curve)
        calmar          = self.calculate_calmar_ratio(cagr, max_dd)
        win_rate        = self.calculate_win_rate(trade_log)
        pf              = self.calculate_profit_factor(trade_log)
        avg_wl          = self.calculate_avg_win_loss_ratio(trade_log)

        start_val       = equity_curve.iloc[0]
        end_val         = equity_curve.iloc[-1]
        total_return    = (end_val - start_val) / start_val

        # ── Benchmark ─────────────────────────────────────────────────────
        alpha, beta, bm_cagr = None, None, None
        if benchmark_curve is not None and len(benchmark_curve) > 10:
            try:
                alpha, beta = self.calculate_alpha_beta(equity_curve, benchmark_curve)
                bm_cagr     = self.calculate_cagr(benchmark_curve)
            except Exception as exc:
                logger.warning("Alpha/beta calculation failed: %s", exc)

        return {
            "total_return_pct":   round(total_return, 6),
            "cagr":               cagr,
            "sharpe_ratio":       sharpe,
            "sortino_ratio":      sortino,
            "calmar_ratio":       calmar,
            "max_drawdown_pct":   max_dd,
            "max_dd_peak_date":   str(pk.date()) if hasattr(pk, "date") else str(pk),
            "max_dd_trough_date": str(tr.date()) if hasattr(tr, "date") else str(tr),
            "total_trades":       len(trade_log),
            "win_rate":           win_rate,
            "profit_factor":      pf,
            "avg_win_loss_ratio": avg_wl,
            "alpha":              alpha,
            "beta":               beta,
            "start_date":         str(equity_curve.index[0].date())
                                  if hasattr(equity_curve.index[0], "date")
                                  else str(equity_curve.index[0]),
            "end_date":           str(equity_curve.index[-1].date())
                                  if hasattr(equity_curve.index[-1], "date")
                                  else str(equity_curve.index[-1]),
            "start_capital":      round(start_val, 2),
            "end_capital":        round(end_val, 2),
            "benchmark_cagr":     bm_cagr,
        }


def _empty_metrics() -> Dict:
    """Return a metrics dict filled with zeros / None when no data is available."""
    return {
        "total_return_pct":   0.0,
        "cagr":               0.0,
        "sharpe_ratio":       0.0,
        "sortino_ratio":      0.0,
        "calmar_ratio":       0.0,
        "max_drawdown_pct":   0.0,
        "max_dd_peak_date":   None,
        "max_dd_trough_date": None,
        "total_trades":       0,
        "win_rate":           0.0,
        "profit_factor":      0.0,
        "avg_win_loss_ratio": 0.0,
        "alpha":              None,
        "beta":               None,
        "start_date":         None,
        "end_date":           None,
        "start_capital":      0.0,
        "end_capital":        0.0,
        "benchmark_cagr":     None,
    }


# ──────────────────────────────────────────────────────────────────────────────
# LEGACY FUNCTION-LEVEL API (backward compatibility for existing code / tests)
# ──────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(
    equity_curve: pd.Series,
    trade_log,
    benchmark_curve: Optional[pd.Series] = None,
    risk_free_rate: float = 0.06,
    trading_days: int = 252,
) -> Dict:
    """
    Legacy wrapper around PerformanceMetrics.generate_summary_report().

    Accepts trade_log as either List[dict] or pd.DataFrame (converts automatically).
    """
    # Accept DataFrame for backward compat
    if isinstance(trade_log, pd.DataFrame):
        trade_log = trade_log.to_dict("records")

    # Build a minimal config object without referencing enclosing scope in a class body
    cfg = BacktestConfig()
    cfg.report.risk_free_rate        = risk_free_rate
    cfg.report.trading_days_per_year = trading_days

    perf = PerformanceMetrics(cfg)
    return perf.generate_summary_report(equity_curve, trade_log, benchmark_curve)
