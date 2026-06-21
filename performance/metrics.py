# performance/metrics.py
# ============================================================
# PERFORMANCE METRICS LIBRARY
#
# All functions accept a pd.Series of daily portfolio equity or returns
# and return scalar values. No side-effects; fully composable.
#
# Annualisation convention: uses trading_days_per_year from ReportConfig.
# ============================================================

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def equity_to_returns(equity: pd.Series) -> pd.Series:
    """Convert an equity series to daily percentage returns."""
    return equity.pct_change().dropna()


def _validate_returns(returns: pd.Series, name: str) -> bool:
    """Return False if returns series is empty or all-NaN."""
    if returns is None or len(returns) == 0 or returns.isna().all():
        logger.warning("%s: empty or all-NaN returns series.", name)
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# CORE METRICS
# ══════════════════════════════════════════════════════════════════════════════

def total_return(equity: pd.Series) -> float:
    """
    Total return over the full period.

    Returns (final_equity / initial_equity) - 1.
    """
    if len(equity) < 2:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def cagr(equity: pd.Series, trading_days: int = 252) -> float:
    """
    Compound Annual Growth Rate.

    CAGR = (final / initial)^(trading_days / n_days) - 1

    Parameters
    ----------
    equity       : pd.Series indexed by datetime.
    trading_days : int — trading days per year (typically 252).
    """
    if len(equity) < 2:
        return 0.0

    n_days = len(equity)
    total  = equity.iloc[-1] / equity.iloc[0]
    if total <= 0:
        return -1.0
    return float(total ** (trading_days / n_days) - 1.0)


def annualised_volatility(returns: pd.Series, trading_days: int = 252) -> float:
    """
    Annualised standard deviation of daily returns.

    σ_annual = σ_daily × √trading_days
    """
    if not _validate_returns(returns, "annualised_volatility"):
        return np.nan
    return float(returns.std() * np.sqrt(trading_days))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.065,
    trading_days: int = 252,
) -> float:
    """
    Annualised Sharpe Ratio.

    Sharpe = (annualised_return - risk_free_rate) / annualised_volatility

    Parameters
    ----------
    returns        : pd.Series of daily returns.
    risk_free_rate : float — annual risk-free rate (decimal, e.g. 0.065).
    trading_days   : int.
    """
    if not _validate_returns(returns, "sharpe_ratio"):
        return np.nan

    daily_rf     = (1.0 + risk_free_rate) ** (1.0 / trading_days) - 1.0
    excess       = returns - daily_rf
    ann_excess   = excess.mean() * trading_days
    ann_vol      = returns.std() * np.sqrt(trading_days)

    if ann_vol == 0:
        return np.nan
    return float(ann_excess / ann_vol)


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.065,
    trading_days: int = 252,
) -> float:
    """
    Annualised Sortino Ratio (uses downside deviation only).

    Sortino = (annualised_return - risk_free_rate) / downside_deviation
    """
    if not _validate_returns(returns, "sortino_ratio"):
        return np.nan

    daily_rf    = (1.0 + risk_free_rate) ** (1.0 / trading_days) - 1.0
    excess      = returns - daily_rf
    ann_excess  = excess.mean() * trading_days

    downside    = returns[returns < daily_rf] - daily_rf
    if len(downside) == 0:
        return np.nan
    downside_std = np.sqrt((downside ** 2).mean()) * np.sqrt(trading_days)

    if downside_std == 0:
        return np.nan
    return float(ann_excess / downside_std)


def calmar_ratio(equity: pd.Series, trading_days: int = 252) -> float:
    """
    Calmar Ratio = CAGR / |Max Drawdown|.
    """
    max_dd = max_drawdown(equity)
    if max_dd == 0:
        return np.nan
    return float(cagr(equity, trading_days) / abs(max_dd))


def max_drawdown(equity: pd.Series) -> float:
    """
    Maximum drawdown as a negative fraction.

    MDD = min((equity - running_peak) / running_peak)
    """
    if len(equity) < 2:
        return 0.0
    rolling_max = equity.cummax()
    drawdown    = (equity - rolling_max) / rolling_max
    return float(drawdown.min())


def drawdown_series(equity: pd.Series) -> pd.Series:
    """
    Full drawdown series (negative fractions at each point in time).
    """
    rolling_max = equity.cummax()
    return (equity - rolling_max) / rolling_max


def max_drawdown_duration(equity: pd.Series) -> int:
    """
    Maximum number of consecutive bars spent below the previous equity peak.
    Returns the duration in bars (trading days).
    """
    dd  = drawdown_series(equity)
    in_dd = (dd < 0).astype(int)
    # Count consecutive runs below peak
    duration    = 0
    max_duration = 0
    for val in in_dd:
        if val:
            duration += 1
            max_duration = max(max_duration, duration)
        else:
            duration = 0
    return max_duration


def win_rate(trade_log: pd.DataFrame) -> float:
    """
    Fraction of closed trades that were profitable.

    Parameters
    ----------
    trade_log : pd.DataFrame — must contain 'realised_pnl' and 'type' columns.
    """
    exits = trade_log[trade_log["type"] == "EXIT"]
    if exits.empty:
        return np.nan
    wins = (exits["realised_pnl"] > 0).sum()
    return float(wins / len(exits))


def profit_factor(trade_log: pd.DataFrame) -> float:
    """
    Sum of winning PnL / |Sum of losing PnL|.
    Values > 1 indicate a profitable system.
    """
    exits = trade_log[trade_log["type"] == "EXIT"]
    if exits.empty:
        return np.nan
    gross_profit = exits.loc[exits["realised_pnl"] > 0, "realised_pnl"].sum()
    gross_loss   = exits.loc[exits["realised_pnl"] < 0, "realised_pnl"].sum()
    if gross_loss == 0:
        return np.inf
    return float(gross_profit / abs(gross_loss))


def average_trade_return(trade_log: pd.DataFrame) -> float:
    """Mean realised PnL per exit trade."""
    exits = trade_log[trade_log["type"] == "EXIT"]
    if exits.empty:
        return np.nan
    return float(exits["realised_pnl"].mean())


def expectancy(trade_log: pd.DataFrame) -> float:
    """
    Expectancy per trade = (win_rate × avg_win) + (loss_rate × avg_loss).

    Positive expectancy is a prerequisite for a viable system.
    """
    exits = trade_log[trade_log["type"] == "EXIT"]
    if exits.empty:
        return np.nan
    winners = exits[exits["realised_pnl"] > 0]["realised_pnl"]
    losers  = exits[exits["realised_pnl"] < 0]["realised_pnl"]
    wr      = len(winners) / len(exits)
    lr      = 1.0 - wr
    avg_win = winners.mean() if not winners.empty else 0.0
    avg_loss = losers.mean() if not losers.empty else 0.0
    return float(wr * avg_win + lr * avg_loss)


def alpha_beta(
    equity: pd.Series,
    benchmark_equity: pd.Series,
    risk_free_rate: float = 0.065,
    trading_days: int = 252,
) -> Tuple[float, float]:
    """
    Jensen's Alpha and Beta vs. a benchmark.

    Both series must be aligned (same dates, same length).

    Returns
    -------
    (alpha_annual, beta) — alpha is annualised.
    """
    port_ret  = equity_to_returns(equity)
    bench_ret = equity_to_returns(benchmark_equity)

    # Align on common dates
    aligned   = pd.concat([port_ret, bench_ret], axis=1, join="inner").dropna()
    if len(aligned) < 10:
        return (np.nan, np.nan)

    p, b     = aligned.iloc[:, 0], aligned.iloc[:, 1]
    cov_mat  = np.cov(p, b)
    beta     = cov_mat[0, 1] / cov_mat[1, 1] if cov_mat[1, 1] != 0 else np.nan

    daily_rf   = (1.0 + risk_free_rate) ** (1.0 / trading_days) - 1.0
    alpha_daily = (p.mean() - daily_rf) - beta * (b.mean() - daily_rf)
    alpha_annual = (1.0 + alpha_daily) ** trading_days - 1.0

    return (float(alpha_annual), float(beta))


def information_ratio(
    equity: pd.Series,
    benchmark_equity: pd.Series,
    trading_days: int = 252,
) -> float:
    """
    Information Ratio = annualised active return / tracking error.
    """
    port_ret  = equity_to_returns(equity)
    bench_ret = equity_to_returns(benchmark_equity)
    aligned   = pd.concat([port_ret, bench_ret], axis=1, join="inner").dropna()
    if len(aligned) < 10:
        return np.nan

    active  = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    ann_act = active.mean() * trading_days
    te      = active.std() * np.sqrt(trading_days)
    if te == 0:
        return np.nan
    return float(ann_act / te)


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical VaR at the given confidence level.
    Returns the loss at the given percentile (negative number).

    E.g., VaR(0.95) = -0.02 means 95% of days have losses < 2%.
    """
    if not _validate_returns(returns, "value_at_risk"):
        return np.nan
    return float(np.percentile(returns.dropna(), (1.0 - confidence) * 100))


def conditional_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Conditional Value at Risk (Expected Shortfall).
    Mean of losses beyond the VaR threshold.
    """
    if not _validate_returns(returns, "conditional_var"):
        return np.nan
    var = value_at_risk(returns, confidence)
    tail = returns[returns <= var]
    if tail.empty:
        return var
    return float(tail.mean())


# ══════════════════════════════════════════════════════════════════════════════
# AGGREGATE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def compute_all_metrics(
    equity: pd.Series,
    trade_log: pd.DataFrame,
    benchmark_equity: Optional[pd.Series] = None,
    risk_free_rate: float = 0.065,
    trading_days: int = 252,
) -> Dict[str, float]:
    """
    Compute the full suite of performance metrics and return as a dict.

    Parameters
    ----------
    equity           : pd.Series — portfolio equity curve indexed by date.
    trade_log        : pd.DataFrame — output of PortfolioManager.get_trade_log().
    benchmark_equity : pd.Series or None — benchmark equity curve.
    risk_free_rate   : float — annual risk-free rate.
    trading_days     : int.

    Returns
    -------
    Dict[str, float] — all metrics keyed by name.
    """
    returns = equity_to_returns(equity)

    results: Dict[str, float] = {
        "total_return":         total_return(equity),
        "cagr":                 cagr(equity, trading_days),
        "annualised_volatility": annualised_volatility(returns, trading_days),
        "sharpe_ratio":         sharpe_ratio(returns, risk_free_rate, trading_days),
        "sortino_ratio":        sortino_ratio(returns, risk_free_rate, trading_days),
        "calmar_ratio":         calmar_ratio(equity, trading_days),
        "max_drawdown":         max_drawdown(equity),
        "max_dd_duration_days": float(max_drawdown_duration(equity)),
        "var_95":               value_at_risk(returns, 0.95),
        "cvar_95":              conditional_var(returns, 0.95),
    }

    if not trade_log.empty:
        results["win_rate"]        = win_rate(trade_log)
        results["profit_factor"]   = profit_factor(trade_log)
        results["avg_trade_pnl"]   = average_trade_return(trade_log)
        results["expectancy"]      = expectancy(trade_log)
        results["num_trades"]      = float(len(trade_log[trade_log["type"] == "EXIT"]))
    else:
        results.update({k: np.nan for k in [
            "win_rate", "profit_factor", "avg_trade_pnl", "expectancy", "num_trades"
        ]})

    if benchmark_equity is not None and len(benchmark_equity) > 1:
        alpha, beta = alpha_beta(equity, benchmark_equity, risk_free_rate, trading_days)
        ir          = information_ratio(equity, benchmark_equity, trading_days)
        results["alpha"]            = alpha
        results["beta"]             = beta
        results["information_ratio"] = ir
        results["benchmark_return"] = total_return(benchmark_equity)
    else:
        results.update({"alpha": np.nan, "beta": np.nan,
                        "information_ratio": np.nan, "benchmark_return": np.nan})

    return results


# ══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE METRICS CLASS — object-oriented wrapper around compute_all_metrics
# ══════════════════════════════════════════════════════════════════════════════

class PerformanceMetrics:
    """
    Object-oriented wrapper used by BacktestEngine._finalise().

    Usage
    -----
    perf    = PerformanceMetrics(config)
    metrics = perf.generate_summary_report(equity_curve, trade_log, benchmark_curve)
    """

    def __init__(self, config) -> None:
        """
        Parameters
        ----------
        config : BacktestConfig
        """
        self._config = config

    def generate_summary_report(
        self,
        equity_curve: pd.Series,
        trade_log: pd.DataFrame,
        benchmark_curve: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """
        Compute the full suite of performance metrics.

        Parameters
        ----------
        equity_curve    : pd.Series indexed by date — portfolio equity.
        trade_log       : pd.DataFrame from PortfolioManager.get_trade_log().
        benchmark_curve : pd.Series or None — benchmark equity (same scale).

        Returns
        -------
        Dict[str, float] — all metrics keyed by name.
        """
        rcfg = self._config.report
        return compute_all_metrics(
            equity=equity_curve,
            trade_log=trade_log if trade_log is not None else pd.DataFrame(),
            benchmark_equity=benchmark_curve,
            risk_free_rate=rcfg.risk_free_rate,
            trading_days=rcfg.trading_days_per_year,
        )
