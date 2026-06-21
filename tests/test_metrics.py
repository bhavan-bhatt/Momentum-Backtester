# tests/test_metrics.py
# ============================================================
# UNIT TESTS — performance/metrics.py (PerformanceMetrics class)
# ============================================================

import numpy as np
import pandas as pd
import pytest
from typing import List

from performance.metrics import PerformanceMetrics, compute_all_metrics
from config import BacktestConfig


# ──────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ──────────────────────────────────────────────────────────────────────────────

def _equity(seed: int = 42, n: int = 252, drift: float = 0.0005) -> pd.Series:
    """Synthetic daily equity curve with DatetimeIndex."""
    rng   = np.random.default_rng(seed)
    rets  = rng.normal(drift, 0.01, n)
    price = 1_000_000.0 * np.cumprod(1 + rets)
    idx   = pd.date_range("2020-01-02", periods=n, freq="B")
    return pd.Series(price, index=idx)


def _trade_log(n_trades: int = 40, seed: int = 42) -> List[dict]:
    """Synthetic trade log as List[dict] with ~55% win rate."""
    rng = np.random.default_rng(seed)
    pnl = rng.normal(2000, 8000, n_trades)
    n_losses = int(n_trades * 0.45)
    pnl[:n_losses] = -np.abs(pnl[:n_losses])
    dates = pd.date_range("2020-01-15", periods=n_trades, freq="5B")
    return [
        {
            "symbol":      "SYM",
            "net_pnl":     float(p),
            "gross_pnl":   float(p) + 20.0,
            "commission":  20.0,
            "entry_date":  dates[i],
            "exit_date":   dates[i] + pd.Timedelta(days=5),
            "quantity":    100,
            "entry_price": 1000.0,
            "exit_price":  1000.0 + float(p) / 100.0,
            "return_pct":  float(p) / 100_000.0,
        }
        for i, p in enumerate(pnl)
    ]


def _perf() -> PerformanceMetrics:
    cfg = BacktestConfig()
    cfg.report.risk_free_rate        = 0.06
    cfg.report.trading_days_per_year = 252
    return PerformanceMetrics(cfg)


# ──────────────────────────────────────────────────────────────────────────────
# calculate_returns
# ──────────────────────────────────────────────────────────────────────────────

class TestCalculateReturns:
    def test_length_is_n_minus_1(self):
        eq  = _equity(n=100)
        ret = _perf().calculate_returns(eq)
        assert len(ret) == len(eq) - 1

    def test_positive_drift_gives_positive_mean(self):
        eq  = _equity(drift=0.002, n=252)
        ret = _perf().calculate_returns(eq)
        assert ret.mean() > 0

    def test_negative_drift(self):
        eq  = _equity(drift=-0.003, n=252)
        ret = _perf().calculate_returns(eq)
        assert ret.mean() < 0


# ──────────────────────────────────────────────────────────────────────────────
# calculate_sharpe_ratio
# ──────────────────────────────────────────────────────────────────────────────

class TestSharpeRatio:
    def test_positive_for_strong_uptrend(self):
        eq = _equity(drift=0.001)
        assert _perf().calculate_sharpe_ratio(eq) > 0

    def test_negative_for_sustained_loss(self):
        eq = _equity(drift=-0.003)
        assert _perf().calculate_sharpe_ratio(eq) < 0

    def test_flat_equity_returns_zero(self):
        eq = pd.Series([1_000_000.0] * 100,
                       index=pd.date_range("2020-01-01", periods=100))
        assert _perf().calculate_sharpe_ratio(eq) == pytest.approx(0.0)

    def test_short_series_no_crash(self):
        eq = pd.Series([100.0, 101.0], index=pd.date_range("2020-01-01", periods=2))
        result = _perf().calculate_sharpe_ratio(eq)
        assert isinstance(result, float)


# ──────────────────────────────────────────────────────────────────────────────
# calculate_cagr
# ──────────────────────────────────────────────────────────────────────────────

class TestCAGR:
    def test_positive_for_positive_drift(self):
        eq = _equity(drift=0.001, n=504)   # ~2 years
        assert _perf().calculate_cagr(eq) > 0

    def test_flat_equity_zero_cagr(self):
        idx = pd.date_range("2020-01-01", periods=252, freq="B")
        eq  = pd.Series([1_000_000.0] * 252, index=idx)
        assert _perf().calculate_cagr(eq) == pytest.approx(0.0, abs=1e-4)

    def test_known_doubling_in_one_year(self):
        # Start at 1M, end at 2M over exactly 1 year → CAGR = 100%
        idx = pd.date_range("2020-01-01", "2020-12-31", periods=252)
        eq  = pd.Series(np.linspace(1_000_000, 2_000_000, 252), index=idx)
        assert _perf().calculate_cagr(eq) == pytest.approx(1.0, rel=0.05)


# ──────────────────────────────────────────────────────────────────────────────
# calculate_max_drawdown
# ──────────────────────────────────────────────────────────────────────────────

class TestMaxDrawdown:
    def test_monotone_increase_no_drawdown(self):
        idx = pd.date_range("2020-01-01", periods=100)
        eq  = pd.Series(np.linspace(100, 200, 100), index=idx)
        dd, _, _ = _perf().calculate_max_drawdown(eq)
        assert dd == pytest.approx(0.0, abs=1e-8)

    def test_known_50_percent_drawdown(self):
        idx = pd.date_range("2020-01-01", periods=3)
        eq  = pd.Series([100.0, 50.0, 150.0], index=idx)
        dd, _, _ = _perf().calculate_max_drawdown(eq)
        assert dd == pytest.approx(-0.5, abs=1e-6)

    def test_returns_three_tuple(self):
        eq   = _equity()
        result = _perf().calculate_max_drawdown(eq)
        assert len(result) == 3

    def test_negative_value_for_losing_equity(self):
        eq   = _equity(drift=-0.002)
        dd, _, _ = _perf().calculate_max_drawdown(eq)
        assert dd < 0


# ──────────────────────────────────────────────────────────────────────────────
# calculate_sortino_ratio
# ──────────────────────────────────────────────────────────────────────────────

class TestSortinoRatio:
    def test_positive_for_uptrend(self):
        eq = _equity(drift=0.0008)
        assert _perf().calculate_sortino_ratio(eq) > 0

    def test_computable(self):
        eq = _equity()
        result = _perf().calculate_sortino_ratio(eq)
        assert isinstance(result, float)
        assert not np.isnan(result)


# ──────────────────────────────────────────────────────────────────────────────
# calculate_calmar_ratio
# ──────────────────────────────────────────────────────────────────────────────

class TestCalmarRatio:
    def test_positive_cagr_positive_dd(self):
        calmar = _perf().calculate_calmar_ratio(0.15, -0.10)
        assert calmar == pytest.approx(1.5, rel=1e-4)

    def test_zero_drawdown_returns_zero(self):
        assert _perf().calculate_calmar_ratio(0.20, 0.0) == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# calculate_win_rate
# ──────────────────────────────────────────────────────────────────────────────

class TestWinRate:
    def test_all_wins(self):
        tl = [{"net_pnl": 100.0}] * 10
        assert _perf().calculate_win_rate(tl) == pytest.approx(1.0)

    def test_all_losses(self):
        tl = [{"net_pnl": -100.0}] * 10
        assert _perf().calculate_win_rate(tl) == pytest.approx(0.0)

    def test_mixed(self):
        tl = _trade_log()
        wr = _perf().calculate_win_rate(tl)
        assert 0.0 < wr < 1.0

    def test_empty_returns_zero(self):
        assert _perf().calculate_win_rate([]) == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# calculate_profit_factor
# ──────────────────────────────────────────────────────────────────────────────

class TestProfitFactor:
    def test_all_wins_returns_inf(self):
        tl = [{"net_pnl": 100.0}] * 5
        pf = _perf().calculate_profit_factor(tl)
        assert pf == float("inf")

    def test_mixed_positive_result(self):
        tl = _trade_log()
        pf = _perf().calculate_profit_factor(tl)
        assert isinstance(pf, float)
        assert pf > 0

    def test_all_losses_returns_zero_gross_profit(self):
        tl = [{"net_pnl": -100.0}] * 5
        pf = _perf().calculate_profit_factor(tl)
        assert pf == pytest.approx(0.0)


# ──────────────────────────────────────────────────────────────────────────────
# calculate_avg_win_loss_ratio
# ──────────────────────────────────────────────────────────────────────────────

class TestAvgWinLossRatio:
    def test_returns_positive_for_mixed_log(self):
        tl     = _trade_log()
        result = _perf().calculate_avg_win_loss_ratio(tl)
        assert isinstance(result, float)
        assert result >= 0

    def test_empty_log_returns_zero(self):
        assert _perf().calculate_avg_win_loss_ratio([]) == 0.0

    def test_only_wins_returns_zero(self):
        tl = [{"net_pnl": 100.0}] * 5
        assert _perf().calculate_avg_win_loss_ratio(tl) == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# calculate_alpha_beta
# ──────────────────────────────────────────────────────────────────────────────

class TestAlphaBeta:
    def test_identical_series_beta_near_one(self):
        eq     = _equity()
        alpha, beta = _perf().calculate_alpha_beta(eq, eq)
        assert beta == pytest.approx(1.0, abs=0.05)

    def test_returns_two_floats(self):
        eq = _equity()
        result = _perf().calculate_alpha_beta(eq, eq)
        assert len(result) == 2
        assert all(isinstance(v, float) for v in result)


# ──────────────────────────────────────────────────────────────────────────────
# calculate_monthly_returns
# ──────────────────────────────────────────────────────────────────────────────

class TestMonthlyReturns:
    def test_returns_dataframe(self):
        eq     = _equity(n=504)   # ~2 years
        result = _perf().calculate_monthly_returns(eq)
        assert isinstance(result, pd.DataFrame)

    def test_columns_are_month_abbreviations(self):
        eq     = _equity(n=504)
        df     = _perf().calculate_monthly_returns(eq)
        months = {"Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"}
        assert set(df.columns).issubset(months)

    def test_rows_are_years(self):
        eq = _equity(n=504)
        df = _perf().calculate_monthly_returns(eq)
        assert all(isinstance(y, (int, np.integer)) for y in df.index)


# ──────────────────────────────────────────────────────────────────────────────
# calculate_drawdown_series
# ──────────────────────────────────────────────────────────────────────────────

class TestDrawdownSeries:
    def test_all_nonpositive(self):
        eq     = _equity()
        dd     = _perf().calculate_drawdown_series(eq)
        assert (dd <= 0).all()

    def test_zero_at_new_highs(self):
        idx = pd.date_range("2020-01-01", periods=3)
        eq  = pd.Series([100.0, 110.0, 120.0], index=idx)
        dd  = _perf().calculate_drawdown_series(eq)
        assert dd.iloc[-1] == pytest.approx(0.0)


# ──────────────────────────────────────────────────────────────────────────────
# calculate_rolling_sharpe
# ──────────────────────────────────────────────────────────────────────────────

class TestRollingSharpe:
    def test_returns_series_same_length(self):
        eq = _equity(n=504)
        rs = _perf().calculate_rolling_sharpe(eq, window=60)
        assert isinstance(rs, pd.Series)
        assert len(rs) == len(eq) - 1   # returns are one shorter than equity

    def test_first_values_nan_within_window(self):
        eq = _equity(n=504)
        rs = _perf().calculate_rolling_sharpe(eq, window=60)
        # First window - 1 values are NaN
        assert rs.iloc[:58].isna().all()


# ──────────────────────────────────────────────────────────────────────────────
# generate_summary_report
# ──────────────────────────────────────────────────────────────────────────────

class TestGenerateSummaryReport:
    _REQUIRED_KEYS = [
        "total_return_pct", "cagr", "sharpe_ratio", "sortino_ratio",
        "calmar_ratio", "max_drawdown_pct", "max_dd_peak_date",
        "max_dd_trough_date", "total_trades", "win_rate", "profit_factor",
        "avg_win_loss_ratio", "alpha", "beta", "start_date", "end_date",
        "start_capital", "end_capital", "benchmark_cagr",
    ]

    def test_returns_all_expected_keys(self):
        eq      = _equity()
        tl      = _trade_log()
        bench   = _equity(seed=99)
        metrics = _perf().generate_summary_report(eq, tl, bench)
        for key in self._REQUIRED_KEYS:
            assert key in metrics, f"Missing key: {key}"

    def test_no_benchmark_works(self):
        eq      = _equity()
        tl      = _trade_log()
        metrics = _perf().generate_summary_report(eq, tl)
        assert metrics["alpha"] is None
        assert metrics["beta"]  is None

    def test_empty_trade_log(self):
        eq      = _equity()
        metrics = _perf().generate_summary_report(eq, [])
        assert metrics["total_trades"] == 0
        assert metrics["win_rate"]     == 0.0

    def test_total_trades_matches_log(self):
        eq      = _equity()
        tl      = _trade_log(n_trades=25)
        metrics = _perf().generate_summary_report(eq, tl)
        assert metrics["total_trades"] == 25

    def test_start_end_capital_present(self):
        eq      = _equity()
        metrics = _perf().generate_summary_report(eq, [])
        assert metrics["start_capital"] == pytest.approx(eq.iloc[0], rel=1e-4)
        assert metrics["end_capital"]   == pytest.approx(eq.iloc[-1], rel=1e-4)


# ──────────────────────────────────────────────────────────────────────────────
# LEGACY compute_all_metrics (backward-compat wrapper)
# ──────────────────────────────────────────────────────────────────────────────

class TestComputeAllMetrics:
    def test_accepts_list_of_dicts(self):
        eq      = _equity()
        tl      = _trade_log()
        metrics = compute_all_metrics(eq, tl)
        assert "total_return_pct" in metrics

    def test_accepts_dataframe(self):
        eq = _equity()
        df = pd.DataFrame(_trade_log())
        metrics = compute_all_metrics(eq, df)
        assert "total_return_pct" in metrics

    def test_no_crash_on_empty(self):
        eq      = _equity()
        metrics = compute_all_metrics(eq, [])
        assert isinstance(metrics, dict)
