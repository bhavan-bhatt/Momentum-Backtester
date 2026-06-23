# performance/statistics.py
# ============================================================
# STATISTICAL SIGNIFICANCE TESTING
# Bootstrap Sharpe CI + Deflated Sharpe Ratio (Bailey & López de Prado)
# ============================================================

import math
import sys
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from config import BacktestConfig


class StatisticalTests:
    """
    Quantifies confidence in a backtested Sharpe ratio via bootstrap CI
    and Deflated Sharpe Ratio (multiple-testing correction).
    """

    def __init__(self, config: BacktestConfig) -> None:
        self.n_iterations = config.advanced.statistics.bootstrap_n_iterations
        self.block_size = config.advanced.statistics.bootstrap_block_size
        self.confidence_level = config.advanced.statistics.confidence_level
        self.n_strategies_tested = config.advanced.statistics.n_strategies_tested
        self.trading_days = config.report.trading_days_per_year
        np.random.seed(config.random_seed)

    @staticmethod
    def _sharpe_from_array(arr: np.ndarray, trading_days: int) -> float:
        std_r = arr.std()
        if std_r <= 0 or len(arr) < 2:
            return 0.0
        return float(arr.mean() / std_r * math.sqrt(trading_days))

    def block_bootstrap_sharpe_ci(
        self, returns: pd.Series
    ) -> Tuple[float, float, float]:
        """Block-bootstrap confidence interval for annualised Sharpe ratio."""
        returns_arr = returns.dropna().values
        n = len(returns_arr)
        if n < self.block_size + 2:
            point = self._sharpe_from_array(returns_arr, self.trading_days)
            return point, point, point

        n_blocks = int(np.ceil(n / self.block_size))
        max_start = max(n - self.block_size, 1)
        bootstrap_sharpes = np.empty(self.n_iterations)

        for i in tqdm(
            range(self.n_iterations),
            desc=f"Bootstrap Sharpe ({self.n_iterations} draws)",
            unit="iter",
            file=sys.stdout,
            leave=False,
        ):
            starts = np.random.randint(0, max_start, size=n_blocks)
            blocks = [returns_arr[s : s + self.block_size] for s in starts]
            resampled = np.concatenate(blocks)[:n]
            bootstrap_sharpes[i] = self._sharpe_from_array(resampled, self.trading_days)

        point_estimate = self._sharpe_from_array(returns_arr, self.trading_days)
        alpha = 1.0 - self.confidence_level
        lower = float(np.percentile(bootstrap_sharpes, 100 * alpha / 2))
        upper = float(np.percentile(bootstrap_sharpes, 100 * (1 - alpha / 2)))
        return point_estimate, lower, upper

    def probabilistic_sharpe_ratio(
        self,
        observed_sharpe: float,
        n_observations: int,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
        benchmark_sharpe: float = 0.0,
    ) -> float:
        """Probabilistic Sharpe Ratio — P(true Sharpe > benchmark)."""
        if n_observations <= 1:
            return 0.5

        sr = observed_sharpe
        numerator = (sr - benchmark_sharpe) * math.sqrt(n_observations - 1)
        denominator = math.sqrt(
            1.0 - skewness * sr + ((kurtosis - 1.0) / 4.0) * sr ** 2
        )
        if denominator <= 0:
            return 0.5
        z = numerator / denominator
        return float(stats.norm.cdf(z))

    def deflated_sharpe_ratio(
        self, returns: pd.Series, n_strategies_tested: Optional[int] = None
    ) -> dict:
        """Deflated Sharpe Ratio correcting for multiple strategy trials."""
        n_trials = n_strategies_tested or self.n_strategies_tested
        daily_returns = returns.dropna().values
        n = len(daily_returns)
        if n < 2:
            return {
                "deflated_sharpe_ratio": 0.5,
                "daily_sharpe": 0.0,
                "annualised_sharpe": 0.0,
                "n_trials_assumed": n_trials,
                "expected_max_sharpe_by_chance": 0.0,
                "skewness": 0.0,
                "kurtosis": 3.0,
                "interpretation": self._interpret_dsr(0.5),
            }

        std = daily_returns.std()
        daily_sharpe = float(daily_returns.mean() / std) if std > 0 else 0.0
        skew = float(stats.skew(daily_returns))
        kurt = float(stats.kurtosis(daily_returns, fisher=False))

        euler_mascheroni = 0.5772156649
        if n_trials <= 1:
            expected_max_sharpe = 0.0
        else:
            expected_max_sharpe = (
                (1.0 - euler_mascheroni)
                * stats.norm.ppf(1.0 - 1.0 / n_trials)
                + euler_mascheroni
                * stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
            ) * math.sqrt(1.0 / n)

        dsr = self.probabilistic_sharpe_ratio(
            observed_sharpe=daily_sharpe,
            n_observations=n,
            skewness=skew,
            kurtosis=kurt,
            benchmark_sharpe=expected_max_sharpe,
        )

        return {
            "deflated_sharpe_ratio": dsr,
            "daily_sharpe": daily_sharpe,
            "annualised_sharpe": daily_sharpe * math.sqrt(self.trading_days),
            "n_trials_assumed": n_trials,
            "expected_max_sharpe_by_chance": expected_max_sharpe * math.sqrt(
                self.trading_days
            ),
            "skewness": skew,
            "kurtosis": kurt,
            "interpretation": self._interpret_dsr(dsr),
        }

    def _interpret_dsr(self, dsr: float) -> str:
        if dsr > 0.95:
            return "Strong evidence of genuine skill, not overfitting."
        if dsr > 0.5:
            return (
                "Plausible edge, but not strongly distinguished from luck "
                "given the number of strategies tested."
            )
        return (
            "Observed performance is consistent with having tried multiple "
            "strategies and reported the best by chance — not validated."
        )

    def jarque_bera_normality_test(self, returns: pd.Series) -> dict:
        """Jarque-Bera test for return normality."""
        clean = returns.dropna().values
        if len(clean) < 8:
            return {
                "statistic": None,
                "p_value": None,
                "is_normal": None,
                "interpretation": "Insufficient data for normality test.",
            }
        stat, pvalue = stats.jarque_bera(clean)
        is_normal = pvalue > 0.05
        interpretation = (
            "Returns approximate normal distribution"
            if is_normal
            else (
                "Returns significantly deviate from normal — interpret Sharpe with "
                "Sortino and tail ratio as well, not in isolation."
            )
        )
        return {
            "statistic": float(stat),
            "p_value": float(pvalue),
            "is_normal": is_normal,
            "interpretation": interpretation,
        }
