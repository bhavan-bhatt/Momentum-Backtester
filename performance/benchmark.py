# performance/benchmark.py
# ============================================================
# BUY-AND-HOLD BENCHMARK, WITH AND WITHOUT TRANSACTION COSTS
# ============================================================

from typing import Dict, Optional

import pandas as pd

from config import BacktestConfig


class BenchmarkComparison:
    """Fair buy-and-hold benchmarks with optional transaction costs."""

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config
        self.initial_capital = config.portfolio.initial_capital

    def _entry_cost_pct(self) -> float:
        ecfg = self.config.execution
        return ecfg.commission_pct + ecfg.exchange_charges_pct + ecfg.stamp_duty_pct

    def _exit_cost_pct(self) -> float:
        ecfg = self.config.execution
        return ecfg.commission_pct + ecfg.exchange_charges_pct + ecfg.stt_pct

    def build_buy_and_hold_curve(
        self, benchmark_prices: pd.Series, apply_costs: bool = False
    ) -> pd.Series:
        """Simulate buy-and-hold on benchmark index prices."""
        if benchmark_prices is None or len(benchmark_prices) < 2:
            return pd.Series(dtype=float)

        start_price = float(benchmark_prices.iloc[0])
        if apply_costs:
            effective_start = start_price * (1.0 + self._entry_cost_pct())
            shares = self.initial_capital / effective_start
        else:
            shares = self.initial_capital / start_price

        curve = benchmark_prices.astype(float) * shares
        if apply_costs:
            curve = curve.copy()
            curve.iloc[-1] = curve.iloc[-1] * (1.0 - self._exit_cost_pct())
        return curve

    def build_equal_weight_basket_curve(
        self, all_symbol_data: Dict[str, pd.Series], apply_costs: bool = False
    ) -> pd.Series:
        """Equal-weight buy-and-hold of the traded stock universe."""
        if not all_symbol_data:
            return pd.Series(dtype=float)

        aligned = pd.DataFrame(all_symbol_data).dropna(how="any")
        if aligned.empty:
            return pd.Series(dtype=float)

        per_symbol_capital = self.initial_capital / len(all_symbol_data)
        curves = []

        for symbol, prices in all_symbol_data.items():
            s = prices.reindex(aligned.index).dropna()
            if s.empty:
                continue
            start = float(s.iloc[0])
            if apply_costs:
                shares = per_symbol_capital / (start * (1.0 + self._entry_cost_pct()))
                sym_curve = s * shares
                sym_curve = sym_curve.copy()
                sym_curve.iloc[-1] = sym_curve.iloc[-1] * (1.0 - self._exit_cost_pct())
            else:
                shares = per_symbol_capital / start
                sym_curve = s * shares
            curves.append(sym_curve.reindex(aligned.index).ffill())

        if not curves:
            return pd.Series(dtype=float)

        basket = sum(curves)
        return basket

    def compare(
        self,
        strategy_curve: pd.Series,
        benchmark_curve: pd.Series,
        benchmark_label: str = "Nifty 50 Buy & Hold",
    ) -> dict:
        """Side-by-side comparison of strategy vs benchmark."""
        aligned_s, aligned_b = strategy_curve.align(benchmark_curve, join="inner")
        if len(aligned_s) < 2:
            return {
                "benchmark_label": benchmark_label,
                "strategy_total_return": 0.0,
                "benchmark_total_return": 0.0,
                "outperformance": 0.0,
                "return_correlation": None,
            }

        strategy_total = aligned_s.iloc[-1] / aligned_s.iloc[0] - 1.0
        benchmark_total = aligned_b.iloc[-1] / aligned_b.iloc[0] - 1.0
        correlation = aligned_s.pct_change().corr(aligned_b.pct_change())

        return {
            "benchmark_label": benchmark_label,
            "strategy_total_return": round(float(strategy_total), 4),
            "benchmark_total_return": round(float(benchmark_total), 4),
            "outperformance": round(float(strategy_total - benchmark_total), 4),
            "return_correlation": round(float(correlation), 4)
            if correlation == correlation
            else None,
        }
