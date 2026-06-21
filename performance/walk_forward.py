# performance/walk_forward.py
# ============================================================
# WALK-FORWARD VALIDATION ENGINE
#
# Splits historical data into anchored or rolling train/test windows,
# runs a backtest on each test split, and aggregates out-of-sample
# performance metrics across all splits.
#
# Window Types
# ------------
# Anchored  : Train window starts fixed; only end grows (expanding window).
# Rolling   : Both start and end advance (fixed-size window). ← Implemented here.
#
# Each split:
#   [train_start ──────── train_end] [test_start ── test_end]
#                                    ↑ out-of-sample results
# ============================================================

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Type

import pandas as pd

from engine.data_handler import DataHandler
from engine.backtest import build_backtest_engine
from engine.strategy import BaseStrategy
from performance.metrics import compute_all_metrics
from config import BacktestConfig

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


@dataclass
class WalkForwardSplit:
    """Represents one train/test split."""
    split_idx:    int
    train_start:  datetime
    train_end:    datetime
    test_start:   datetime
    test_end:     datetime
    train_bars:   int = 0
    test_bars:    int = 0
    metrics:      Dict[str, float] = field(default_factory=dict)
    equity_curve: Optional[pd.Series] = None


class WalkForwardEngine:
    """
    Orchestrates walk-forward validation of a strategy.

    Usage
    -----
    engine = WalkForwardEngine(config, strategy_cls)
    results = engine.run()
    summary = engine.summary()

    Parameters
    ----------
    config        : BacktestConfig — master config (walk_forward sub-config used).
    strategy_cls  : Type[Strategy] — strategy class to instantiate per split.
    """

    def __init__(self, config: BacktestConfig, strategy_cls: Type[BaseStrategy]) -> None:
        self._config       = config
        self._strategy_cls = strategy_cls
        self._splits:  List[WalkForwardSplit] = []

    def generate_splits(self, all_dates: List[datetime]) -> List["WalkForwardSplit"]:
        """
        Produce train/test split metadata from a sorted list of trading dates.

        Parameters
        ----------
        all_dates : List[datetime] — all common trading dates from DataHandler.

        Returns
        -------
        List[WalkForwardSplit] — one entry per test window.
        """
        wfcfg = self._config.walk_forward
        train_days = wfcfg.train_years * TRADING_DAYS_PER_YEAR
        test_days  = wfcfg.test_years  * TRADING_DAYS_PER_YEAR
        step_days  = wfcfg.step_years  * TRADING_DAYS_PER_YEAR

        splits: List[WalkForwardSplit] = []
        n = len(all_dates)
        split_idx = 0
        cursor = 0  # index into all_dates where current train window begins

        while cursor + train_days + test_days <= n:
            train_start_idx = cursor
            train_end_idx   = cursor + train_days - 1
            test_start_idx  = cursor + train_days
            test_end_idx    = min(cursor + train_days + test_days - 1, n - 1)

            train_bars = train_end_idx - train_start_idx + 1
            test_bars  = test_end_idx  - test_start_idx  + 1

            if train_bars < wfcfg.min_train_bars:
                logger.warning(
                    "Split %d skipped: only %d train bars (min=%d).",
                    split_idx, train_bars, wfcfg.min_train_bars,
                )
                cursor += step_days
                continue

            split = WalkForwardSplit(
                split_idx   = split_idx,
                train_start = all_dates[train_start_idx],
                train_end   = all_dates[train_end_idx],
                test_start  = all_dates[test_start_idx],
                test_end    = all_dates[test_end_idx],
                train_bars  = train_bars,
                test_bars   = test_bars,
            )
            splits.append(split)
            split_idx += 1
            cursor    += step_days

        logger.info("Generated %d walk-forward splits.", len(splits))
        return splits

    def run(self) -> List[WalkForwardSplit]:
        """
        Execute the full walk-forward validation.

        For each split:
          1. Clone config with test window's date range.
          2. Run a full backtest on just the test window.
          3. Compute out-of-sample performance metrics.
          4. Store results in the split.

        Returns
        -------
        List[WalkForwardSplit] with metrics populated.
        """
        # Build a temporary engine just to access aligned dates
        temp_engine = build_backtest_engine(self._config, self._strategy_cls)
        all_dates   = temp_engine.data_handler._all_dates
        del temp_engine

        splits = self.generate_splits(all_dates)
        if not splits:
            logger.error("No walk-forward splits generated. Check date range and window sizes.")
            return []

        self._splits = splits
        rcfg = self._config.report

        for split in splits:
            if self._config.verbose:
                print(
                    f"\n[WF Split {split.split_idx + 1}/{len(splits)}]  "
                    f"Train: {split.train_start.date()} → {split.train_end.date()}  "
                    f"| Test: {split.test_start.date()} → {split.test_end.date()}"
                )

            # Clone config with test window's date range
            test_cfg = self._make_test_config(split)

            try:
                bt      = build_backtest_engine(test_cfg, self._strategy_cls)
                results = bt.run()

                equity    = results["equity_curve"]
                trade_log = results["trade_log"]

                bench_equity = None
                if bt.data_handler.benchmark_data is not None:
                    bench_eq = bt.data_handler.benchmark_data["close"]
                    bench_equity = bench_eq / bench_eq.iloc[0] * test_cfg.portfolio.initial_capital

                metrics = compute_all_metrics(
                    equity,
                    trade_log if trade_log is not None else pd.DataFrame(),
                    benchmark_equity=bench_equity,
                    risk_free_rate=rcfg.risk_free_rate,
                    trading_days=rcfg.trading_days_per_year,
                )
                split.metrics      = metrics
                split.equity_curve = equity

            except Exception as exc:
                logger.error("Walk-forward split %d failed: %s", split.split_idx, exc)
                split.metrics = {}

        return splits

    def summary(self) -> pd.DataFrame:
        """
        Return a DataFrame with one row per split and columns for each metric.
        Also appends a final row with the mean across all splits.
        """
        if not self._splits:
            return pd.DataFrame()

        rows = []
        for s in self._splits:
            row = {
                "split":       s.split_idx + 1,
                "train_start": s.train_start.date(),
                "train_end":   s.train_end.date(),
                "test_start":  s.test_start.date(),
                "test_end":    s.test_end.date(),
                "train_bars":  s.train_bars,
                "test_bars":   s.test_bars,
            }
            row.update(s.metrics)
            rows.append(row)

        df = pd.DataFrame(rows)
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        mean_row = {"split": "MEAN"}
        for col in numeric_cols:
            if col not in ("split", "train_bars", "test_bars"):
                mean_row[col] = df[col].mean()

        df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
        return df

    def _make_test_config(self, split: WalkForwardSplit) -> BacktestConfig:
        """
        Return a BacktestConfig with start_date/end_date restricted to the
        test window of this split.
        """
        import copy
        cfg = copy.deepcopy(self._config)
        cfg.data.start_date = split.test_start.strftime("%Y-%m-%d")
        cfg.data.end_date   = split.test_end.strftime("%Y-%m-%d")
        cfg.verbose         = False  # suppress per-split verbosity
        return cfg
