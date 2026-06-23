# performance/walk_forward.py
# ============================================================
# WALK-FORWARD VALIDATION ENGINE
# Splits data into rolling train/test windows, runs a full backtest
# on each split, and aggregates out-of-sample metrics.
# ============================================================

import copy
import itertools
import logging
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

from config import BacktestConfig, StrategyConfig
from engine.strategy import BaseStrategy

logger = logging.getLogger(__name__)


class WalkForwardEngine:
    """
    Automates walk-forward validation to produce honest out-of-sample results.

    Example (train_years=3, test_years=1)
    --------------------------------------
    Split 1: Train [2016 → 2018]  | Test [2019]
    Split 2: Train [2017 → 2019]  | Test [2020]
    Split 3: Train [2018 → 2020]  | Test [2021]
    Split 4: Train [2019 → 2021]  | Test [2022]

    For each split:
      1. (Optional) Optimise strategy params on training window.
      2. Run a full backtest on the test window.
      3. Record out-of-sample metrics.
    Final result = aggregate (mean ± std) across all test windows.
    """

    def __init__(self, config: BacktestConfig) -> None:
        self._config       = config
        self.split_results: List[dict] = []
        self.splits:        List[Tuple[str, str, str, str]] = []

    # ──────────────────────────────────────────────────────────────────────
    # SPLIT GENERATION
    # ──────────────────────────────────────────────────────────────────────

    def generate_splits(self) -> List[Tuple[str, str, str, str]]:
        """
        Generate all (train_start, train_end, test_start, test_end) date tuples.

        Returns
        -------
        List of 4-tuples of "YYYY-MM-DD" strings.
        """
        wf   = self._config.walk_forward
        data = self._config.data

        train_years = wf.train_years
        test_years  = wf.test_years
        step_years  = getattr(wf, "step_years", test_years)

        data_start = datetime.strptime(data.start_date, "%Y-%m-%d")
        data_end   = datetime.strptime(data.end_date,   "%Y-%m-%d")

        self.splits = []
        current_train_start = data_start

        while True:
            train_end  = current_train_start + relativedelta(years=train_years) - timedelta(days=1)
            test_start = train_end + timedelta(days=1)
            test_end   = test_start + relativedelta(years=test_years) - timedelta(days=1)

            if test_end > data_end:
                break

            self.splits.append((
                current_train_start.strftime("%Y-%m-%d"),
                train_end.strftime("%Y-%m-%d"),
                test_start.strftime("%Y-%m-%d"),
                test_end.strftime("%Y-%m-%d"),
            ))

            current_train_start += relativedelta(years=step_years)

        logger.info("Generated %d walk-forward splits.", len(self.splits))
        return self.splits

    # ──────────────────────────────────────────────────────────────────────
    # SINGLE SPLIT EXECUTION
    # ──────────────────────────────────────────────────────────────────────

    def run_single_split(
        self,
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str,
        strategy_class: Type[BaseStrategy],
        split_index: int,
    ) -> dict:
        """
        Run one full backtest on the test window and return its metrics.
        """
        logger.info(
            "Split %d | test window: %s → %s", split_index, test_start, test_end
        )

        # ── Build config for this test window ─────────────────────────────
        split_config = copy.deepcopy(self._config)
        split_config.data.start_date = test_start
        split_config.data.end_date   = test_end
        split_config.verbose         = False   # quieter during WF runs

        # ── Optional in-sample optimisation ──────────────────────────────
        if split_config.walk_forward.optimize_in_train:
            best_strategy_cfg = self._optimize_on_train(
                train_start, train_end, strategy_class
            )
            split_config.strategy = best_strategy_cfg

        # ── Wire up and run ───────────────────────────────────────────────
        from collections import deque
        from engine.data_handler import DataHandler
        from engine.portfolio    import PortfolioManager
        from engine.execution    import ExecutionHandler
        from engine.backtest     import BacktestEngine

        queue     = deque()
        data      = DataHandler(split_config, queue)
        strategy  = strategy_class(split_config, queue)
        portfolio = PortfolioManager(split_config, queue)
        execution = ExecutionHandler(split_config, queue)
        engine    = BacktestEngine(split_config, data, strategy, portfolio, execution)

        try:
            results = engine.run()
        except Exception as exc:
            logger.error("Split %d failed: %s", split_index, exc)
            return {
                "split_index": split_index,
                "train_start": train_start,
                "train_end":   train_end,
                "test_start":  test_start,
                "test_end":    test_end,
                "error":       str(exc),
            }

        m = results["metrics"]
        m["split_index"] = split_index
        m["train_start"] = train_start
        m["train_end"]   = train_end
        m["test_start"]  = test_start
        m["test_end"]    = test_end
        return m

    # ──────────────────────────────────────────────────────────────────────
    # RUN ALL SPLITS
    # ──────────────────────────────────────────────────────────────────────

    def run_all_splits(self, strategy_class: Type[BaseStrategy]) -> List[dict]:
        """
        Run every walk-forward split sequentially and collect results.
        """
        if not self.splits:
            self.generate_splits()

        if not self.splits:
            raise RuntimeError("No walk-forward splits were generated.")

        self.split_results = []
        split_iter = tqdm(
            enumerate(self.splits, start=1),
            total=len(self.splits),
            desc="Walk-forward splits",
            unit="split",
            file=sys.stdout,
        )
        for i, (tr_s, tr_e, te_s, te_e) in split_iter:
            split_iter.set_postfix(test=f"{te_s[:4]}→{te_e[:4]}", refresh=False)
            result = self.run_single_split(tr_s, tr_e, te_s, te_e, strategy_class, i)
            self.split_results.append(result)

        logger.info(
            "All splits complete. %d out-of-sample results collected.",
            len(self.split_results),
        )
        return self.split_results

    # ──────────────────────────────────────────────────────────────────────
    # AGGREGATE
    # ──────────────────────────────────────────────────────────────────────

    def aggregate_results(self) -> dict:
        """
        Compute mean and standard deviation of each metric across all splits.
        """
        if not self.split_results:
            raise RuntimeError("No split results — run run_all_splits() first.")

        df = pd.DataFrame(self.split_results)
        numeric_cols = [
            "sharpe_ratio", "max_drawdown_pct", "cagr", "sortino_ratio",
            "calmar_ratio", "win_rate", "profit_factor", "total_trades",
            "total_return_pct",
        ]

        agg: dict = {"n_splits": len(self.split_results), "per_split": self.split_results}
        for col in numeric_cols:
            if col in df.columns:
                valid = df[col].dropna()
                agg[col] = {
                    "mean": round(float(valid.mean()), 4) if len(valid) > 0 else None,
                    "std":  round(float(valid.std()),  4) if len(valid) > 1 else None,
                }

        return agg

    # ──────────────────────────────────────────────────────────────────────
    # OPTIONAL: IN-SAMPLE OPTIMISATION
    # ──────────────────────────────────────────────────────────────────────

    def _optimize_on_train(
        self,
        train_start: str,
        train_end: str,
        strategy_class: Type[BaseStrategy],
    ) -> StrategyConfig:
        """
        Simple grid search over key strategy parameters on the training window.

        IMPORTANT: parameters are selected on IN-SAMPLE training data only,
        then applied to the subsequent OUT-OF-SAMPLE test window. This is the
        correct walk-forward procedure. Never use optimised params on train data
        for the Sharpe comparison — compare on the unseen test split only.

        Returns
        -------
        StrategyConfig — best-performing parameter set found on training data.
        """
        fast_windows = [10, 20, 30]
        slow_windows = [40, 50, 100]
        rsi_periods  = [10, 14, 20]

        best_sharpe = float("-inf")
        best_cfg    = copy.deepcopy(self._config.strategy)

        from collections import deque
        from engine.data_handler import DataHandler
        from engine.portfolio    import PortfolioManager
        from engine.execution    import ExecutionHandler
        from engine.backtest     import BacktestEngine

        for fast, slow, rsi in itertools.product(fast_windows, slow_windows, rsi_periods):
            if fast >= slow:
                continue

            trial_config = copy.deepcopy(self._config)
            trial_config.data.start_date          = train_start
            trial_config.data.end_date            = train_end
            trial_config.strategy.fast_ma_window  = fast
            trial_config.strategy.slow_ma_window  = slow
            trial_config.strategy.rsi_period      = rsi
            trial_config.verbose                  = False

            try:
                queue     = deque()
                data      = DataHandler(trial_config, queue)
                strategy  = strategy_class(trial_config, queue)
                portfolio = PortfolioManager(trial_config, queue)
                execution = ExecutionHandler(trial_config, queue)
                engine    = BacktestEngine(
                    trial_config, data, strategy, portfolio, execution
                )
                results = engine.run()
                sharpe  = results["metrics"].get("sharpe_ratio", float("-inf"))
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_cfg    = copy.deepcopy(trial_config.strategy)
            except Exception as exc:
                logger.debug(
                    "Grid search trial (fast=%d, slow=%d, rsi=%d) failed: %s",
                    fast, slow, rsi, exc,
                )

        logger.info(
            "Train optimisation complete. Best Sharpe=%.3f  "
            "fast=%d, slow=%d, rsi=%d",
            best_sharpe, best_cfg.fast_ma_window, best_cfg.slow_ma_window, best_cfg.rsi_period,
        )
        return best_cfg

    # ──────────────────────────────────────────────────────────────────────
    # REPORT TABLE
    # ──────────────────────────────────────────────────────────────────────

    def generate_walk_forward_report(self) -> pd.DataFrame:
        """
        Return a clean DataFrame of per-split and aggregate results.
        """
        if not self.split_results:
            return pd.DataFrame()

        cols = {
            "split_index":      "Split",
            "test_start":       "Test Start",
            "test_end":         "Test End",
            "cagr":             "CAGR",
            "sharpe_ratio":     "Sharpe",
            "max_drawdown_pct": "Max DD",
            "sortino_ratio":    "Sortino",
            "win_rate":         "Win Rate",
            "total_trades":     "# Trades",
        }

        df = pd.DataFrame(self.split_results)
        # Keep only columns that exist
        avail = {k: v for k, v in cols.items() if k in df.columns}
        df = df[list(avail.keys())].rename(columns=avail)

        # Aggregate row
        numeric_df = df.select_dtypes(include=[np.number])
        agg_row    = numeric_df.mean().to_dict()
        agg_row.update({"Split": "MEAN", "Test Start": "", "Test End": ""})
        agg_series = pd.Series(agg_row)
        df = pd.concat([df, agg_series.to_frame().T], ignore_index=True)

        return df
