# performance/walk_forward_v2.py
# ============================================================
# WALK-FORWARD V2: ANCHORED + ROLLING, PARAMETER STABILITY
# ============================================================

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Type

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from config import BacktestConfig
from engine.strategy import BaseStrategy
from performance.walk_forward import WalkForwardEngine

logger = logging.getLogger(__name__)


class WalkForwardEngineV2(WalkForwardEngine):
    """
    Extends Phase 1 walk-forward with anchored splits and parameter
    stability reporting.
    """

    def generate_splits_anchored(self) -> List[Tuple[str, str, str, str]]:
        """Generate splits with training window anchored at dataset start."""
        wf = self._config.walk_forward
        data = self._config.data

        data_start = datetime.strptime(data.start_date, "%Y-%m-%d")
        data_end = datetime.strptime(data.end_date, "%Y-%m-%d")
        data_start_str = data.start_date

        first_train_end = (
            data_start + relativedelta(years=wf.train_years) - timedelta(days=1)
        )
        current_test_start = first_train_end + timedelta(days=1)

        splits: List[Tuple[str, str, str, str]] = []
        while True:
            test_end = (
                current_test_start
                + relativedelta(years=wf.test_years)
                - timedelta(days=1)
            )
            if test_end > data_end:
                break

            train_end = current_test_start - timedelta(days=1)
            splits.append((
                data_start_str,
                train_end.strftime("%Y-%m-%d"),
                current_test_start.strftime("%Y-%m-%d"),
                test_end.strftime("%Y-%m-%d"),
            ))
            current_test_start += relativedelta(years=wf.step_years)

        logger.info(
            "Generated %d ANCHORED walk-forward splits (train window grows each "
            "iteration, always starting from %s).",
            len(splits),
            data_start_str,
        )
        self.splits = splits
        return splits

    def run_both_variants(
        self, strategy_class: Type[BaseStrategy]
    ) -> Dict[str, list]:
        """Run rolling (Phase 1) and anchored walk-forward variants."""
        self.generate_splits()
        rolling_results = self.run_all_splits(strategy_class)

        self.split_results = []
        self.splits = self.generate_splits_anchored()
        anchored_results = []
        for i, (tr_s, tr_e, te_s, te_e) in enumerate(self.splits, start=1):
            result = self.run_single_split(
                tr_s, tr_e, te_s, te_e, strategy_class, i
            )
            anchored_results.append(result)

        return {"rolling": rolling_results, "anchored": anchored_results}

    def run_parameter_stability_analysis(
        self, strategy_class: Type[BaseStrategy]
    ) -> pd.DataFrame:
        """Record best in-sample parameters chosen on each split."""
        if not self.splits:
            self.generate_splits()

        rows = []
        for split_index, (train_start, train_end, test_start, test_end) in enumerate(
            self.splits
        ):
            best_config = self._optimize_on_train(
                train_start, train_end, strategy_class
            )
            rows.append({
                "split_index": split_index,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "best_fast_ma_window": best_config.fast_ma_window,
                "best_slow_ma_window": best_config.slow_ma_window,
                "best_rsi_period": best_config.rsi_period,
            })

        df = pd.DataFrame(rows)
        for col in ("best_fast_ma_window", "best_slow_ma_window", "best_rsi_period"):
            if col in df.columns and df[col].std() != 0 and df[col].mean() != 0:
                cv = df[col].std() / df[col].mean()
                logger.info(
                    "Parameter stability — %s CV: %.2f (lower = more stable). "
                    "Values across splits: %s",
                    col,
                    cv,
                    df[col].tolist(),
                )
        return df

    def generate_stability_report(self, stability_df: pd.DataFrame) -> dict:
        """Summarise parameter stability with verdicts per parameter."""
        if stability_df.empty:
            return {}

        skip = {"split_index", "train_start", "train_end", "test_start", "test_end"}
        report = {}
        for col in stability_df.select_dtypes(include=[np.number]).columns:
            if col in skip:
                continue
            mean = float(stability_df[col].mean())
            std = float(stability_df[col].std())
            cv = std / mean if mean != 0 else float("inf")
            if cv < 0.25:
                verdict = "STABLE"
            elif cv < 0.5:
                verdict = "MODERATE"
            else:
                verdict = "UNSTABLE"
            report[col] = {
                "mean": round(mean, 4),
                "std": round(std, 4),
                "cv": round(cv, 4) if cv != float("inf") else None,
                "verdict": verdict,
                "values_by_split": stability_df[col].tolist(),
            }
        return report
