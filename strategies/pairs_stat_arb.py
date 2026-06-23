# strategies/pairs_stat_arb.py
# ============================================================
# COINTEGRATION-BASED PAIRS TRADING (STATISTICAL ARBITRAGE)
# ============================================================

import logging
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

from config import BacktestConfig
from engine.events import MarketEvent, SignalDirection
from engine.strategy import BaseStrategy

logger = logging.getLogger(__name__)


class PairsStatArbStrategy(BaseStrategy):
    """
    Trades the mean-reverting spread between two cointegrated stocks.

    Emits two signals per trade (one per leg). Requires allow_short=True.
    """

    def __init__(
        self,
        config: BacktestConfig,
        event_queue: deque,
        regime_filter=None,
    ) -> None:
        super().__init__(config, event_queue)

        pcfg = config.advanced.pairs
        self.lookback_window = pcfg.lookback_window
        self.cointegration_pvalue_threshold = pcfg.cointegration_pvalue_threshold
        self.zscore_entry = pcfg.zscore_entry
        self.zscore_exit = pcfg.zscore_exit
        self.zscore_stop = pcfg.zscore_stop
        self.recheck_cointegration_every = pcfg.recheck_cointegration_every
        self.candidate_pairs: List[Tuple[str, str]] = [
            tuple(pair) for pair in pcfg.candidate_pairs
        ]

        self.pair_state: Dict[Tuple[str, str], dict] = {}
        for pair in self.candidate_pairs:
            self.pair_state[pair] = {
                "is_cointegrated": False,
                "hedge_ratio": None,
                "spread_mean": None,
                "spread_std": None,
                "last_check_bar": -1,
                "position": None,
            }

        self._bar_counter = 0
        self.strategy_id = "PairsStatArb"
        self.set_regime_filter(regime_filter, "PairsStatArb")

        if not config.strategy.allow_short:
            logger.warning(
                "PairsStatArbStrategy requires allow_short=True. Short legs will not "
                "execute correctly — exclude from config.advanced.ensemble.enabled_strategies."
            )

    def calculate_signals(self, event: MarketEvent, data_handler) -> None:
        """Process each bar on symbol_b trigger for each candidate pair."""
        if self._config.data.symbols and event.symbol == self._config.data.symbols[0]:
            self._bar_counter += 1

        for symbol_a, symbol_b in self.candidate_pairs:
            if event.symbol != symbol_b:
                continue

            state = self.pair_state[(symbol_a, symbol_b)]
            due_for_recheck = (
                self._bar_counter - state["last_check_bar"]
                >= self.recheck_cointegration_every
            )

            if due_for_recheck or state["hedge_ratio"] is None:
                result = self._test_cointegration(
                    symbol_a, symbol_b, event.timestamp, data_handler
                )
                state.update(result)
                state["last_check_bar"] = self._bar_counter

            if not state["is_cointegrated"]:
                if state["position"] is not None:
                    self._close_pair(symbol_a, symbol_b, state, event.timestamp)
                continue

            z = self._compute_zscore(symbol_a, symbol_b, state, data_handler)
            if z is None:
                continue

            self._evaluate_pair_signals(
                symbol_a, symbol_b, z, state, event.timestamp
            )

    def _test_cointegration(
        self, symbol_a: str, symbol_b: str, date: datetime, data_handler
    ) -> dict:
        """Run Engle-Granger cointegration test and estimate hedge ratio via OLS."""
        bars_a = data_handler.get_latest_bars(symbol_a, self.lookback_window)
        bars_b = data_handler.get_latest_bars(symbol_b, self.lookback_window)

        if (
            bars_a is None
            or bars_b is None
            or len(bars_a) < self.lookback_window
            or len(bars_b) < self.lookback_window
        ):
            return {
                "is_cointegrated": False,
                "hedge_ratio": None,
                "spread_mean": None,
                "spread_std": None,
            }

        prices_a = bars_a["close"].values.astype(float)
        prices_b = bars_b["close"].values.astype(float)

        _, pvalue, _ = coint(prices_a, prices_b)
        is_cointegrated = pvalue < self.cointegration_pvalue_threshold

        hedge_ratio = spread_mean = spread_std = None
        if is_cointegrated:
            x = sm.add_constant(prices_b)
            model = sm.OLS(prices_a, x).fit()
            hedge_ratio = float(model.params[1])
            spread = prices_a - hedge_ratio * prices_b
            spread_mean = float(spread.mean())
            spread_std = float(spread.std())

        logger.info(
            "Pair (%s,%s) cointegration p=%.4f → %s",
            symbol_a,
            symbol_b,
            pvalue,
            "ACTIVE" if is_cointegrated else "INACTIVE",
        )

        return {
            "is_cointegrated": is_cointegrated,
            "hedge_ratio": hedge_ratio,
            "spread_mean": spread_mean,
            "spread_std": spread_std,
        }

    def _compute_zscore(
        self, symbol_a: str, symbol_b: str, state: dict, data_handler
    ) -> Optional[float]:
        """Compute current spread z-score using fixed mean/std from last coint check."""
        price_a = data_handler.get_current_price(symbol_a)
        price_b = data_handler.get_current_price(symbol_b)
        if price_a is None or price_b is None:
            return None

        spread_std = state["spread_std"]
        if state["hedge_ratio"] is None or spread_std is None or spread_std == 0:
            return None

        current_spread = price_a - state["hedge_ratio"] * price_b
        return float((current_spread - state["spread_mean"]) / spread_std)

    def _signal_strength(self, z: float) -> float:
        return min(1.0, abs(z) / self.zscore_stop)

    def _close_pair(
        self,
        symbol_a: str,
        symbol_b: str,
        state: dict,
        timestamp: datetime,
    ) -> None:
        """Emit exit signals for both legs and clear position state."""
        if state["position"] == "short_spread":
            self._emit_signal(symbol_a, timestamp, SignalDirection.EXIT_SHORT)
            self._emit_signal(symbol_b, timestamp, SignalDirection.EXIT_LONG)
        elif state["position"] == "long_spread":
            self._emit_signal(symbol_a, timestamp, SignalDirection.EXIT_LONG)
            self._emit_signal(symbol_b, timestamp, SignalDirection.EXIT_SHORT)
        state["position"] = None

    def _evaluate_pair_signals(
        self,
        symbol_a: str,
        symbol_b: str,
        z: float,
        state: dict,
        timestamp: datetime,
    ) -> None:
        """Apply entry/exit/stop rules and emit signals for both legs."""
        position = state["position"]

        if position is None:
            strength = self._signal_strength(z)
            if z > self.zscore_entry:
                self._emit_signal(
                    symbol_a, timestamp, SignalDirection.SHORT, strength
                )
                self._emit_signal(
                    symbol_b, timestamp, SignalDirection.LONG, strength
                )
                state["position"] = "short_spread"
            elif z < -self.zscore_entry:
                self._emit_signal(
                    symbol_a, timestamp, SignalDirection.LONG, strength
                )
                self._emit_signal(
                    symbol_b, timestamp, SignalDirection.SHORT, strength
                )
                state["position"] = "long_spread"
            return

        if position == "short_spread":
            if abs(z) < self.zscore_exit or abs(z) > self.zscore_stop:
                self._close_pair(symbol_a, symbol_b, state, timestamp)
                if abs(z) > self.zscore_stop:
                    logger.warning(
                        "STOP-OUT: pair (%s,%s) diverged beyond zscore_stop — "
                        "possible cointegration breakdown.",
                        symbol_a,
                        symbol_b,
                    )
        elif position == "long_spread":
            if abs(z) < self.zscore_exit or abs(z) > self.zscore_stop:
                self._close_pair(symbol_a, symbol_b, state, timestamp)
                if abs(z) > self.zscore_stop:
                    logger.warning(
                        "STOP-OUT: pair (%s,%s) diverged beyond zscore_stop — "
                        "possible cointegration breakdown.",
                        symbol_a,
                        symbol_b,
                    )
