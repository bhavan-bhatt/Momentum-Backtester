# strategies/cross_sectional_momentum.py
# ============================================================
# CROSS-SECTIONAL MOMENTUM (Jegadeesh-Titman style)
# Rank stocks by trailing return, go long top / short bottom fraction.
# ============================================================

import logging
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from config import BacktestConfig
from data.constituents import ConstituentTracker
from engine.events import MarketEvent, SignalDirection
from engine.strategy import BaseStrategy

logger = logging.getLogger(__name__)


class CrossSectionalMomentumStrategy(BaseStrategy):
    """
    Ranks the active universe by trailing return and trades the cross-section.

    Rebalances monthly or weekly. Unlike Phase 1 strategies, this evaluates
    all symbols simultaneously on rebalance dates only.
    """

    def __init__(
        self,
        config: BacktestConfig,
        event_queue: deque,
        constituent_tracker: Optional[ConstituentTracker] = None,
        regime_filter=None,
    ) -> None:
        super().__init__(config, event_queue)

        ccfg = config.advanced.cross_sectional
        self.lookback_months = ccfg.lookback_months
        self.skip_recent_days = ccfg.skip_recent_days
        self.rebalance_freq = ccfg.rebalance_freq
        self.top_decile_pct = ccfg.top_decile_pct
        self.bottom_decile_pct = ccfg.bottom_decile_pct
        self.allow_short = config.strategy.allow_short

        self.constituent_tracker = constituent_tracker
        self.lookback_days = self.lookback_months * 21
        self.required_bars = self.lookback_days + self.skip_recent_days + 5

        self.current_longs: Set[str] = set()
        self.current_shorts: Set[str] = set()
        self.last_rebalance_date: Optional[datetime] = None
        self._warned_small_universe = False

        self.strategy_id = f"CrossSectionalMomentum_{self.lookback_months}m"
        self.all_symbols = list(config.data.symbols)
        self.set_regime_filter(regime_filter, "CrossSectionalMomentum")

    def calculate_signals(self, event: MarketEvent, data_handler) -> None:
        """Run cross-sectional ranking and emit signals on rebalance dates."""
        if not self.all_symbols or event.symbol != self.all_symbols[0]:
            return

        if data_handler.get_bar_index() < self.required_bars:
            return

        if not self._is_rebalance_date(event.timestamp):
            return

        active_universe = self._get_active_universe(event.timestamp, data_handler)
        momentum_scores = self._compute_momentum_scores(
            active_universe, event.timestamp, data_handler
        )

        if len(momentum_scores) < 2:
            if not self._warned_small_universe:
                logger.warning(
                    "[%s] Universe too small to rank (%d symbols with valid scores). "
                    "Add more symbols or reduce lookback_months.",
                    self.strategy_id,
                    len(momentum_scores),
                )
                self._warned_small_universe = True
            return

        new_longs, new_shorts = self._rank_and_select(momentum_scores)
        held = self.current_longs | self.current_shorts
        target = new_longs | new_shorts

        for symbol in held - target:
            if symbol in self.current_longs:
                self._emit_signal(symbol, event.timestamp, SignalDirection.EXIT_LONG)
            elif symbol in self.current_shorts:
                self._emit_signal(symbol, event.timestamp, SignalDirection.EXIT_SHORT)

        for symbol in new_longs - self.current_longs:
            self._emit_signal(symbol, event.timestamp, SignalDirection.LONG)

        for symbol in new_shorts - self.current_shorts:
            self._emit_signal(symbol, event.timestamp, SignalDirection.SHORT)

        self.current_longs = new_longs
        self.current_shorts = new_shorts
        self.last_rebalance_date = event.timestamp

    def _is_rebalance_date(self, current_date: datetime) -> bool:
        """Return True if a rebalance should occur on this bar."""
        if self.last_rebalance_date is None:
            return True

        if self.rebalance_freq == "monthly":
            return (
                current_date.month != self.last_rebalance_date.month
                or current_date.year != self.last_rebalance_date.year
            )

        if self.rebalance_freq == "weekly":
            return (
                current_date.isocalendar()[1]
                != self.last_rebalance_date.isocalendar()[1]
            )

        raise ValueError(
            f"Unknown rebalance_freq '{self.rebalance_freq}'. Use 'monthly' or 'weekly'."
        )

    def _get_active_universe(
        self, date: datetime, data_handler
    ) -> List[str]:
        """Return symbols eligible for ranking on this rebalance."""
        if self.constituent_tracker is not None:
            candidates = list(self.constituent_tracker.get_active_universe(date))
        else:
            candidates = self.all_symbols

        eligible: List[str] = []
        for symbol in candidates:
            bars = data_handler.get_latest_bars(symbol, self.required_bars)
            if self._has_enough_bars(bars, self.required_bars):
                eligible.append(symbol)
        return eligible

    def _compute_momentum_scores(
        self,
        universe: List[str],
        date: datetime,
        data_handler,
    ) -> Dict[str, float]:
        """Compute formation-period return for each symbol in the universe."""
        scores: Dict[str, float] = {}

        for symbol in universe:
            bars = data_handler.get_latest_bars(symbol, self.required_bars)
            if not self._has_enough_bars(bars, self.required_bars):
                continue

            prices = bars["close"]
            end_idx = -1 - self.skip_recent_days
            start_idx = end_idx - self.lookback_days

            if abs(start_idx) > len(prices):
                continue

            p_end = prices.iloc[end_idx]
            p_start = prices.iloc[start_idx]

            if p_start <= 0 or pd.isna(p_start) or pd.isna(p_end):
                continue

            scores[symbol] = float((p_end / p_start) - 1.0)

        return scores

    def _rank_and_select(
        self, scores: Dict[str, float]
    ) -> Tuple[Set[str], Set[str]]:
        """Rank symbols by score and select top/bottom fractions."""
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        n = len(ranked)

        if n <= 8:
            n_long = max(1, n // 2)
            n_short = max(0, n // 3) if self.allow_short else 0
        else:
            n_long = max(1, int(n * self.top_decile_pct))
            n_short = int(n * self.bottom_decile_pct) if self.allow_short else 0

        longs = {symbol for symbol, _ in ranked[:n_long]}
        shorts = (
            {symbol for symbol, _ in ranked[-n_short:]}
            if n_short > 0
            else set()
        )

        if n_long + n_short >= n:
            logger.warning(
                "[%s] Long/short selection overlaps on small universe (n=%d). "
                "Reducing short count.",
                self.strategy_id,
                n,
            )
            overlap = longs & shorts
            shorts -= overlap
            if len(shorts) > 0 and n_long + len(shorts) >= n:
                n_short = max(0, n - n_long - 1)
                shorts = {symbol for symbol, _ in ranked[-n_short:]} if n_short else set()
                shorts -= longs

        return longs, shorts
