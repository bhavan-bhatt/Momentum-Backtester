# data/constituents.py
# ============================================================
# POINT-IN-TIME NIFTY 50 MEMBERSHIP
# Answers: "was this symbol actually in the index on this date?"
# ============================================================

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from config import BacktestConfig

logger = logging.getLogger(__name__)


class ConstituentTracker:
    """
    Tracks which symbols were members of the Nifty 50 on any given date.

    If the membership file is missing, all symbols in config.data.symbols are
    treated as always tradable (Phase 1 behaviour) with a clear warning logged.
    """

    def __init__(self, config: BacktestConfig) -> None:
        self._config = config
        self.membership: Dict[str, List[Tuple[datetime, Optional[datetime]]]] = {}
        self.is_available: bool = False

        file_path = config.advanced.constituents.membership_file
        enforce = config.advanced.constituents.enforce_point_in_time

        if not enforce or not os.path.exists(file_path):
            self.is_available = False
            logger.warning(
                "Point-in-time constituents not available. All symbols treated as "
                "always tradable. Survivorship bias is NOT corrected in this run — "
                "disclose this in the report."
            )
            return

        df = pd.read_csv(file_path)
        required_cols = {"symbol", "start_date", "end_date"}
        if not required_cols.issubset(df.columns):
            logger.warning(
                "Membership file %s missing required columns %s — degraded mode.",
                file_path,
                required_cols - set(df.columns),
            )
            return

        df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
        df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")

        interval_count = 0
        for symbol, group in df.groupby("symbol"):
            intervals: List[Tuple[datetime, Optional[datetime]]] = []
            for _, row in group.iterrows():
                start = row["start_date"]
                if pd.isna(start):
                    continue
                start_dt = pd.Timestamp(start).to_pydatetime()
                end_dt: Optional[datetime] = None
                if not pd.isna(row["end_date"]):
                    end_dt = pd.Timestamp(row["end_date"]).to_pydatetime()
                intervals.append((start_dt, end_dt))
                interval_count += 1
            if intervals:
                self.membership[str(symbol)] = intervals

        self.is_available = True
        logger.info(
            "Loaded point-in-time membership for %d symbols, %d intervals.",
            len(self.membership),
            interval_count,
        )

    @staticmethod
    def _normalize_date(date: datetime) -> datetime:
        """Strip time component for consistent interval comparisons."""
        if hasattr(date, "to_pydatetime"):
            date = date.to_pydatetime()
        return datetime(date.year, date.month, date.day)

    def is_member_on_date(self, symbol: str, date: datetime) -> bool:
        """Return True if symbol was a Nifty 50 member on the given date."""
        if not self.is_available:
            return True

        intervals = self.membership.get(symbol)
        if not intervals:
            return False

        check = self._normalize_date(date)
        for start, end in intervals:
            start_n = self._normalize_date(start)
            if check < start_n:
                continue
            if end is None or check <= self._normalize_date(end):
                return True
        return False

    def get_active_universe(self, date: datetime) -> Set[str]:
        """Return tradable symbols from config.data.symbols on the given date."""
        if not self.is_available:
            return set(self._config.data.symbols)

        return {
            symbol
            for symbol in self._config.data.symbols
            if self.is_member_on_date(symbol, date)
        }

    def get_membership_changes(
        self, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """Return a chronological log of index joins/exits within a date range."""
        if not self.is_available:
            return pd.DataFrame({
                "date": [],
                "symbol": [],
                "event": [],
                "note": ["Point-in-time membership data was not available."],
            })

        start_n = self._normalize_date(start_date)
        end_n = self._normalize_date(end_date)
        rows: List[dict] = []

        for symbol, intervals in self.membership.items():
            for interval_start, interval_end in intervals:
                join = self._normalize_date(interval_start)
                if start_n <= join <= end_n:
                    rows.append({
                        "date": join,
                        "symbol": symbol,
                        "event": "JOINED",
                    })
                if interval_end is not None:
                    exit_dt = self._normalize_date(interval_end)
                    if start_n <= exit_dt <= end_n:
                        rows.append({
                            "date": exit_dt,
                            "symbol": symbol,
                            "event": "EXITED",
                        })

        if not rows:
            return pd.DataFrame(columns=["date", "symbol", "event"])

        df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        return df

    def build_sample_membership_csv(self, output_path: str) -> None:
        """Scaffold a starter membership CSV for manual correction."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        df = pd.DataFrame({
            "symbol": self._config.data.symbols,
            "start_date": self._config.data.start_date,
            "end_date": "",
        })
        df.to_csv(output_path, index=False)
        logger.info(
            "Sample membership file created at %s. Edit start_date/end_date per "
            "symbol using NSE index reconstitution history before setting "
            "enforce_point_in_time=True for accurate results.",
            output_path,
        )
