# engine/data_handler.py
# ============================================================
# LOADS CSV DATA AND SERVES BARS TO THE EVENT LOOP ONE AT A TIME
# Handles missing columns, date formats, and gaps gracefully.
# ============================================================

import os
import logging
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from engine.events import MarketEvent
from config import BacktestConfig

logger = logging.getLogger(__name__)

# Canonical column name variants accepted from CSV files
_COLUMN_ALIASES: Dict[str, set] = {
    "date":      {"date", "timestamp", "time", "datetime"},
    "open":      {"open", "open_price", "o"},
    "high":      {"high", "high_price", "h"},
    "low":       {"low", "low_price", "l"},
    "close":     {"close", "close_price", "last", "c"},
    "adj_close": {"adj close", "adj_close", "adjusted close", "adjclose", "adjusted_close"},
    "volume":    {"volume", "vol", "v", "turnover"},
}

_DATE_FORMATS = [
    "%Y-%m-%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%Y/%m/%d",
    "%d-%b-%Y",
    "%d/%b/%Y",
]

_OHLCV_COLS = ["open", "high", "low", "close", "volume"]


class DataHandler:
    """
    Manages all historical market data for the backtest.

    Responsibilities
    ----------------
    1. Load CSV files for each symbol.
    2. Normalise column names to a canonical schema.
    3. Align all symbols to a common date range (inner join on dates).
    4. Forward-fill small gaps up to config.data.max_gap_fill consecutive bars.
    5. Serve bars one at a time via update_bars().
    6. Provide historical bar access for indicator computation via get_latest_bars().

    Graceful Degradation
    --------------------
    - Missing adj_close → use close.
    - Missing high/low  → has_high_low[symbol] = False; ATR unavailable.
    - Missing volume    → has_volume[symbol] = False.
    - Missing file      → symbol removed from active list; warning logged.
    """

    def __init__(self, config: BacktestConfig, event_queue: deque) -> None:
        self._config = config
        self._event_queue = event_queue

        self.symbol_data: Dict[str, pd.DataFrame] = {}
        self.benchmark_data: Optional[pd.DataFrame] = None
        self.secondary_benchmark_data: Optional[pd.DataFrame] = None
        self.has_high_low: Dict[str, bool] = {}
        self.has_volume: Dict[str, bool] = {}

        self._bar_index: int = 0
        self._all_dates: List[datetime] = []

        # ── Load trading symbols ─────────────────────────────────────────
        for symbol in config.data.symbols:
            df = self._load_and_normalise(symbol)
            if df is not None:
                self.symbol_data[symbol] = df
            else:
                logger.warning("Symbol %s skipped — file not found or unreadable.", symbol)

        if not self.symbol_data:
            raise RuntimeError(
                "DataHandler: Zero symbols loaded successfully. "
                "Check csv_dir and symbol names in config."
            )

        # Drop symbols listed after backtest start (inner join would shrink the window).
        start = pd.Timestamp(self._config.data.start_date)
        min_first_date = start + pd.Timedelta(days=5)
        dropped: List[str] = []
        for symbol in list(self.symbol_data.keys()):
            first_bar = self.symbol_data[symbol].index.min()
            if first_bar > min_first_date:
                dropped.append(symbol)
                del self.symbol_data[symbol]
                self.has_high_low.pop(symbol, None)
                self.has_volume.pop(symbol, None)
        if dropped:
            logger.warning(
                "Dropped %d symbol(s) with no history from %s: %s",
                len(dropped),
                self._config.data.start_date,
                ", ".join(dropped[:8]) + ("..." if len(dropped) > 8 else ""),
            )
        if not self.symbol_data:
            raise RuntimeError(
                "DataHandler: No symbols remain after history filter. "
                "Use older/larger-cap names or an earlier start_date."
            )

        # ── Load benchmark ───────────────────────────────────────────────
        bench = self._load_and_normalise(config.data.benchmark_symbol)
        if bench is None:
            logger.warning(
                "Benchmark %s not loaded — alpha/beta metrics unavailable.",
                config.data.benchmark_symbol,
            )
        self.benchmark_data = bench

        sec_sym = getattr(config.data, "secondary_benchmark_symbol", None)
        if sec_sym:
            sec_bench = self._load_and_normalise(sec_sym)
            if sec_bench is None:
                logger.warning(
                    "Secondary benchmark %s not loaded — Nifty 500 comparison unavailable.",
                    sec_sym,
                )
            self.secondary_benchmark_data = sec_bench

        # ── Align to common dates and trim to config date range ──────────
        self._align_dates()

        n_symbols = len(self.symbol_data)
        n_bars = len(self._all_dates)
        logger.info("Loaded %d symbol(s) × %d trading days.", n_symbols, n_bars)
        if config.verbose:
            print(
                f"[DataHandler] Loaded {n_symbols} symbol(s) × {n_bars} trading days "
                f"({self._all_dates[0].date()} → {self._all_dates[-1].date()})"
            )

    # ──────────────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ──────────────────────────────────────────────────────────────────────

    def _load_and_normalise(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load a CSV file and normalise it to the canonical OHLCV schema."""
        path = os.path.join(self._config.data.csv_dir, f"{symbol}.csv")
        if not os.path.exists(path):
            logger.warning("CSV not found: %s", path)
            return None

        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as exc:
            logger.error("Failed to read %s: %s", path, exc)
            return None

        df = self._normalise_column_names(df)
        df = self._parse_dates(df)

        # Set datetime index, sort ascending
        df = df.set_index("date").sort_index()
        df.index.name = "date"

        # Remove duplicate dates (keep last)
        df = df[~df.index.duplicated(keep="last")]

        # ── Price column selection ───────────────────────────────────────
        price_col = self._config.data.price_column
        if price_col == "auto":
            if "adj_close" in df.columns:
                df["close"] = pd.to_numeric(df["adj_close"], errors="coerce")
                logger.debug("%s: using adj_close as price.", symbol)
            else:
                logger.debug("%s: adj_close absent — using raw close.", symbol)
        elif price_col == "close":
            pass  # already present
        elif price_col == "adj_close":
            if "adj_close" not in df.columns:
                raise ValueError(
                    f"price_column='adj_close' configured but adj_close not found in {path}"
                )
            df["close"] = pd.to_numeric(df["adj_close"], errors="coerce")

        # ── Coerce numeric columns ───────────────────────────────────────
        for col in _OHLCV_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Ensure all canonical columns exist (fill missing with NaN)
        for col in _OHLCV_COLS:
            if col not in df.columns:
                df[col] = np.nan

        if df["close"].isna().all():
            logger.error("%s: close column is entirely NaN after normalisation.", symbol)
            return None

        # ── Per-symbol availability flags ────────────────────────────────
        self.has_high_low[symbol] = (
            not df["high"].isna().all() and not df["low"].isna().all()
        )
        self.has_volume[symbol] = not df["volume"].isna().all()

        # ── Forward-fill gaps up to max_gap_fill consecutive NaNs ────────
        max_fill = self._config.data.max_gap_fill
        df = df.fillna(method="ffill", limit=max_fill)

        # Drop rows where close is still NaN (unfillable gaps)
        df = df.dropna(subset=["close"])

        return df[_OHLCV_COLS]

    def _normalise_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns from various CSV formats to the canonical schema."""
        # Lowercase + strip whitespace
        df.columns = [c.lower().strip() for c in df.columns]

        rename_map: Dict[str, str] = {}
        for canonical, aliases in _COLUMN_ALIASES.items():
            for col in df.columns:
                if col in aliases and col not in rename_map:
                    rename_map[col] = canonical

        return df.rename(columns=rename_map)

    def _parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert the date column to datetime64, trying multiple formats."""
        # If date column is the index (CSV was loaded with index_col), bring it back
        if "date" not in df.columns and df.index.name == "date":
            df = df.reset_index()
        elif "date" not in df.columns:
            # Try to find a date-like column in the index
            df = df.reset_index()
            df = self._normalise_column_names(df)

        if "date" not in df.columns:
            raise ValueError("DataHandler: Cannot identify a date column in CSV.")

        raw = df["date"]

        # Fast path: pandas auto-infer
        try:
            parsed = pd.to_datetime(raw, utc=False)
            df["date"] = parsed.dt.tz_localize(None) if parsed.dt.tz is not None else parsed
            return df
        except Exception:
            pass

        # Slow path: try each format
        for fmt in _DATE_FORMATS:
            try:
                parsed = pd.to_datetime(raw, format=fmt)
                df["date"] = parsed
                logger.debug("Date parsed with format: %s", fmt)
                return df
            except Exception:
                continue

        raise ValueError(
            f"DataHandler: None of the date formats {_DATE_FORMATS} matched "
            f"the date column. Sample values: {raw.head(3).tolist()}"
        )

    def _align_dates(self) -> None:
        """
        Compute the common set of trading dates across all symbols, trimmed to
        [start_date, end_date]. Reindex each DataFrame to this common set and
        forward-fill any gaps introduced by alignment.
        """
        start = pd.Timestamp(self._config.data.start_date)
        end   = pd.Timestamp(self._config.data.end_date)

        # Start with the union of all dates so we don't lose any trading day,
        # then intersect ensures every symbol has a quote on each common date.
        date_sets = [set(df.index) for df in self.symbol_data.values()]
        if not date_sets:
            self._all_dates = []
            return

        common_dates = date_sets[0].intersection(*date_sets[1:]) if len(date_sets) > 1 else date_sets[0]

        # Optionally also intersect with benchmark dates
        if self.benchmark_data is not None:
            bench_dates = set(self.benchmark_data.index)
            # Don't discard trading dates if benchmark data has gaps; only filter
            # dates that fall completely outside the benchmark range.
            bench_min = min(bench_dates)
            bench_max = max(bench_dates)
            common_dates = {d for d in common_dates if bench_min <= d <= bench_max}

        # Trim to configured date range
        common_dates = {d for d in common_dates if start <= d <= end}
        sorted_dates = sorted(common_dates)

        if not sorted_dates:
            raise RuntimeError(
                "DataHandler: No common trading dates found after alignment. "
                f"Check start_date ({self._config.data.start_date}) / end_date ({self._config.data.end_date}) "
                "and that CSV files actually contain data in this range."
            )

        self._all_dates = sorted_dates

        # Reindex every symbol to the common date set; ffill gaps
        dt_index = pd.DatetimeIndex(sorted_dates)
        max_fill = self._config.data.max_gap_fill

        for symbol, df in self.symbol_data.items():
            reindexed = df.reindex(dt_index)
            reindexed = reindexed.fillna(method="ffill", limit=max_fill)
            # Any remaining NaN in close after reindex → forward-fill without limit
            # (halt scenario — keep last valid price)
            reindexed["close"] = reindexed["close"].fillna(method="ffill")
            self.symbol_data[symbol] = reindexed

        # Also trim benchmark to the same date range
        if self.benchmark_data is not None:
            self.benchmark_data = self.benchmark_data.reindex(dt_index).fillna(method="ffill")

        if self.secondary_benchmark_data is not None:
            self.secondary_benchmark_data = (
                self.secondary_benchmark_data.reindex(dt_index).fillna(method="ffill")
            )

        logger.info(
            "Aligned to %d common trading days: %s → %s",
            len(sorted_dates),
            sorted_dates[0].date(),
            sorted_dates[-1].date(),
        )

    # ──────────────────────────────────────────────────────────────────────
    # PUBLIC INTERFACE — EVENT LOOP
    # ──────────────────────────────────────────────────────────────────────

    def update_bars(self) -> None:
        """
        Advance the bar pointer by one step and push MarketEvents for every
        active symbol into the event queue.
        """
        if self._bar_index >= len(self._all_dates):
            return

        current_date = self._all_dates[self._bar_index]
        is_last = self._bar_index == len(self._all_dates) - 1

        for symbol, df in self.symbol_data.items():
            row = df.loc[current_date]

            def _val(col: str) -> Optional[float]:
                v = row[col]
                return None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)

            event = MarketEvent(
                timestamp=current_date,
                symbol=symbol,
                open=_val("open"),
                high=_val("high"),
                low=_val("low"),
                close=float(row["close"]),
                volume=_val("volume"),
                is_last_bar=is_last,
            )
            self._event_queue.append(event)

        self._bar_index += 1

    def has_more_bars(self) -> bool:
        """True if there are more bars remaining in the simulation."""
        return self._bar_index < len(self._all_dates)

    def get_total_bars(self) -> int:
        """Total number of aligned trading days in the simulation."""
        return len(self._all_dates)

    def get_bar_index(self) -> int:
        """Current bar pointer (0 before first update_bars(), then 1…total)."""
        return self._bar_index

    # ──────────────────────────────────────────────────────────────────────
    # PUBLIC INTERFACE — DATA ACCESS
    # ──────────────────────────────────────────────────────────────────────

    def get_latest_bars(self, symbol: str, N: int = 1) -> Optional[pd.DataFrame]:
        """
        Return the last N bars for a symbol up to and including the current bar.

        CRITICAL: Never looks ahead. Bars at indices 0 … (_bar_index - 1) only.
        """
        if symbol not in self.symbol_data:
            logger.warning("get_latest_bars: unknown symbol '%s'.", symbol)
            return None

        current_idx = self._bar_index  # update_bars already incremented this
        end_idx   = current_idx        # exclusive upper bound for iloc
        start_idx = max(0, current_idx - N)

        df = self.symbol_data[symbol]
        return df.iloc[start_idx:end_idx]

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Return the close price of the current bar for mark-to-market."""
        bars = self.get_latest_bars(symbol, 1)
        if bars is None or bars.empty:
            return None
        return float(bars["close"].iloc[-1])

    def get_next_open(self, symbol: str) -> Optional[float]:
        """
        Return the open price of the NEXT bar.
        This is the single intentional look-ahead — used to fill orders at
        next-day open price after the signal bar has closed.
        """
        next_idx = self._bar_index  # already points at next bar after update_bars()
        df = self.symbol_data.get(symbol)
        if df is None:
            return None

        if next_idx >= len(df):
            # End of data — use last known close
            last_close = df["close"].iloc[-1]
            return float(last_close) if not np.isnan(last_close) else None

        open_price = df["open"].iloc[next_idx]
        if np.isnan(open_price):
            close_now = df["close"].iloc[next_idx - 1] if next_idx > 0 else df["close"].iloc[0]
            return float(close_now)

        return float(open_price)

    def get_symbols(self) -> List[str]:
        """Return successfully loaded symbol list (excludes benchmark)."""
        return list(self.symbol_data.keys())

    def get_current_datetime(self) -> Optional[datetime]:
        """Return the datetime of the bar most recently served."""
        if self._bar_index == 0:
            return None
        return self._all_dates[self._bar_index - 1]

    def get_benchmark_bars(self, N: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Return historical bars for the benchmark up to and including current bar.
        N=None → all bars up to current bar.
        """
        if self.benchmark_data is None:
            return None

        current_idx = self._bar_index
        if N is None:
            return self.benchmark_data.iloc[:current_idx]
        start_idx = max(0, current_idx - N)
        return self.benchmark_data.iloc[start_idx:current_idx]

    def get_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Return a deep copy of the entire dataset for all symbols.
        Used by WalkForwardEngine for train/test splitting.
        """
        return {sym: df.copy() for sym, df in self.symbol_data.items()}
