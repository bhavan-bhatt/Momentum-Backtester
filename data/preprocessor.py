# data/preprocessor.py
# ============================================================
# CSV NORMALISATION UTILITY (PRE-RUN)
#
# Run this once before backtesting to normalise raw CSV files
# from any source (yfinance, NSEpy, Kite, etc.) into the
# canonical format that DataHandler expects.
#
# Output: cleaned CSVs written to the same directory (or a new
# one) with standardised column names and date formats.
# ============================================================

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Accepted date formats (tried in order)
_DATE_FORMATS = [
    "%Y-%m-%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%Y/%m/%d",
    "%d-%b-%Y",
    "%d/%b/%Y",
    "%Y%m%d",
]

# Column alias map (lowercase → canonical)
_ALIASES: dict = {
    "date":      {"date", "timestamp", "time", "datetime", "index"},
    "open":      {"open", "open_price", "o"},
    "high":      {"high", "high_price", "h"},
    "low":       {"low", "low_price", "l"},
    "close":     {"close", "close_price", "last", "c"},
    "adj_close": {"adj close", "adj_close", "adjusted close", "adjclose", "adjusted_close"},
    "volume":    {"volume", "vol", "v", "turnover"},
}

_CANONICAL_ORDER = ["date", "open", "high", "low", "close", "adj_close", "volume"]


class CSVPreprocessor:
    """
    Normalises raw market data CSVs into the canonical schema.

    Features
    --------
    - Auto-detects column names from common variants.
    - Auto-detects date formats.
    - Removes duplicated dates.
    - Forward-fills small gaps (configurable).
    - Validates OHLC consistency (H >= max(O,C), L <= min(O,C)).
    - Adds adj_close column equal to close if missing (for Tier-1 compatibility).
    - Writes normalised CSV next to original (or to output_dir).

    Usage
    -----
    preprocessor = CSVPreprocessor(csv_dir="csv", output_dir="csv/normalised")
    preprocessor.process_all()
    """

    def __init__(
        self,
        csv_dir: str = "csv",
        output_dir: Optional[str] = None,
        max_gap_fill: int = 5,
        fix_ohlc: bool = True,
        verbose: bool = True,
    ) -> None:
        self._csv_dir     = Path(csv_dir)
        self._output_dir  = Path(output_dir) if output_dir else self._csv_dir
        self._max_gap     = max_gap_fill
        self._fix_ohlc    = fix_ohlc
        self._verbose     = verbose

    def process_all(self, pattern: str = "*.csv") -> List[str]:
        """
        Normalise all CSV files matching pattern in csv_dir.

        Returns
        -------
        List[str] — paths to successfully written output files.
        """
        files = list(self._csv_dir.glob(pattern))
        if not files:
            logger.warning("No CSV files found in %s matching '%s'.", self._csv_dir, pattern)
            return []

        self._output_dir.mkdir(parents=True, exist_ok=True)
        results = []

        for f in files:
            try:
                out = self.process_file(f)
                if out:
                    results.append(out)
            except Exception as exc:
                logger.error("Failed to preprocess %s: %s", f.name, exc)

        if self._verbose:
            print(f"\n[Preprocessor] Processed {len(results)}/{len(files)} files successfully.")
        return results

    def process_file(self, path: Path) -> Optional[str]:
        """
        Normalise a single CSV file.

        Parameters
        ----------
        path : Path — path to the input CSV.

        Returns
        -------
        str — output path, or None on failure.
        """
        if self._verbose:
            print(f"  Processing: {path.name} ...", end=" ", flush=True)

        df = pd.read_csv(path, low_memory=False)

        # ── Normalise columns ──────────────────────────────────────────
        df = self._normalise_columns(df)

        # ── Parse dates ────────────────────────────────────────────────
        df = self._parse_dates(df)
        df = df.set_index("date").sort_index()
        df = df[~df.index.duplicated(keep="last")]

        # ── Coerce numerics ────────────────────────────────────────────
        for col in ["open", "high", "low", "close", "adj_close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # ── Ensure adj_close exists ────────────────────────────────────
        if "adj_close" not in df.columns:
            df["adj_close"] = df["close"]

        # ── Forward-fill small gaps ────────────────────────────────────
        df = df.fillna(method="ffill", limit=self._max_gap)

        # ── Validate and optionally fix OHLC relationships ─────────────
        if self._fix_ohlc and all(c in df.columns for c in ["open", "high", "low", "close"]):
            df = self._repair_ohlc(df)

        # ── Ensure canonical column order ──────────────────────────────
        existing = [c for c in _CANONICAL_ORDER if c in df.columns]
        df = df[existing]

        # ── Report ─────────────────────────────────────────────────────
        n_rows = len(df)
        n_nan  = df["close"].isna().sum() if "close" in df.columns else 0

        # ── Write output ───────────────────────────────────────────────
        out_path = self._output_dir / path.name
        df.to_csv(out_path)

        if self._verbose:
            print(f"OK  ({n_rows} bars, {n_nan} NaN closes)")

        logger.info("Preprocessed %s → %s (%d bars)", path.name, out_path, n_rows)
        return str(out_path)

    # ──────────────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ──────────────────────────────────────────────────────────────────────

    def _normalise_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns to canonical schema."""
        df.columns = [c.lower().strip() for c in df.columns]
        rename_map = {}
        for canonical, aliases in _ALIASES.items():
            for col in df.columns:
                if col in aliases and canonical not in rename_map.values():
                    rename_map[col] = canonical
        return df.rename(columns=rename_map)

    def _parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse the date column into datetime64."""
        if "date" not in df.columns:
            # Try to use the index
            df = df.reset_index()
            df = self._normalise_columns(df)

        if "date" not in df.columns:
            raise ValueError("Cannot identify a date column.")

        raw = df["date"]

        # Fast path
        try:
            parsed = pd.to_datetime(raw)
            df["date"] = parsed.dt.tz_localize(None) if parsed.dt.tz is not None else parsed
            return df
        except Exception:
            pass

        # Slow path
        for fmt in _DATE_FORMATS:
            try:
                df["date"] = pd.to_datetime(raw, format=fmt)
                return df
            except Exception:
                continue

        raise ValueError(f"No date format matched. Sample: {raw.head(3).tolist()}")

    def _repair_ohlc(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Repair OHLC consistency violations:
          - High < max(Open, Close) → High = max(Open, Close)
          - Low  > min(Open, Close) → Low  = min(Open, Close)
        """
        for col in ["open", "high", "low", "close"]:
            if col not in df.columns:
                return df

        true_high = df[["open", "close"]].max(axis=1)
        true_low  = df[["open", "close"]].min(axis=1)

        violations_h = (df["high"] < true_high).sum()
        violations_l = (df["low"]  > true_low).sum()

        if violations_h > 0 or violations_l > 0:
            logger.debug(
                "Repaired %d High violations and %d Low violations.",
                violations_h, violations_l,
            )

        df["high"] = df[["high", "open", "close"]].max(axis=1)
        df["low"]  = df[["low",  "open", "close"]].min(axis=1)
        return df


def run_preprocessor_cli() -> None:
    """
    Command-line entry point for preprocessing.

    Usage:
        python -m data.preprocessor --csv_dir csv --output_dir csv/normalised
    """
    import argparse
    parser = argparse.ArgumentParser(
        description="Normalise raw market CSV files for the backtester."
    )
    parser.add_argument("--csv_dir",    default="csv",            help="Input CSV directory.")
    parser.add_argument("--output_dir", default=None,             help="Output directory (default: same as input).")
    parser.add_argument("--max_gap",    default=5,   type=int,    help="Max consecutive NaNs to forward-fill.")
    parser.add_argument("--no_fix_ohlc", action="store_true",     help="Skip OHLC consistency repair.")
    parser.add_argument("--pattern",    default="*.csv",          help="File glob pattern.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    pp = CSVPreprocessor(
        csv_dir    = args.csv_dir,
        output_dir = args.output_dir,
        max_gap_fill = args.max_gap,
        fix_ohlc   = not args.no_fix_ohlc,
        verbose    = True,
    )
    pp.process_all(pattern=args.pattern)


if __name__ == "__main__":
    run_preprocessor_cli()
