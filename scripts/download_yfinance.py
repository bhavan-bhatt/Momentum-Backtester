#!/usr/bin/env python3
"""Download OHLCV history from Yahoo Finance and write backtester CSVs."""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

DEFAULT_SYMBOLS = [
    "RELIANCE.NS",
    "INFY.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "WIPRO.NS",
]
DEFAULT_BENCHMARK = "^NSEI"


def _symbol_to_filename(symbol: str) -> str:
    return f"{symbol}.csv"


def download_symbol(
    symbol: str,
    start: str,
    end: str,
    output_dir: Path,
) -> Path:
    """Fetch one ticker from yfinance and save in backtester CSV format."""
    logger.info("Downloading %s (%s → %s)", symbol, start, end)

    df = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        raise ValueError(f"No data returned for {symbol}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    rename = {
        "Date": "Date",
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Adj Close": "Adj Close",
        "Volume": "Volume",
    }
    df = df.rename(columns=rename)

    required = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{symbol} missing columns: {missing}")

    df = df[required].copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")

    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Close"])
    if df.empty:
        raise ValueError(f"{symbol} has no valid close prices after cleaning")

    out_path = output_dir / _symbol_to_filename(symbol)
    df.to_csv(out_path, index=False)
    logger.info("Saved %s (%d bars)", out_path.name, len(df))
    return out_path


def download_all(
    symbols: list[str],
    benchmark: str,
    start: str,
    end: str,
    output_dir: str,
) -> list[Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tickers = list(symbols)
    if benchmark and benchmark not in tickers:
        tickers.append(benchmark)

    saved: list[Path] = []
    failed: list[str] = []

    for symbol in tickers:
        try:
            saved.append(download_symbol(symbol, start, end, out))
        except Exception as exc:
            logger.error("Failed to download %s: %s", symbol, exc)
            failed.append(symbol)

    if failed:
        raise RuntimeError(f"Download failed for: {', '.join(failed)}")

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download NSE equity data from Yahoo Finance into csv/."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help="Ticker symbols (default: 5 Nifty stocks).",
    )
    parser.add_argument(
        "--benchmark",
        default=DEFAULT_BENCHMARK,
        help="Benchmark index symbol (default: ^NSEI).",
    )
    parser.add_argument("--start", default="2018-01-01", help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", default="2026-12-31", help="End date (YYYY-MM-DD).")
    parser.add_argument(
        "--output-dir",
        default="csv",
        help="Directory for output CSV files.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        paths = download_all(
            symbols=args.symbols,
            benchmark=args.benchmark,
            start=args.start,
            end=args.end,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\nDownloaded {len(paths)} file(s) to {args.output_dir}/")
    for p in paths:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
