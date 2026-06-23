#!/usr/bin/env python3
"""Download OHLCV history from Yahoo Finance and write backtester CSVs."""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_BENCHMARKS = ["^NSEI", "^CRSLDX"]


def load_symbols_file(path: str) -> list[str]:
    symbols: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                symbols.append(line)
    return symbols


def _symbol_to_filename(symbol: str) -> str:
    return f"{symbol}.csv"


def download_symbol(
    symbol: str,
    start: str,
    end: str,
    output_dir: Path,
) -> Path:
    """Fetch one ticker from yfinance and save in backtester CSV format."""
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
    return out_path


def download_all(
    symbols: list[str],
    benchmarks: list[str],
    start: str,
    end: str,
    output_dir: str,
    min_success_pct: float = 0.85,
) -> tuple[list[Path], list[str]]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tickers: list[str] = []
    seen: set[str] = set()
    for t in list(symbols) + list(benchmarks):
        if t and t not in seen:
            tickers.append(t)
            seen.add(t)

    saved: list[Path] = []
    failed: list[str] = []

    for symbol in tqdm(tickers, desc="Downloading", unit="ticker"):
        try:
            saved.append(download_symbol(symbol, start, end, out))
        except Exception as exc:
            logger.warning("Failed %s: %s", symbol, exc)
            failed.append(symbol)

    n_stocks = len([s for s in symbols if s not in failed])
    min_ok = int(len(symbols) * min_success_pct)
    if n_stocks < min_ok:
        raise RuntimeError(
            f"Only {n_stocks}/{len(symbols)} stocks downloaded "
            f"(need at least {min_ok}). Failed: {', '.join(failed[:20])}..."
        )

    if failed:
        logger.warning(
            "Skipped %d tickers: %s",
            len(failed),
            ", ".join(failed[:15]) + ("..." if len(failed) > 15 else ""),
        )

    return saved, failed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download NSE equity data from Yahoo Finance into csv/."
    )
    parser.add_argument(
        "--symbols-file",
        default="configs/symbols_n100.txt",
        help="Path to symbol list (one ticker per line).",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=DEFAULT_BENCHMARKS,
        help="Benchmark indices (default: ^NSEI ^CRSLDX).",
    )
    parser.add_argument("--start", default="2018-01-01", help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", default="2026-06-24", help="End date (YYYY-MM-DD).")
    parser.add_argument(
        "--output-dir",
        default="csv",
        help="Directory for output CSV files.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    sym_path = Path(args.symbols_file)
    if not sym_path.is_file():
        print(f"ERROR: symbols file not found: {sym_path}", file=sys.stderr)
        sys.exit(1)

    symbols = load_symbols_file(str(sym_path))
    print(f"Universe: {len(symbols)} symbols + {len(args.benchmarks)} benchmarks")

    try:
        paths, failed = download_all(
            symbols=symbols,
            benchmarks=args.benchmarks,
            start=args.start,
            end=args.end,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\nDownloaded {len(paths)} file(s) to {args.output_dir}/")
    if failed:
        print(f"Failed ({len(failed)}): {', '.join(failed)}")


if __name__ == "__main__":
    main()
