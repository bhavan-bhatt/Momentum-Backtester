# Market data (not committed)

OHLCV CSV files are **not** stored in git. Download them locally before running a backtest.

## Quick download

From the project root:

```bash
python scripts/download_yfinance.py \
  --symbols-file configs/symbols_n150.txt \
  --start 2018-01-01 \
  --end 2026-06-23 \
  --output-dir csv
```

This fetches **150 NSE symbols** plus **Nifty 50** (`^NSEI`) and **Nifty 500** (`^CRSLDX`) benchmarks via Yahoo Finance.

## Expected layout

```
csv/
├── RELIANCE.NS.csv
├── TCS.NS.csv
├── ...
├── ^NSEI.csv
└── ^CRSLDX.csv
```

See the main [README](../README.md#data-contract) for the CSV column format.
