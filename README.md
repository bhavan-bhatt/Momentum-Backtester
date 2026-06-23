# Backtester — Event-Driven Quantitative Backtesting Framework

A production-quality, event-driven backtesting system for Indian equities (NSE).
Supports multiple strategies, realistic transaction cost modelling, and walk-forward validation.

---

## Folder Structure

```
backtester/
├── csv/                 # Market data CSVs (one file per symbol)
│   └── constituents/    # Point-in-time Nifty 50 membership (Phase 2)
├── engine/              # Core event loop, data handler, portfolio, execution
├── strategies/          # Concrete strategy implementations
├── portfolio/           # Multi-strategy ensemble (Phase 2)
├── performance/         # Metrics library + walk-forward engine
├── reports/             # Chart functions + HTML report generator
├── data/                # CSV normalisation + index constituents
├── configs/             # YAML strategy / ensemble configs (Phase 2)
├── tests/               # Unit tests (pytest)
├── output/              # Auto-created at runtime
│   ├── reports/         # HTML reports
│   ├── charts/          # PNG charts
│   └── logs/            # Run logs
├── config.py            # All parameters — single source of truth
├── run_backtest.py      # Entry point: single backtest
└── run_walk_forward.py  # Entry point: walk-forward validation
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your CSV data

Place CSV files in the `csv/` directory.  
File naming must match the symbols in `config.py`:
- `RELIANCE.NS.csv`
- `INFY.NS.csv`
- `^NSEI.csv` (benchmark)

See [Data Contract](#data-contract) for accepted formats.

### 3. Run a backtest

```bash
cd backtester

# Combined strategy (default)
python run_backtest.py

# Dual Moving Average crossover
python run_backtest.py --strategy dual_ma

# RSI mean-reversion
python run_backtest.py --strategy rsi

# Cross-sectional momentum (Phase 2)
python run_backtest.py --strategy cross_sectional

# Pairs / stat arb (requires allow_short=True and both leg CSVs)
python run_backtest.py --strategy pairs_stat_arb

# Donchian / ATR volatility breakout
python run_backtest.py --strategy volatility_breakout

# Override capital and date range
python run_backtest.py --strategy combined --capital 2000000 --start 2018-01-01 --end 2022-12-31
```

### 4. Run walk-forward validation

```bash
python run_walk_forward.py --strategy combined --train 3 --test 1
```

### 5. Run tests

```bash
pytest tests/ -v
```

---

## Configuration

All parameters live in `config.py`. Key sections:

| Dataclass | Purpose |
|---|---|
| `DataConfig` | CSV directory, symbol list, date range |
| `StrategyConfig` | MA windows, RSI thresholds, ATR period |
| `PortfolioConfig` | Capital, sizing method, risk limits |
| `ExecutionConfig` | Slippage, brokerage, STT, stamp duty |
| `WalkForwardConfig` | Train/test window sizes |
| `ReportConfig` | Output directories, chart style |
| `AdvancedConfig` | Phase 2: constituents, ensemble, regime filter, statistics |

Phase 2 settings live under `config.advanced` and are fully backward compatible — Phase 1 runs work unchanged with defaults.

---

## Strategies

### Phase 1 — Time-Series (per-symbol)

#### 1. Dual MA Crossover (`strategies/dual_ma.py`)
- **Entry**: Fast MA crosses above Slow MA (golden cross).
- **Exit**: Fast MA crosses below Slow MA (death cross).
- Parameters: `fast_ma_window`, `slow_ma_window`, `ma_type` (SMA/EMA).

#### 2. RSI Mean-Reversion (`strategies/rsi_strategy.py`)
- **Entry**: RSI recovers above oversold level (reversal confirmation).
- **Exit**: RSI reaches overbought, or RSI below 50 for 30+ bars.
- Parameters: `rsi_period`, `rsi_oversold`, `rsi_overbought`, `rsi_entry_on_reversal`.

#### 3. Combined (`strategies/combined.py`)
- **Trend Filter**: Only long when fast MA > slow MA.
- **Entry**: RSI reversal within confirmed uptrend.
- **Exit**: ATR stop-loss, RSI overbought, or MA death cross.
- Uses all parameters from StrategyConfig.

### Phase 2 — Cross-Sectional & Multi-Strategy (in progress)

#### 4. Cross-Sectional Momentum (`strategies/cross_sectional_momentum.py`)
- **Logic**: Rank the active universe by trailing N-month return (skipping the most recent month).
- **Long**: Top fraction (`top_decile_pct`). **Short**: Bottom fraction (if `allow_short=True`).
- **Rebalance**: Monthly or weekly. Uses `ConstituentTracker` for point-in-time universe.
- Parameters: `config.advanced.cross_sectional.*`

#### 5. Pairs / Stat Arb (`strategies/pairs_stat_arb.py`)
- **Logic**: Engle-Granger cointegration test on candidate pairs; trade spread z-score.
- **Entry**: |z| > `zscore_entry`. **Exit**: |z| < `zscore_exit` or stop at `zscore_stop`.
- Requires `allow_short=True`. Default pair: HDFCBANK / ICICIBANK.
- Parameters: `config.advanced.pairs.*`

#### 6. Volatility Breakout (`strategies/volatility_breakout.py`)
- **Logic**: Donchian channel breakout confirmed by ATR-multiple move.
- **Exit**: Opposite channel or ATR trailing stop (`exit_method`).
- Parameters: `config.advanced.breakout.*`

### Phase 2 — Research pipeline

```bash
# Full ensemble research run from YAML config
python run_research.py --config configs/base.yaml

# With walk-forward parameter stability (slow)
python run_research.py --config configs/ensemble_all.yaml --with-stability
```

See `configs/README.md` for available YAML configs.

#### Planned (Phase 2 modules)
- **Ensemble** — risk-parity combination of strategy sleeves via `portfolio/ensemble.py`
- **Advanced execution** — liquidity-aware slippage via `engine/execution_advanced.py`
- **Regime filter** — pause strategies outside preferred regime via `engine/regime_filter.py`

---

## Data Contract

### Minimum Required Columns

| Tier | Columns Required | Features |
|---|---|---|
| 1 (Full) | date, open, high, low, close, adj_close, volume | ATR sizing, vol filters |
| 2 (Reduced) | date, open, high, low, close | ATR sizing, no vol |
| 3 (Minimal) | date, close | Equal-weight sizing only |

### Accepted Date Formats

`YYYY-MM-DD`, `DD-MM-YYYY`, `DD/MM/YYYY`, `YYYY/MM/DD`, `DD-Mon-YYYY`

### Accepted Column Name Variants

| Canonical | Accepted Variants |
|---|---|
| date | Date, timestamp, time |
| open | Open, OPEN, open_price |
| high | High, HIGH, high_price |
| low | Low, LOW, low_price |
| close | Close, CLOSE, close_price, last |
| adj_close | Adj Close, adj_close, Adjusted Close |
| volume | Volume, VOLUME, vol, turnover |

---

## Position Sizing Methods

| Method | Logic |
|---|---|
| `atr` | Risk `risk_per_trade_pct` × equity, stop = `atr_stop_multiplier` × ATR |
| `equal_weight` | Equity ÷ `max_open_positions` per trade |
| `fixed` | `fixed_quantity` shares per trade |

---

## Transaction Cost Model (NSE Delivery)

| Cost | Rate | Side |
|---|---|---|
| Brokerage | 0.03% | Both |
| Exchange charges | 0.00345% | Both |
| STT | 0.1% | Sell |
| Stamp duty | 0.015% | Buy |
| Slippage | 0.05% | Directional |

---

## Output Files

After a run, the `output/` directory contains:
- `output/reports/` — Self-contained HTML report with interactive Plotly chart
- `output/charts/` — Static PNG charts (equity curve, monthly heatmap, trade analysis)
- `output/` — Equity curve CSV and trade log CSV

---

## Architecture

The system uses a **classic event-driven architecture**. Phase 1 runs a single strategy sleeve; Phase 2 adds point-in-time universe filtering, regime gating, multiple strategy sleeves, and an ensemble layer:

```
data/constituents.py  →  filters tradable universe per bar
        ↓
DataHandler  →  MarketEvent (+ regime state in Phase 2)
        ↓
regime_filter.py  →  each strategy queries is_active(strategy_name)
        ↓
┌──────────────┬──────────────┬──────────────┬──────────────────┐
│ Dual MA/RSI  │ Cross-Sect.  │ Pairs/Arb    │ Vol. Breakout    │
│ / Combined   │ Momentum     │              │                  │
└──────┬───────┴──────┬───────┴──────┬───────┴────────┬─────────┘
       └──────────────► SignalEvents ──────────────────┘
                              ↓
              portfolio/ensemble.py  (risk-parity weights)
                              ↓
              PortfolioManager  +  audit_log.py
                              ↓
              execution_advanced.py  (liquidity-aware fills)
                              ↓
              performance/  +  reports/advanced_charts.py
```

All modules communicate exclusively through events. No direct calls between modules.

---

## Replacing Demo Data

1. Download historical data from yfinance, NSEpy, or Kite API.
2. Place CSVs in the `csv/` directory with filenames matching `config.data.symbols`.
3. Optionally run the preprocessor to normalise formats:
   ```bash
   python -m data.preprocessor --csv_dir csv --output_dir csv
   ```
4. Run the backtest.

---

## Disclaimer

This framework is intended for **research and educational purposes only**.
It does not constitute financial advice. Past performance in a backtest
does not guarantee future returns.
