# config.py
# ============================================================
# SINGLE SOURCE OF TRUTH FOR ALL BACKTEST PARAMETERS
# Change values here to re-run with different settings.
# ============================================================

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    """Configuration for data loading."""

    csv_dir: str = "csv"
    symbols: List[str] = field(default_factory=lambda: [
        "RELIANCE.NS",
        "INFY.NS",
        "TCS.NS",
        "HDFCBANK.NS",
        "WIPRO.NS",
    ])
    benchmark_symbol: str = "^NSEI"
    start_date: str = "2016-01-01"
    end_date: str = "2022-12-31"
    price_column: str = "auto"
    max_gap_fill: int = 5


@dataclass
class StrategyConfig:
    """Parameters for all three strategies."""

    # ── Dual MA Crossover ──────────────────────────────────
    fast_ma_window: int = 20
    slow_ma_window: int = 50
    ma_type: str = "SMA"

    # ── RSI ────────────────────────────────────────────────
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    rsi_entry_on_reversal: bool = True

    # ── ATR (used in Combined Strategy and Position Sizing) ──
    atr_period: int = 14
    atr_stop_multiplier: float = 2.0

    # ── Trend Filter (Combined Strategy only) ──────────────
    trend_filter_enabled: bool = True

    # ── Signal Direction ───────────────────────────────────
    allow_short: bool = False


@dataclass
class PortfolioConfig:
    """Parameters for position sizing and risk management."""

    initial_capital: float = 1_000_000.0
    risk_per_trade_pct: float = 0.01
    max_position_pct: float = 0.20
    max_open_positions: int = 5
    max_total_exposure_pct: float = 0.95
    sizing_method: str = "atr"
    fixed_quantity: int = 10


@dataclass
class ExecutionConfig:
    """Parameters for simulated trade execution."""

    slippage_pct: float = 0.0005
    commission_pct: float = 0.0003
    stt_pct: float = 0.001          # SELL side only
    exchange_charges_pct: float = 0.0000345
    stamp_duty_pct: float = 0.00015  # BUY side only
    fill_at: str = "next_open"


@dataclass
class WalkForwardConfig:
    """Parameters for walk-forward validation."""

    train_years: int = 3
    test_years: int = 1
    step_years: int = 1
    min_train_bars: int = 252
    optimize_in_train: bool = False


@dataclass
class ReportConfig:
    """Parameters for output reports."""

    output_dir: str = "output"
    save_html_report: bool = True
    save_csv_trades: bool = True
    save_equity_curve_csv: bool = True
    save_charts_png: bool = True
    chart_style: str = "seaborn-v0_8"
    risk_free_rate: float = 0.065
    trading_days_per_year: int = 252


@dataclass
class BacktestConfig:
    """Master config — pass this single object to all modules."""

    data: DataConfig = field(default_factory=DataConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    report: ReportConfig = field(default_factory=ReportConfig)

    verbose: bool = True
    random_seed: int = 42


# ── DEFAULT INSTANCE ──────────────────────────────────────────────────────
CONFIG = BacktestConfig()
