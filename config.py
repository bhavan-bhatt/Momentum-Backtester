# config.py
# ============================================================
# SINGLE SOURCE OF TRUTH FOR ALL BACKTEST PARAMETERS
# Change values here to re-run with different settings.
# ============================================================

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


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
    secondary_benchmark_symbol: Optional[str] = None
    symbols_file: Optional[str] = None
    start_date: str = "2018-01-01"
    end_date: str = "2026-12-31"
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
class ConstituentsConfig:
    """Point-in-time index membership settings."""

    membership_file: str = "csv/constituents/nifty50_membership.csv"
    enforce_point_in_time: bool = True


@dataclass
class CrossSectionalConfig:
    """Cross-sectional momentum strategy parameters."""

    lookback_months: int = 9
    skip_recent_days: int = 21
    rebalance_freq: str = "monthly"
    top_decile_pct: float = 0.3
    bottom_decile_pct: float = 0.3


@dataclass
class PairsStatArbConfig:
    """Mean-reversion pairs trading parameters."""

    lookback_window: int = 252
    cointegration_pvalue_threshold: float = 0.05
    zscore_entry: float = 2.0
    zscore_exit: float = 0.5
    zscore_stop: float = 3.5
    recheck_cointegration_every: int = 63
    candidate_pairs: List[tuple] = field(default_factory=lambda: [
        ("HDFCBANK.NS", "ICICIBANK.NS"),
    ])


@dataclass
class VolatilityBreakoutConfig:
    """Donchian channel / ATR breakout parameters."""

    donchian_window: int = 20
    atr_period: int = 14
    atr_breakout_multiplier: float = 1.0
    exit_method: str = "opposite_channel"


@dataclass
class RegimeFilterConfig:
    """Market regime detection — turns strategies on/off based on market state."""

    enable_gating: bool = True
    method: str = "sma_200"
    adx_threshold: float = 25.0
    sma_window: int = 200
    strategy_regime_map: Dict[str, str] = field(default_factory=lambda: {
        "DualMA":                 "trend",
        "CrossSectionalMomentum": "trend",
        "VolatilityBreakout":     "trend",
        "PairsStatArb":           "range",
        "RSI":                    "range",
    })


@dataclass
class EnsembleConfig:
    """Multi-strategy portfolio construction."""

    enabled_strategies: List[str] = field(default_factory=lambda: [
        "DualMA", "RSI", "CrossSectionalMomentum", "VolatilityBreakout",
    ])
    weighting_method: str = "equal_vol"
    vol_lookback_days: int = 63
    rebalance_freq: str = "monthly"
    max_sleeve_weight: float = 0.5


@dataclass
class AdvancedExecutionConfig:
    """Liquidity-aware execution costs (supersedes Phase 1 flat-rate model)."""

    use_liquidity_aware_slippage: bool = True
    participation_rate_limit: float = 0.10
    impact_coefficient: float = 0.1
    min_slippage_pct: float = 0.0003


@dataclass
class StatisticsConfig:
    """Statistical significance testing parameters."""

    bootstrap_n_iterations: int = 5000
    bootstrap_block_size: int = 20
    confidence_level: float = 0.90
    n_strategies_tested: int = 4


@dataclass
class AdvancedConfig:
    """Container for all Phase 2 configuration groups."""

    constituents: ConstituentsConfig = field(default_factory=ConstituentsConfig)
    cross_sectional: CrossSectionalConfig = field(default_factory=CrossSectionalConfig)
    pairs: PairsStatArbConfig = field(default_factory=PairsStatArbConfig)
    breakout: VolatilityBreakoutConfig = field(default_factory=VolatilityBreakoutConfig)
    regime: RegimeFilterConfig = field(default_factory=RegimeFilterConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    advanced_execution: AdvancedExecutionConfig = field(default_factory=AdvancedExecutionConfig)
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)


@dataclass
class BacktestConfig:
    """Master config — pass this single object to all modules."""

    data: DataConfig = field(default_factory=DataConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    advanced: AdvancedConfig = field(default_factory=AdvancedConfig)

    verbose: bool = True
    random_seed: int = 42


# ── DEFAULT INSTANCE ──────────────────────────────────────────────────────
CONFIG = BacktestConfig()
