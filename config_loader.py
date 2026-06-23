# config_loader.py
# ============================================================
# LOAD BacktestConfig FROM YAML
# ============================================================

import dataclasses
import logging
import os
from typing import Any, Dict, List, Type, TypeVar

import yaml

from config import (
    AdvancedConfig,
    AdvancedExecutionConfig,
    BacktestConfig,
    ConstituentsConfig,
    CrossSectionalConfig,
    DataConfig,
    EnsembleConfig,
    ExecutionConfig,
    PairsStatArbConfig,
    PortfolioConfig,
    RegimeFilterConfig,
    ReportConfig,
    StatisticsConfig,
    StrategyConfig,
    VolatilityBreakoutConfig,
    WalkForwardConfig,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _build_dataclass(dataclass_type: Type[T], values: Dict[str, Any]) -> T:
    """Construct a dataclass from a dict, rejecting unknown keys."""
    valid_fields = {f.name for f in dataclasses.fields(dataclass_type)}
    unknown_keys = set(values.keys()) - valid_fields
    if unknown_keys:
        raise ValueError(
            f"Unknown config keys for {dataclass_type.__name__}: {unknown_keys}. "
            "Check for typos."
        )
    filtered = {k: v for k, v in values.items() if k in valid_fields}
    return dataclass_type(**filtered)


def _build_advanced(raw: Dict[str, Any]) -> AdvancedConfig:
    """Build AdvancedConfig from nested YAML section."""
    return AdvancedConfig(
        constituents=_build_dataclass(ConstituentsConfig, raw.get("constituents", {})),
        cross_sectional=_build_dataclass(
            CrossSectionalConfig, raw.get("cross_sectional", {})
        ),
        pairs=_build_dataclass(PairsStatArbConfig, raw.get("pairs", {})),
        breakout=_build_dataclass(
            VolatilityBreakoutConfig, raw.get("breakout", {})
        ),
        regime=_build_dataclass(RegimeFilterConfig, raw.get("regime", {})),
        ensemble=_build_dataclass(EnsembleConfig, raw.get("ensemble", {})),
        advanced_execution=_build_dataclass(
            AdvancedExecutionConfig, raw.get("advanced_execution", {})
        ),
        statistics=_build_dataclass(StatisticsConfig, raw.get("statistics", {})),
    )


def load_symbols_from_file(filepath: str) -> List[str]:
    """Load ticker symbols from a text file (one per line, # comments ignored)."""
    symbols: List[str] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            symbols.append(line)
    return symbols


def _resolve_symbols_file(data_raw: Dict[str, Any], config_path: str) -> Dict[str, Any]:
    """If symbols_file is set, load symbols list from that path."""
    data_raw = dict(data_raw)
    sym_file = data_raw.pop("symbols_file", None)
    if sym_file:
        if not os.path.isabs(sym_file):
            base = os.path.dirname(os.path.abspath(config_path))
            sym_file = os.path.join(base, sym_file)
        if not os.path.exists(sym_file):
            raise FileNotFoundError(f"symbols_file not found: {sym_file}")
        data_raw["symbols"] = load_symbols_from_file(sym_file)
        logger.info("Loaded %d symbols from %s", len(data_raw["symbols"]), sym_file)
    return data_raw


def load_config_from_yaml(filepath: str) -> BacktestConfig:
    """Load a complete BacktestConfig from a YAML file."""
    with open(filepath, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    top_keys = {"data", "strategy", "portfolio", "execution", "walk_forward",
                "report", "advanced", "verbose", "random_seed"}
    unknown_top = set(raw.keys()) - top_keys
    if unknown_top:
        raise ValueError(f"Unknown top-level config keys: {unknown_top}")

    data_raw = _resolve_symbols_file(raw.get("data", {}), filepath)

    config = BacktestConfig(
        data=_build_dataclass(DataConfig, data_raw),
        strategy=_build_dataclass(StrategyConfig, raw.get("strategy", {})),
        portfolio=_build_dataclass(PortfolioConfig, raw.get("portfolio", {})),
        execution=_build_dataclass(ExecutionConfig, raw.get("execution", {})),
        walk_forward=_build_dataclass(
            WalkForwardConfig, raw.get("walk_forward", {})
        ),
        report=_build_dataclass(ReportConfig, raw.get("report", {})),
        advanced=_build_advanced(raw.get("advanced", {})),
        verbose=raw.get("verbose", True),
        random_seed=raw.get("random_seed", 42),
    )

    logger.info(
        "Loaded config from %s: symbols=%d, date_range=%s→%s",
        filepath,
        len(config.data.symbols),
        config.data.start_date,
        config.data.end_date,
    )
    return config


def save_config_to_yaml(config: BacktestConfig, filepath: str) -> None:
    """Serialise BacktestConfig to YAML."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(
            dataclasses.asdict(config),
            f,
            default_flow_style=False,
            sort_keys=False,
        )
    logger.info("Config saved to %s", filepath)


def validate_config(config: BacktestConfig) -> list:
    """Return advisory warnings for potentially misconfigured settings."""
    warnings: List[str] = []

    if config.strategy.fast_ma_window >= config.strategy.slow_ma_window:
        warnings.append("fast_ma_window must be < slow_ma_window")

    enabled = config.advanced.ensemble.enabled_strategies
    if "PairsStatArb" in enabled and not config.strategy.allow_short:
        warnings.append(
            "PairsStatArb requires allow_short=True — it will not function correctly."
        )

    n_sleeves = len(enabled)
    if n_sleeves > 0:
        max_w = config.advanced.ensemble.max_sleeve_weight
        if max_w * n_sleeves < 1.0:
            warnings.append(
                f"max_sleeve_weight too restrictive — weights cannot sum to 1.0 "
                f"with {n_sleeves} sleeves and a {max_w} cap each."
            )

    if (
        config.advanced.constituents.enforce_point_in_time
        and not os.path.exists(config.advanced.constituents.membership_file)
    ):
        warnings.append(
            "Point-in-time enforcement requested but membership file not found — "
            "will silently degrade to always-tradable. Run "
            "ConstituentTracker.build_sample_membership_csv() first."
        )

    if config.advanced.statistics.n_strategies_tested < 1:
        warnings.append(
            "n_strategies_tested must be >= 1 for valid Deflated Sharpe Ratio."
        )

    return warnings
