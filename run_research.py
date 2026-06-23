# run_research.py
# ============================================================
# RESEARCH-MODE ENTRY POINT — FULL PHASE 2 PIPELINE
# ============================================================

import argparse
import logging
import os
import sys
from collections import deque
from datetime import datetime

from config_loader import load_config_from_yaml, validate_config
from data.constituents import ConstituentTracker
from engine.audit_log import AuditLog
from engine.data_handler import DataHandler
from engine.execution_advanced import AdvancedExecutionHandler
from engine.portfolio import PortfolioManager
from engine.regime_filter import RegimeFilter
from engine.research_backtest import ResearchBacktestEngine
from performance.benchmark import BenchmarkComparison
from performance.metrics import PerformanceMetrics
from performance.risk_metrics import ExtendedRiskMetrics
from performance.statistics import StatisticalTests
from performance.walk_forward_v2 import WalkForwardEngineV2
from portfolio.ensemble import EnsembleAllocator
from reports.generator import ReportGenerator
from run_backtest import get_strategy, setup_logging
from strategies.combined import CombinedStrategy
from strategies.cross_sectional_momentum import CrossSectionalMomentumStrategy
from strategies.dual_ma import DualMAStrategy
from strategies.pairs_stat_arb import PairsStatArbStrategy
from strategies.rsi_strategy import RSIStrategy
from strategies.volatility_breakout import VolatilityBreakoutStrategy

logger = logging.getLogger(__name__)

_ENSEMBLE_CLASS_MAP = {
    "DualMA":                 DualMAStrategy,
    "RSI":                    RSIStrategy,
    "Combined":               CombinedStrategy,
    "CrossSectionalMomentum": CrossSectionalMomentumStrategy,
    "VolatilityBreakout":     VolatilityBreakoutStrategy,
    "PairsStatArb":           PairsStatArbStrategy,
}

_CLI_TO_ENSEMBLE = {
    "dual_ma":              "DualMA",
    "rsi":                  "RSI",
    "combined":             "Combined",
    "cross_sectional":      "CrossSectionalMomentum",
    "volatility_breakout":  "VolatilityBreakout",
    "pairs_stat_arb":       "PairsStatArb",
}


def _build_ensemble_strategies(config, queue, tracker, regime, audit_log):
    """Instantiate all enabled strategy sleeves."""
    strategies = []
    for name in config.advanced.ensemble.enabled_strategies:
        cls = _ENSEMBLE_CLASS_MAP.get(name)
        if cls is None:
            logger.warning("Unknown ensemble strategy '%s' — skipped.", name)
            continue

        if cls is CrossSectionalMomentumStrategy:
            s = cls(config, queue, constituent_tracker=tracker, regime_filter=regime)
        elif cls in (PairsStatArbStrategy, VolatilityBreakoutStrategy):
            s = cls(config, queue, regime_filter=regime)
        else:
            s = cls(config, queue)
            s.set_regime_filter(regime, name)

        if audit_log is not None:
            s.set_audit_log(audit_log)
        strategies.append(s)

    return strategies


def run_full_research_pipeline(config_path: str, with_stability: bool = False) -> None:
    """Orchestrate the complete Phase 2 research pipeline."""
    config = load_config_from_yaml(config_path)
    warnings = validate_config(config)
    for w in warnings:
        print(f"  CONFIG WARNING: {w}")

    setup_logging(config)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"research_{run_ts}"

    tracker = ConstituentTracker(config)
    regime = RegimeFilter(config)
    audit_log = AuditLog(run_id=run_id)
    queue = deque()

    strategies = _build_ensemble_strategies(config, queue, tracker, regime, audit_log)
    if not strategies:
        print("ERROR: No valid strategies in ensemble.enabled_strategies.")
        sys.exit(1)

    strategy_ids = [s.strategy_id for s in strategies]
    ensemble = EnsembleAllocator(config, strategy_ids=strategy_ids)

    data_handler = DataHandler(config, queue)
    portfolio = PortfolioManager(config, queue)
    portfolio.enable_ensemble_mode()
    execution = AdvancedExecutionHandler(config, queue)

    engine = ResearchBacktestEngine(
        config,
        data_handler,
        strategies,
        portfolio,
        execution,
        regime_filter=regime,
        ensemble_allocator=ensemble,
        audit_log=audit_log,
    )

    print(f"\n{'='*60}")
    print(f"  Research run — {len(strategies)} strategy sleeve(s)")
    print(f"  Config: {config_path}")
    print(f"{'='*60}\n")

    results = engine.run()
    equity_curve = results["equity_curve"]
    trade_log = results["trade_log"]
    metrics = results["metrics"]
    benchmark_curve = results["benchmark_curve"]

    perf = PerformanceMetrics(config)
    returns = perf.calculate_returns(equity_curve)

    stats = StatisticalTests(config)
    sharpe_ci = stats.block_bootstrap_sharpe_ci(returns)
    dsr = stats.deflated_sharpe_ratio(returns)
    jb = stats.jarque_bera_normality_test(returns)

    risk = ExtendedRiskMetrics(config)
    tail_ratio = risk.calculate_tail_ratio(returns)
    worst_recovery, recovery_df = risk.calculate_time_to_recovery(equity_curve)
    turnover = risk.calculate_turnover(trade_log, equity_curve)
    omega = risk.calculate_omega_ratio(returns)

    avg_volumes = {}
    for sym in config.data.symbols:
        bars = data_handler.get_latest_bars(sym, 20)
        if bars is not None and "volume" in bars.columns:
            vol = bars["volume"].dropna().mean()
            if vol and vol > 0:
                avg_volumes[sym] = float(vol)

    capacity_report = execution.get_capacity_report()
    capacity = risk.estimate_capacity(trade_log, capacity_report, avg_volumes)

    bench = BenchmarkComparison(config)
    bench_comparisons = []
    if benchmark_curve is not None and len(benchmark_curve) > 1:
        raw_prices = data_handler.benchmark_data["close"] if data_handler.benchmark_data is not None else None
        if raw_prices is not None:
            for apply_costs, label_suffix in ((False, ""), (True, " (cost-adjusted)")):
                bh = bench.build_buy_and_hold_curve(raw_prices, apply_costs=apply_costs)
                bench_comparisons.append(
                    bench.compare(
                        equity_curve,
                        bh,
                        f"Nifty 50 Buy & Hold{label_suffix}",
                    )
                )

    symbol_prices = {}
    for sym in config.data.symbols:
        if sym in data_handler.symbol_data:
            symbol_prices[sym] = data_handler.symbol_data[sym]["close"]

    if symbol_prices:
        for apply_costs, label_suffix in ((False, ""), (True, " (cost-adjusted)")):
            basket = bench.build_equal_weight_basket_curve(
                symbol_prices, apply_costs=apply_costs
            )
            if len(basket) > 1:
                bench_comparisons.append(
                    bench.compare(
                        equity_curve,
                        basket,
                        f"Equal-Weight Basket{label_suffix}",
                    )
                )

    wf2_results = None
    stability_df = None
    stability_report = None
    if with_stability:
        wf_engine = WalkForwardEngineV2(config)
        primary_cls = type(strategies[0])
        wf2_results = wf_engine.run_both_variants(primary_cls)
        stability_df = wf_engine.run_parameter_stability_analysis(primary_cls)
        stability_report = wf_engine.generate_stability_report(stability_df)

    advanced_analysis = {
        "sharpe_ci": {
            "point": sharpe_ci[0],
            "lower": sharpe_ci[1],
            "upper": sharpe_ci[2],
        },
        "deflated_sharpe": dsr,
        "jarque_bera": jb,
        "tail_ratio": tail_ratio,
        "worst_recovery_days": worst_recovery,
        "recovery_episodes": recovery_df,
        "turnover": turnover,
        "omega_ratio": omega,
        "capacity": capacity,
        "benchmark_comparisons": bench_comparisons,
        "regime_series": regime.get_regime_series(),
        "ensemble_weights": ensemble.get_weight_history(),
        "wf2_results": wf2_results,
        "stability_report": stability_report,
    }

    metrics.update({
        "dsr": dsr.get("deflated_sharpe_ratio"),
        "tail_ratio": tail_ratio,
        "annual_turnover": turnover.get("annual_turnover"),
        "omega_ratio": omega if omega != float("inf") else None,
    })

    reporter = ReportGenerator(config)
    strategy_label = "Ensemble_" + "_".join(config.advanced.ensemble.enabled_strategies[:3])
    report_path = reporter.generate_all(
        equity_curve=equity_curve,
        trade_log=trade_log,
        metrics=metrics,
        benchmark_curve=benchmark_curve,
        strategy_name=strategy_label,
    )
    reporter.generate_advanced_sections(
        report_path=report_path,
        advanced_analysis=advanced_analysis,
        audit_log=audit_log,
    )

    audit_path = os.path.join(
        config.report.output_dir, "logs", f"{run_id}_audit.jsonl"
    )
    os.makedirs(os.path.dirname(audit_path), exist_ok=True)
    audit_log.save_jsonl(audit_path)

    print(f"\nReport saved to: {report_path}")
    print(f"Audit log saved to: {audit_path}")
    print(f"\nDeflated Sharpe Ratio: {dsr['deflated_sharpe_ratio']:.1%}")
    print(f"  → {dsr['interpretation']}")
    print(
        f"Bootstrap Sharpe CI ({config.advanced.statistics.confidence_level:.0%}): "
        f"[{sharpe_ci[1]:.2f}, {sharpe_ci[2]:.2f}]"
    )
    if sharpe_ci[1] <= 0:
        print("  → Lower CI bound ≤ 0: positive Sharpe may be noise.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 research pipeline")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--with-stability",
        action="store_true",
        help="Run parameter stability analysis (slow).",
    )
    args = parser.parse_args()
    run_full_research_pipeline(args.config, with_stability=args.with_stability)
