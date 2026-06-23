# tests/test_phase2_analysis.py
# ============================================================
# UNIT TESTS — Modules H–M (statistics, risk, benchmark, config, audit)
# ============================================================

import os
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from config import BacktestConfig
from config_loader import load_config_from_yaml, save_config_to_yaml, validate_config
from engine.audit_log import AuditLog
from performance.benchmark import BenchmarkComparison
from performance.risk_metrics import ExtendedRiskMetrics
from performance.statistics import StatisticalTests
from performance.walk_forward_v2 import WalkForwardEngineV2


@pytest.fixture
def sample_returns():
    rng = np.random.default_rng(42)
    idx = pd.date_range("2018-01-01", periods=300, freq="B")
    r = pd.Series(rng.normal(0.0005, 0.01, len(idx)), index=idx)
    return r


@pytest.fixture
def sample_equity(sample_returns):
    return (1 + sample_returns).cumprod() * 1_000_000


class TestStatisticalTests:
    def test_bootstrap_sharpe_ci(self, sample_returns):
        cfg = BacktestConfig()
        cfg.advanced.statistics.bootstrap_n_iterations = 200
        stats = StatisticalTests(cfg)
        point, lower, upper = stats.block_bootstrap_sharpe_ci(sample_returns)
        assert lower <= point <= upper or point == lower == upper

    def test_deflated_sharpe_returns_dict(self, sample_returns):
        stats = StatisticalTests(BacktestConfig())
        dsr = stats.deflated_sharpe_ratio(sample_returns, n_strategies_tested=4)
        assert 0 <= dsr["deflated_sharpe_ratio"] <= 1
        assert "interpretation" in dsr

    def test_jarque_bera(self, sample_returns):
        stats = StatisticalTests(BacktestConfig())
        jb = stats.jarque_bera_normality_test(sample_returns)
        assert "is_normal" in jb


class TestExtendedRiskMetrics:
    def test_tail_ratio(self, sample_returns):
        risk = ExtendedRiskMetrics(BacktestConfig())
        tr = risk.calculate_tail_ratio(sample_returns)
        assert tr >= 0

    def test_time_to_recovery(self, sample_equity):
        risk = ExtendedRiskMetrics(BacktestConfig())
        worst, df = risk.calculate_time_to_recovery(sample_equity)
        assert isinstance(df, pd.DataFrame)

    def test_turnover_empty(self):
        risk = ExtendedRiskMetrics(BacktestConfig())
        t = risk.calculate_turnover([], pd.Series([1_000_000, 1_100_000]))
        assert t["total_round_trips"] == 0

    def test_omega_ratio(self, sample_returns):
        risk = ExtendedRiskMetrics(BacktestConfig())
        assert risk.calculate_omega_ratio(sample_returns) > 0


class TestBenchmarkComparison:
    def test_buy_and_hold(self):
        cfg = BacktestConfig()
        prices = pd.Series(
            np.linspace(100, 150, 100),
            index=pd.date_range("2020-01-01", periods=100, freq="B"),
        )
        bench = BenchmarkComparison(cfg)
        curve = bench.build_buy_and_hold_curve(prices, apply_costs=True)
        assert len(curve) == len(prices)
        assert curve.iloc[-1] > 0

    def test_compare(self, sample_equity):
        cfg = BacktestConfig()
        bench = BenchmarkComparison(cfg)
        bm = sample_equity * 0.9
        result = bench.compare(sample_equity, bm, "Test")
        assert "outperformance" in result


class TestConfigLoader:
    def test_load_base_yaml(self):
        path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "base.yaml"
        )
        cfg = load_config_from_yaml(path)
        assert len(cfg.data.symbols) >= 5
        assert "DualMA" in cfg.advanced.ensemble.enabled_strategies

    def test_unknown_key_raises(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("strategy:\n  not_a_field: 1\n")
        with pytest.raises(ValueError, match="Unknown"):
            load_config_from_yaml(str(bad))

    def test_save_and_validate(self, tmp_path):
        cfg = BacktestConfig()
        out = str(tmp_path / "out.yaml")
        save_config_to_yaml(cfg, out)
        assert os.path.exists(out)
        warnings = validate_config(cfg)
        assert isinstance(warnings, list)


class TestAuditLog:
    def test_record_and_export(self, tmp_path):
        log = AuditLog("test_run")
        rec = log.record(
            datetime(2020, 1, 1),
            "SIGNAL",
            "RELIANCE.NS",
            "test reason",
            strategy_id="DualMA",
            values={"fast_ma": 100.0},
        )
        log.link_outcome(rec, 500.0)
        df = log.to_dataframe()
        assert len(df) == 1
        path = str(tmp_path / "audit.jsonl")
        log.save_jsonl(path)
        assert os.path.exists(path)

    def test_filter_by(self):
        log = AuditLog("test")
        log.record(datetime(2020, 1, 1), "SIGNAL", "X", "r", strategy_id="A")
        log.record(datetime(2020, 1, 2), "FILL", "Y", "r2")
        assert len(log.filter_by(event_type="SIGNAL")) == 1


class TestWalkForwardV2:
    def test_anchored_splits_generated(self):
        cfg = BacktestConfig()
        wf = WalkForwardEngineV2(cfg)
        splits = wf.generate_splits_anchored()
        assert len(splits) >= 1
        assert splits[0][0] == cfg.data.start_date

    def test_stability_report_empty(self):
        wf = WalkForwardEngineV2(BacktestConfig())
        report = wf.generate_stability_report(pd.DataFrame())
        assert report == {}
