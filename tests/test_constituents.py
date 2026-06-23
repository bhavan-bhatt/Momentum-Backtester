# tests/test_constituents.py
# ============================================================
# UNIT TESTS — data/constituents.py
# ============================================================

import os
import tempfile
from datetime import datetime

import pandas as pd
import pytest

from config import BacktestConfig
from data.constituents import ConstituentTracker


@pytest.fixture
def membership_file(tmp_path):
    path = tmp_path / "membership.csv"
    df = pd.DataFrame({
        "symbol": ["RELIANCE.NS", "RELIANCE.NS", "YESBANK.NS"],
        "start_date": ["2010-01-01", "2020-01-01", "2010-01-01"],
        "end_date": ["2019-06-30", "", "2019-08-15"],
    })
    df.to_csv(path, index=False)
    return str(path)


def _cfg_with_membership(path: str) -> BacktestConfig:
    cfg = BacktestConfig()
    cfg.advanced.constituents.membership_file = path
    cfg.advanced.constituents.enforce_point_in_time = True
    cfg.data.symbols = ["RELIANCE.NS", "YESBANK.NS", "INFY.NS"]
    return cfg


class TestConstituentTracker:
    def test_degraded_mode_when_file_missing(self):
        cfg = BacktestConfig()
        cfg.advanced.constituents.membership_file = "nonexistent/path.csv"
        tracker = ConstituentTracker(cfg)
        assert tracker.is_available is False
        assert tracker.is_member_on_date("RELIANCE.NS", datetime(2015, 6, 1)) is True

    def test_member_within_interval(self, membership_file):
        tracker = ConstituentTracker(_cfg_with_membership(membership_file))
        assert tracker.is_available is True
        assert tracker.is_member_on_date("RELIANCE.NS", datetime(2015, 1, 1)) is True
        assert tracker.is_member_on_date("RELIANCE.NS", datetime(2019, 8, 1)) is False
        assert tracker.is_member_on_date("RELIANCE.NS", datetime(2021, 1, 1)) is True

    def test_exited_symbol_not_member(self, membership_file):
        tracker = ConstituentTracker(_cfg_with_membership(membership_file))
        assert tracker.is_member_on_date("YESBANK.NS", datetime(2020, 1, 1)) is False

    def test_unknown_symbol_not_member(self, membership_file):
        tracker = ConstituentTracker(_cfg_with_membership(membership_file))
        assert tracker.is_member_on_date("INFY.NS", datetime(2015, 1, 1)) is False

    def test_get_active_universe_filters_symbols(self, membership_file):
        tracker = ConstituentTracker(_cfg_with_membership(membership_file))
        active = tracker.get_active_universe(datetime(2015, 1, 1))
        assert active == {"RELIANCE.NS", "YESBANK.NS"}

    def test_get_membership_changes(self, membership_file):
        tracker = ConstituentTracker(_cfg_with_membership(membership_file))
        changes = tracker.get_membership_changes(
            datetime(2010, 1, 1), datetime(2021, 12, 31)
        )
        assert not changes.empty
        assert set(changes["event"]) <= {"JOINED", "EXITED"}

    def test_build_sample_membership_csv(self, tmp_path):
        cfg = BacktestConfig()
        cfg.data.symbols = ["RELIANCE.NS", "INFY.NS"]
        cfg.data.start_date = "2016-01-01"
        tracker = ConstituentTracker(cfg)
        out = str(tmp_path / "sample.csv")
        tracker.build_sample_membership_csv(out)
        assert os.path.exists(out)
        df = pd.read_csv(out)
        assert list(df["symbol"]) == cfg.data.symbols
