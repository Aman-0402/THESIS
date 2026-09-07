# tests/sensitivity/test_dataset_builder.py
import pandas as pd

from sensitivity.lib.dataset_builder import (
    period_ends_in_range,
    compute_target_with_threshold,
    apply_feature_scope,
)


def test_period_ends_in_range_quarterly_matches_pipeline_behavior():
    ends = period_ends_in_range(pd.Timestamp("2014-07-01"), pd.Timestamp("2026-03-31"), "QE")
    assert ends[0] == pd.Timestamp("2014-09-30")
    assert ends[-1] == pd.Timestamp("2026-03-31")


def test_period_ends_in_range_monthly_has_more_periods_than_quarterly():
    start, end = pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31")
    monthly = period_ends_in_range(start, end, "ME")
    quarterly = period_ends_in_range(start, end, "QE")
    assert len(monthly) == 12
    assert len(quarterly) == 4


def test_compute_target_with_threshold_simple_mode_matches_pipeline():
    target, keep = compute_target_with_threshold(100.0, 110.0, mode="simple", deadzone_pct=0.0)
    assert (target, keep) == (1, True)
    target, keep = compute_target_with_threshold(100.0, 90.0, mode="simple", deadzone_pct=0.0)
    assert (target, keep) == (0, True)


def test_compute_target_with_threshold_deadzone_drops_small_moves():
    # +0.5% move: inside the 1% dead zone -> dropped
    target, keep = compute_target_with_threshold(100.0, 100.5, mode="deadzone", deadzone_pct=0.01)
    assert keep is False
    # +2% move: outside the dead zone -> kept, labeled positive
    target, keep = compute_target_with_threshold(100.0, 102.0, mode="deadzone", deadzone_pct=0.01)
    assert (target, keep) == (1, True)
    # -2% move: outside the dead zone -> kept, labeled negative
    target, keep = compute_target_with_threshold(100.0, 98.0, mode="deadzone", deadzone_pct=0.01)
    assert (target, keep) == (0, True)


def test_apply_feature_scope_market_only_nulls_fundamentals():
    market = {"return_1q": 0.1}
    fundamental = {"net_profit_margin": 0.2}
    m, f = apply_feature_scope(market, fundamental, "market_only")
    assert m == {"return_1q": 0.1}
    assert f == {"net_profit_margin": None}


def test_apply_feature_scope_fundamental_only_nulls_market():
    market = {"return_1q": 0.1}
    fundamental = {"net_profit_margin": 0.2}
    m, f = apply_feature_scope(market, fundamental, "fundamental_only")
    assert m == {"return_1q": None}
    assert f == {"net_profit_margin": 0.2}


def test_apply_feature_scope_all_keeps_both_unchanged():
    market = {"return_1q": 0.1}
    fundamental = {"net_profit_margin": 0.2}
    m, f = apply_feature_scope(market, fundamental, "all")
    assert m == market
    assert f == fundamental
