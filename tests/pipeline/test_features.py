import pandas as pd
from pipeline.lib.features import market_features, fundamental_features, MARKET_FEATURE_NAMES, FUNDAMENTAL_FEATURE_NAMES

def _sample_daily_prices(n=300, start=100.0):
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    prices = pd.Series(start + pd.RangeIndex(n) * 0.1)
    volume = pd.Series([1_000_000] * n)
    return pd.DataFrame({"date": dates, "close": prices.values, "volume": volume.values})

def test_market_feature_count_is_eleven():
    assert len(MARKET_FEATURE_NAMES) == 11

def test_fundamental_feature_count_is_eleven():
    assert len(FUNDAMENTAL_FEATURE_NAMES) == 11

def test_market_features_returns_all_expected_keys():
    prices = _sample_daily_prices()
    feats = market_features(prices, as_of=prices["date"].iloc[-1], price_col="close")
    assert set(feats.keys()) == set(MARKET_FEATURE_NAMES)
    for name in MARKET_FEATURE_NAMES:
        assert feats[name] is None or isinstance(feats[name], float)

def test_fundamental_features_returns_all_expected_keys():
    row = {
        "total_revenue": 1000.0, "prior_revenue": 900.0,
        "net_income": 100.0, "operating_income": 150.0,
        "total_assets": 5000.0, "total_liabilities": 2000.0,
        "total_equity": 3000.0, "current_assets": 1200.0,
        "current_liabilities": 600.0, "market_cap": 8000.0,
        "total_debt": 1800.0,
    }
    feats = fundamental_features(row)
    assert set(feats.keys()) == set(FUNDAMENTAL_FEATURE_NAMES)
    assert feats["net_profit_margin"] == 100.0 / 1000.0
    assert feats["current_ratio"] == 1200.0 / 600.0
    assert feats["debt_to_equity"] == 1800.0 / 3000.0
