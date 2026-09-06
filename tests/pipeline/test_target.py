import pandas as pd
from pipeline.lib.target import last_trading_day_price, compute_binary_target

def test_last_trading_day_price_uses_most_recent_prior_trading_day():
    prices = pd.DataFrame({
        "date": pd.to_datetime(["2024-03-27", "2024-03-28", "2024-04-01"]),
        "close": [10.0, 11.0, 12.0],
    })
    # 2024-03-31 is a Sunday, no trading that day -> should use 2024-03-28
    assert last_trading_day_price(prices, pd.Timestamp("2024-03-31"), "close") == 11.0

def test_compute_binary_target_positive_return():
    assert compute_binary_target(price_t=100.0, price_t_plus_1=110.0) == 1

def test_compute_binary_target_zero_or_negative_return():
    assert compute_binary_target(price_t=100.0, price_t_plus_1=100.0) == 0
    assert compute_binary_target(price_t=100.0, price_t_plus_1=90.0) == 0
