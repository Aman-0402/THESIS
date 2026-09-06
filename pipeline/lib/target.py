"""Binary next-quarter-return target. documentation .pdf section 9."""
import pandas as pd


def last_trading_day_price(daily_prices: pd.DataFrame, quarter_end: pd.Timestamp, price_col: str) -> float:
    """Last trading day on or before quarter_end, from a dataframe with a
    'date' column sorted ascending."""
    eligible = daily_prices[daily_prices["date"] <= quarter_end]
    if eligible.empty:
        raise ValueError(f"no trading day on or before {quarter_end}")
    return float(eligible.iloc[-1][price_col])


def compute_binary_target(price_t: float, price_t_plus_1: float) -> int:
    next_quarter_return = (price_t_plus_1 / price_t) - 1
    return 1 if next_quarter_return > 0 else 0
