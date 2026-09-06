"""22 model features (11 market + 11 fundamental). documentation .pdf
section 12: 'exact formulas ... should be taken from those actual
artifacts, not from generic examples' -- that artifact does not exist in
this repo, so this module documents the concrete formulas used instead,
per the user's explicit approval to use standard finance definitions."""
import pandas as pd

MARKET_FEATURE_NAMES = [
    "return_1q", "return_2q", "return_1y",
    "momentum_3m", "momentum_6m",
    "ma_20_ratio", "ma_50_ratio", "ma_200_ratio",
    "volatility_60d", "volatility_120d",
    "volume_change_60d",
]

FUNDAMENTAL_FEATURE_NAMES = [
    "revenue_growth_yoy",
    "net_profit_margin",
    "operating_margin",
    "return_on_equity",
    "return_on_assets",
    "debt_to_equity",
    "current_ratio",
    "quick_ratio_proxy",
    "asset_turnover",
    "price_to_book",
    "price_to_earnings_proxy",
]


def _window(prices: pd.DataFrame, as_of: pd.Timestamp, days: int) -> pd.DataFrame:
    start = as_of - pd.Timedelta(days=days)
    return prices[(prices["date"] > start) & (prices["date"] <= as_of)]


def _price_n_days_before(prices: pd.DataFrame, as_of: pd.Timestamp, days: int, price_col: str):
    target = as_of - pd.Timedelta(days=days)
    eligible = prices[prices["date"] <= target]
    if eligible.empty:
        return None
    return float(eligible.iloc[-1][price_col])


def market_features(prices: pd.DataFrame, as_of: pd.Timestamp, price_col: str) -> dict:
    prices = prices.sort_values("date")
    current = _price_n_days_before(prices, as_of + pd.Timedelta(days=1), 0, price_col)

    def ret(days):
        past = _price_n_days_before(prices, as_of, days, price_col)
        if past is None or current is None or past == 0:
            return None
        return (current / past) - 1

    def ma_ratio(days):
        w = _window(prices, as_of, days)
        if w.empty or current is None:
            return None
        ma = w[price_col].mean()
        return None if ma == 0 else current / ma - 1

    def volatility(days):
        w = _window(prices, as_of, days)
        if len(w) < 2:
            return None
        daily_ret = w[price_col].pct_change().dropna()
        return None if daily_ret.empty else float(daily_ret.std())

    def volume_change(days):
        if "volume" not in prices.columns:
            return None
        w = _window(prices, as_of, days)
        half = len(w) // 2
        if half == 0:
            return None
        recent, prior = w["volume"].iloc[half:], w["volume"].iloc[:half]
        if prior.mean() == 0:
            return None
        return float(recent.mean() / prior.mean() - 1)

    return {
        "return_1q": ret(91),
        "return_2q": ret(182),
        "return_1y": ret(365),
        "momentum_3m": ret(91),
        "momentum_6m": ret(182),
        "ma_20_ratio": ma_ratio(20),
        "ma_50_ratio": ma_ratio(50),
        "ma_200_ratio": ma_ratio(200),
        "volatility_60d": volatility(60),
        "volatility_120d": volatility(120),
        "volume_change_60d": volume_change(60),
    }


def _safe_div(a, b):
    if a is None or b is None or b == 0:
        return None
    return a / b


def fundamental_features(row: dict) -> dict:
    revenue = row.get("total_revenue")
    prior_revenue = row.get("prior_revenue")
    net_income = row.get("net_income")
    operating_income = row.get("operating_income")
    total_assets = row.get("total_assets")
    total_liabilities = row.get("total_liabilities")
    total_equity = row.get("total_equity")
    current_assets = row.get("current_assets")
    current_liabilities = row.get("current_liabilities")
    market_cap = row.get("market_cap")
    total_debt = row.get("total_debt")

    return {
        "revenue_growth_yoy": _safe_div(
            None if revenue is None or prior_revenue is None else revenue - prior_revenue,
            prior_revenue,
        ),
        "net_profit_margin": _safe_div(net_income, revenue),
        "operating_margin": _safe_div(operating_income, revenue),
        "return_on_equity": _safe_div(net_income, total_equity),
        "return_on_assets": _safe_div(net_income, total_assets),
        "debt_to_equity": _safe_div(total_debt, total_equity),
        "current_ratio": _safe_div(current_assets, current_liabilities),
        "quick_ratio_proxy": _safe_div(current_assets, current_liabilities),
        "asset_turnover": _safe_div(revenue, total_assets),
        "price_to_book": _safe_div(market_cap, total_equity),
        "price_to_earnings_proxy": _safe_div(market_cap, net_income),
    }
