# sensitivity/lib/dataset_builder.py
"""Config-driven dataset construction for the sensitivity sweep. Reuses
pipeline.lib.* and pipeline._04_features_target's financial-parsing
internals -- see this plan's Task 3 context for why importing "private"
helpers from that module is the intentional, DRY choice here."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from pipeline._04_features_target import (
    _load_stock_clean,
    _normalize_financial_long,
    _pivot_fundamentals,
    latest_available_financial_row,
)
from pipeline.lib.companies import COMPANIES
from pipeline.lib.features import (
    FUNDAMENTAL_FEATURE_NAMES,
    MARKET_FEATURE_NAMES,
    fundamental_features,
    market_features,
)
from pipeline.lib.target import last_trading_day_price

PIPELINE_OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "pipeline" / "outputs"
STOCK_CLEAN_DIR = PIPELINE_OUTPUTS_DIR / "stock_clean"


def period_ends_in_range(start: pd.Timestamp, end: pd.Timestamp, freq: str) -> list[pd.Timestamp]:
    """Generalizes pipeline._03_build_panel.quarter_ends_in_range to any
    pandas offset alias -- "QE" reproduces the exact quarterly behavior,
    "ME" gives monthly, "W-FRI" gives weekly (Friday-anchored)."""
    return list(pd.date_range(start=start, end=end, freq=freq))


def compute_target_with_threshold(price_t: float, price_t1: float, mode: str, deadzone_pct: float):
    """Returns (target, keep). mode="simple" reproduces
    pipeline.lib.target.compute_binary_target's >0 rule exactly (keep is
    always True). mode="deadzone" drops rows whose absolute return is
    smaller than deadzone_pct, treating only clear moves as signal."""
    if price_t <= 0:
        raise ValueError(f"price_t must be positive, got {price_t}")
    ret = (price_t1 / price_t) - 1

    if mode == "simple":
        return (1 if ret > 0 else 0), True

    if mode == "deadzone":
        if abs(ret) < deadzone_pct:
            return None, False
        return (1 if ret >= deadzone_pct else 0), True

    raise ValueError(f"unknown threshold mode: {mode}")


def apply_feature_scope(market: dict, fundamental: dict, scope: str):
    """Returns (market, fundamental) dicts, nulling out the excluded
    category so the training stage's existing all-NaN-column-drop logic
    naturally excludes it -- no special-casing needed downstream."""
    if scope == "all":
        return market, fundamental
    if scope == "market_only":
        return market, {k: None for k in fundamental}
    if scope == "fundamental_only":
        return {k: None for k in market}, fundamental
    raise ValueError(f"unknown feature scope: {scope}")


def build_dataset_for_config(config: dict, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """fundamentals is the already-pivoted table from _pivot_fundamentals()
    -- computed once in run_all.py and shared across all 12 configs, since
    it doesn't depend on any config value except lag (handled per-row via
    lag_days_override)."""
    lag_override = {"quarterly": config["quarterly_lag_days"], "annual": config["annual_lag_days"]}
    out_rows = []

    for company in COMPANIES:
        folder = company["folder"]
        stock_path = STOCK_CLEAN_DIR / f"{folder}.csv"
        if not stock_path.exists():
            continue
        stock = _load_stock_clean(stock_path)
        if stock.empty:
            continue

        periods = period_ends_in_range(stock["date"].min(), stock["date"].max(), config["horizon_freq"])
        fin_rows = fundamentals[fundamentals["company_folder"] == folder]

        for i, period_end in enumerate(periods[:-1]):
            next_period_end = periods[i + 1]
            price_t = last_trading_day_price(stock, period_end, "close")
            price_t1 = last_trading_day_price(stock, next_period_end, "close")

            target, keep = compute_target_with_threshold(
                price_t, price_t1, config["threshold_mode"], config["deadzone_pct"]
            )
            if not keep:
                continue

            market = market_features(stock, as_of=period_end, price_col="close")
            fin_row = latest_available_financial_row(fin_rows, period_end, folder, lag_days_override=lag_override)
            fundamental = (
                fundamental_features(fin_row.to_dict()) if fin_row is not None
                else {name: None for name in FUNDAMENTAL_FEATURE_NAMES}
            )
            market, fundamental = apply_feature_scope(market, fundamental, config["feature_scope"])

            row = {"company_folder": folder, "period_end": period_end}
            row.update(market)
            row.update(fundamental)
            row["target"] = target
            out_rows.append(row)

    columns = ["company_folder", "period_end"] + MARKET_FEATURE_NAMES + FUNDAMENTAL_FEATURE_NAMES + ["target"]
    if not out_rows:
        return pd.DataFrame(columns=columns)
    # Sort to match the real pipeline's row order exactly: pipeline._04_features_target.main()
    # iterates via panel.groupby("company_folder") (pandas default sort=True -> alphabetical)
    # then ascending quarter_end per company, whereas the loop above iterates COMPANIES in
    # its own (non-alphabetical) order. Same cell values either way, but row order matters
    # for order-sensitive estimators like RandomForestClassifier's positional bootstrap
    # sampling under a fixed random_state -- without this sort, random_forest's accuracy on
    # the base config diverges from pipeline/outputs/metrics.json even though every other
    # model matches bit-for-bit.
    return pd.DataFrame(out_rows)[columns].sort_values(["company_folder", "period_end"]).reset_index(drop=True)


def build_fundamentals_table() -> pd.DataFrame:
    """Loads and pivots pipeline/outputs/raw_financial.parquet once, shared
    across every config's build_dataset_for_config() call."""
    financial = pd.read_parquet(PIPELINE_OUTPUTS_DIR / "raw_financial.parquet")
    financial_long = _normalize_financial_long(financial)
    fundamentals = _pivot_fundamentals(financial_long)
    fundamentals["publication_date"] = pd.NaT
    return fundamentals
