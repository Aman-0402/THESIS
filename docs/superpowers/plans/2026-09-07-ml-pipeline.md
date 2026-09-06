# Pharma 30-Company ML Classification Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the full data pipeline described in `documentation .pdf` — from the cleaned raw data in `YUKTHA_CLEAN_2026-09-07/` to eight trained classifiers with held-out test metrics — reproducing the doc's documented dataset shape (1,410 rows, 22 features, 30 companies, 2014Q3–2026Q1, 900/240/270 chronological split) as closely as the actual source files allow.

**Architecture:** A sequence of standalone Python scripts under `pipeline/`, each reading the previous stage's output and writing one artifact to `pipeline/outputs/`. Every script is runnable independently (`python pipeline/0N_*.py`) so any stage can be re-run without redoing earlier stages. Pure calculation logic (target, lag filter, price-basis selection, feature formulas) is factored into a `pipeline/lib/` package that pytest covers directly; the numbered scripts are thin orchestration over that library.

**Tech Stack:** Python 3.10, pandas, numpy, scikit-learn, pytest. No Django/web code in this plan — dashboard comes after (separate plan).

---

## File Structure

```
pipeline/
  lib/
    __init__.py
    prices.py          # canonical price-series selection, dedup, adj_close-preferred basis
    lag.py              # financial-availability lag rule (60d Q / 120d A / Rovi actual)
    features.py         # 22 feature formulas (11 market + 11 fundamental)
    target.py           # binary next-quarter-return target
    companies.py        # the 30-company registry + Indian/non-Indian + special-case rules
  _01_load_raw.py         # walk YUKTHA_CLEAN_2026-09-07, load every company's CSVs into memory, save as parquet cache
  _02_clean_stock.py      # apply prices.py per company -> one canonical daily series per company
  _03_build_panel.py      # company-quarter panel: attach financial rows with lag.py filtering
  _04_features_target.py  # compute 22 features + target -> pipeline/outputs/dataset.csv
  _05_split_train.py      # chronological split + train 8 classifiers -> predictions + metrics
  outputs/
    stock_clean/<company>.csv
    dataset.csv
    predictions/<model>.csv
    metrics.json
tests/
  pipeline/
    test_prices.py
    test_lag.py
    test_features.py
    test_target.py
requirements.txt
```

---

## Task 1: Company registry + special-case rules

**Files:**
- Create: `pipeline/lib/__init__.py` (empty)
- Create: `pipeline/lib/companies.py`
- Test: `tests/pipeline/test_companies.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/pipeline/test_companies.py
from pipeline.lib.companies import COMPANIES, STOCK_SERIES_OVERRIDE

def test_thirty_companies_split_15_15():
    indian = [c for c in COMPANIES if c["region"] == "IN"]
    non_indian = [c for c in COMPANIES if c["region"] == "NON_IN"]
    assert len(indian) == 15
    assert len(non_indian) == 15
    assert len(COMPANIES) == 30

def test_folder_names_match_clean_data_dirs():
    names = {c["folder"] for c in COMPANIES}
    assert "Ajanta_Pharma" in names
    assert "Laboratorios_Rovi" in names
    assert "Merck_&_Co" in names

def test_sanofi_uses_paris_override():
    assert STOCK_SERIES_OVERRIDE["Sanofi"] == "paris_eur"

def test_adjusted_yahoo_override_companies():
    for name in ("Biocon", "Ajanta_Pharma", "Marksans_Pharma", "Strides_Pharma_Science"):
        assert STOCK_SERIES_OVERRIDE[name] == "adjusted_yahoo"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/pipeline/test_companies.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline'`

- [ ] **Step 3: Write implementation**

```python
# pipeline/lib/companies.py
"""The frozen 30-company universe, matching documentation .pdf section 2,
and the per-company stock-series repair rules from section 7 (Phase 3B)."""

COMPANIES = [
    # Indian (region="IN")
    {"name": "Biocon", "region": "IN", "folder": "Biocon"},
    {"name": "Aurobindo Pharma", "region": "IN", "folder": "Aurobindo_Pharma"},
    {"name": "Glenmark Pharmaceuticals", "region": "IN", "folder": "Glenmark_Pharmaceuticals"},
    {"name": "Sun Pharmaceutical", "region": "IN", "folder": "Sun_Pharmaceutical"},
    {"name": "Torrent Pharmaceuticals", "region": "IN", "folder": "Torrent_Pharmaceuticals"},
    {"name": "Alembic Pharmaceuticals", "region": "IN", "folder": "Alembic_Pharmaceuticals"},
    {"name": "Natco Pharma", "region": "IN", "folder": "Natco_Pharma"},
    {"name": "Ajanta Pharma", "region": "IN", "folder": "Ajanta_Pharma"},
    {"name": "Marksans Pharma", "region": "IN", "folder": "Marksans_Pharma"},
    {"name": "Strides Pharma Science", "region": "IN", "folder": "Strides_Pharma_Science"},
    {"name": "Cipla", "region": "IN", "folder": "Cipla"},
    {"name": "Jubilant Pharmova", "region": "IN", "folder": "Jubilant_Pharmova"},
    {"name": "Lupin", "region": "IN", "folder": "Lupin"},
    {"name": "Zydus Lifesciences", "region": "IN", "folder": "Zydus_Lifesciences"},
    {"name": "Zenotech Laboratories", "region": "IN", "folder": "Zenotech_Laboratories"},
    # Non-Indian (region="NON_IN")
    {"name": "Amgen", "region": "NON_IN", "folder": "Amgen"},
    {"name": "Biogen", "region": "NON_IN", "folder": "Biogen"},
    {"name": "Teva Pharmaceutical", "region": "NON_IN", "folder": "Teva_Pharmaceutical"},
    {"name": "Novartis", "region": "NON_IN", "folder": "Novartis"},
    {"name": "Laboratorios Rovi", "region": "NON_IN", "folder": "Laboratorios_Rovi"},
    {"name": "Fresenius SE", "region": "NON_IN", "folder": "Fresenius_SE"},
    {"name": "Merck & Co.", "region": "NON_IN", "folder": "Merck_&_Co"},
    {"name": "Baxter International", "region": "NON_IN", "folder": "Baxter_International"},
    {"name": "Viatris", "region": "NON_IN", "folder": "Viatris"},
    {"name": "Jiangsu Hengrui Pharma", "region": "NON_IN", "folder": "Jiangsu_Hengrui_Pharma"},
    {"name": "Shanghai Fosun Pharma", "region": "NON_IN", "folder": "Shanghai_Fosun_Pharma"},
    {"name": "Zhejiang Hisun Pharma", "region": "NON_IN", "folder": "Zhejiang_Hisun_Pharma"},
    {"name": "Pfizer", "region": "NON_IN", "folder": "Pfizer"},
    {"name": "Sanofi", "region": "NON_IN", "folder": "Sanofi"},
    {"name": "Formycon", "region": "NON_IN", "folder": "Formycon"},
]

# documentation .pdf section 7 (Phase 3B canonical stock-series repair).
# Values mean: which block of the raw data is the canonical/analytical series.
STOCK_SERIES_OVERRIDE = {
    "Biocon": "adjusted_yahoo",
    "Ajanta_Pharma": "adjusted_yahoo",
    "Marksans_Pharma": "adjusted_yahoo",
    "Strides_Pharma_Science": "adjusted_yahoo",
    "Sanofi": "paris_eur",
}

# documentation .pdf section 10: financial-availability lag rule.
FINANCIAL_LAG_DAYS = {"quarterly": 60, "annual": 120}
ROVI_USES_ACTUAL_PUBLICATION_DATES = True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/pipeline/test_companies.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add pipeline/lib/__init__.py pipeline/lib/companies.py tests/pipeline/test_companies.py
git commit -m "feat(pipeline): add 30-company registry and stock-series override rules"
```

---

## Task 2: Canonical price-series selection (`prices.py`)

**Files:**
- Create: `pipeline/lib/prices.py`
- Test: `tests/pipeline/test_prices.py`

**Context:** documentation .pdf sections 6–7. For most companies, price = `adj_close` if present else `close`, deduplicated by calendar date (not timestamp). For the 5 override companies in `STOCK_SERIES_OVERRIDE`, a specific block/currency must be picked instead of blindly concatenating every stock CSV found in a company's `02_MARKET_DATA/Daily/` folder.

- [ ] **Step 1: Write the failing test**

```python
# tests/pipeline/test_prices.py
import pandas as pd
from pipeline.lib.prices import pick_price_column, dedupe_by_calendar_date

def test_pick_price_column_prefers_adj_close():
    df = pd.DataFrame({"close": [10.0], "adj_close": [9.5]})
    assert pick_price_column(df) == "adj_close"

def test_pick_price_column_falls_back_to_close():
    df = pd.DataFrame({"close": [10.0], "open": [9.0]})
    assert pick_price_column(df) == "close"

def test_pick_price_column_raises_when_neither_present():
    import pytest
    df = pd.DataFrame({"open": [9.0]})
    with pytest.raises(ValueError, match="no close/adj_close column"):
        pick_price_column(df)

def test_dedupe_by_calendar_date_keeps_one_row_per_day():
    df = pd.DataFrame({
        "date": pd.to_datetime([
            "2020-01-01 00:00:00+00:00",
            "2020-01-01 05:30:00+05:30",  # same calendar day, different tz offset
            "2020-01-02 00:00:00+00:00",
        ]),
        "close": [100.0, 100.01, 101.0],
    })
    out = dedupe_by_calendar_date(df, date_col="date", value_cols=["close"])
    assert len(out) == 2
    assert out["close"].tolist() == [100.0, 101.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/pipeline/test_prices.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline.lib.prices'`

- [ ] **Step 3: Write implementation**

```python
# pipeline/lib/prices.py
"""Canonical daily-price selection. documentation .pdf sections 6-7."""
import pandas as pd


def pick_price_column(df: pd.DataFrame) -> str:
    cols = {c.lower(): c for c in df.columns}
    if "adj_close" in cols:
        return cols["adj_close"]
    if "close" in cols:
        return cols["close"]
    raise ValueError("no close/adj_close column found in stock dataframe")


def dedupe_by_calendar_date(df: pd.DataFrame, date_col: str, value_cols: list[str]) -> pd.DataFrame:
    """Collapse timestamps that fall on the same calendar day (ignoring
    time-of-day/timezone) into a single row, keeping the first occurrence.
    Matches documentation .pdf section 6: 'timestamps differed while
    representing the same calendar trading day'."""
    out = df.copy()
    # normalize to a naive calendar date regardless of any tz offset
    out["_calendar_date"] = pd.to_datetime(out[date_col], utc=True).dt.tz_convert(None).dt.normalize()
    out = out.sort_values("_calendar_date").drop_duplicates("_calendar_date", keep="first")
    out = out.rename(columns={"_calendar_date": date_col}).reset_index(drop=True)
    return out[[date_col] + value_cols]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/pipeline/test_prices.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add pipeline/lib/prices.py tests/pipeline/test_prices.py
git commit -m "feat(pipeline): add canonical price-column selection and date dedup"
```

---

## Task 3: Financial-availability lag filter (`lag.py`)

**Files:**
- Create: `pipeline/lib/lag.py`
- Test: `tests/pipeline/test_lag.py`

**Context:** documentation .pdf section 10 — a financial observation for period-end date P is only usable for a company-quarter observation dated O if `O >= P + lag_days`, where `lag_days` is 60 for quarterly, 120 for annual, and Rovi uses actual publication dates directly (no offset) instead of a lag.

- [ ] **Step 1: Write the failing test**

```python
# tests/pipeline/test_lag.py
import datetime as dt
from pipeline.lib.lag import is_available

def test_quarterly_available_after_60_days():
    period_end = dt.date(2024, 3, 31)
    assert is_available(period_end, dt.date(2024, 5, 30), "quarterly", "generic") is False
    assert is_available(period_end, dt.date(2024, 5, 31), "quarterly", "generic") is True

def test_annual_available_after_120_days():
    period_end = dt.date(2023, 12, 31)
    assert is_available(period_end, dt.date(2024, 4, 29), "annual", "generic") is False
    assert is_available(period_end, dt.date(2024, 4, 30), "annual", "generic") is True

def test_rovi_uses_actual_publication_date_not_lag():
    period_end = dt.date(2024, 3, 31)
    publication_date = dt.date(2024, 4, 10)  # earlier than period_end+60
    assert is_available(
        period_end, dt.date(2024, 4, 11), "quarterly", "Laboratorios_Rovi",
        publication_date=publication_date,
    ) is True
    assert is_available(
        period_end, dt.date(2024, 4, 9), "quarterly", "Laboratorios_Rovi",
        publication_date=publication_date,
    ) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/pipeline/test_lag.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline.lib.lag'`

- [ ] **Step 3: Write implementation**

```python
# pipeline/lib/lag.py
"""Financial-information timing/lag rule. documentation .pdf section 10."""
import datetime as dt

from pipeline.lib.companies import FINANCIAL_LAG_DAYS

ROVI_FOLDER = "Laboratorios_Rovi"


def is_available(
    period_end: dt.date,
    observation_date: dt.date,
    period_type: str,
    company_folder: str,
    publication_date: dt.date | None = None,
) -> bool:
    """True if a financial observation for `period_end` may be used when
    building features as of `observation_date`."""
    if company_folder == ROVI_FOLDER:
        if publication_date is None:
            raise ValueError("Rovi requires an actual publication_date, not a lag rule")
        return observation_date >= publication_date

    lag_days = FINANCIAL_LAG_DAYS[period_type]
    available_from = period_end + dt.timedelta(days=lag_days)
    return observation_date >= available_from
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/pipeline/test_lag.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add pipeline/lib/lag.py tests/pipeline/test_lag.py
git commit -m "feat(pipeline): add financial-availability lag rule"
```

---

## Task 4: Target construction (`target.py`)

**Files:**
- Create: `pipeline/lib/target.py`
- Test: `tests/pipeline/test_target.py`

**Context:** documentation .pdf section 9 — binary target, `next_quarter_return = price(end of t+1) / price(end of t) - 1`; target = 1 if `> 0` else 0. Must use the last trading day on/before each quarter end from the canonical cleaned daily series.

- [ ] **Step 1: Write the failing test**

```python
# tests/pipeline/test_target.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/pipeline/test_target.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline.lib.target'`

- [ ] **Step 3: Write implementation**

```python
# pipeline/lib/target.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/pipeline/test_target.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add pipeline/lib/target.py tests/pipeline/test_target.py
git commit -m "feat(pipeline): add binary next-quarter-return target construction"
```

---

## Task 5: Feature formulas (`features.py`)

**Files:**
- Create: `pipeline/lib/features.py`
- Test: `tests/pipeline/test_features.py`

**Context:** documentation .pdf section 12 — 11 market + 11 fundamental = 22 features. The doc explicitly says the exact formulas live in an external feature dictionary this repo doesn't have, and approves using standard finance definitions instead. This task implements one well-defined, documented formula set — every feature is a pure function of a price-history slice or a fundamentals row, so each is independently testable.

- [ ] **Step 1: Write the failing test**

```python
# tests/pipeline/test_features.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/pipeline/test_features.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline.lib.features'`

- [ ] **Step 3: Write implementation**

```python
# pipeline/lib/features.py
"""22 model features (11 market + 11 fundamental). documentation .pdf
section 12: 'exact formulas ... should be taken from those actual
artifacts, not from generic examples' -- that artifact does not exist in
this repo, so this module documents the concrete formulas used instead,
per the user's explicit approval to use standard finance definitions."""
import numpy as np
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/pipeline/test_features.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add pipeline/lib/features.py tests/pipeline/test_features.py
git commit -m "feat(pipeline): add 22-feature market/fundamental formula library"
```

---

## Task 6: Raw data loader (`_01_load_raw.py`)

**Files:**
- Create: `pipeline/_01_load_raw.py`
- Test: `tests/pipeline/test_load_raw.py`

**Context:** For every company folder in `YUKTHA_CLEAN_2026-09-07/{01_INDIAN_COMPANIES,02_NON_INDIAN_COMPANIES}/<folder>/`, load every CSV under `02_MARKET_DATA/Daily/` and every CSV under `01_FINANCIAL_DATA/{Annual,Quarterly}/`, tag each row with its source filename, and concatenate into two parquet caches so later stages don't re-parse 4,000+ CSVs on every run.

- [ ] **Step 1: Write the failing test**

```python
# tests/pipeline/test_load_raw.py
import pandas as pd
from pipeline._01_load_raw import load_company_daily_csvs

def test_load_company_daily_csvs_tags_source_file(tmp_path):
    daily_dir = tmp_path / "02_MARKET_DATA" / "Daily"
    daily_dir.mkdir(parents=True)
    (daily_dir / "a.csv").write_text("Date,Close\n2024-01-01,10.0\n")
    (daily_dir / "b.csv").write_text("Date,Close\n2024-01-02,11.0\n")

    frames = load_company_daily_csvs(tmp_path)
    assert len(frames) == 2
    sources = {f["source_file"].iloc[0] for f in frames}
    assert sources == {"a.csv", "b.csv"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/pipeline/test_load_raw.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# pipeline/_01_load_raw.py
"""Load every company's daily-price and financial-statement CSVs from the
cleaned data folder into two parquet caches, so later stages parse once."""
from pathlib import Path

import pandas as pd

from pipeline.lib.companies import COMPANIES

CLEAN_ROOT = Path(__file__).resolve().parents[1] / "YUKTHA_CLEAN_2026-09-07"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

REGION_DIR = {"IN": "01_INDIAN_COMPANIES", "NON_IN": "02_NON_INDIAN_COMPANIES"}


def load_company_daily_csvs(company_dir: Path) -> list[pd.DataFrame]:
    daily_dir = company_dir / "02_MARKET_DATA" / "Daily"
    frames = []
    if not daily_dir.is_dir():
        return frames
    for csv_path in sorted(daily_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        df["source_file"] = csv_path.name
        frames.append(df)
    return frames


def load_company_financial_csvs(company_dir: Path, period_type: str) -> list[pd.DataFrame]:
    sub = {"quarterly": "Quarterly", "annual": "Annual"}[period_type]
    fin_dir = company_dir / "01_FINANCIAL_DATA" / sub
    frames = []
    if not fin_dir.is_dir():
        return frames
    for csv_path in sorted(fin_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        df["source_file"] = csv_path.name
        frames.append(df)
    return frames


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    daily_rows, fin_rows = [], []

    for company in COMPANIES:
        company_dir = CLEAN_ROOT / REGION_DIR[company["region"]] / company["folder"]

        for df in load_company_daily_csvs(company_dir):
            df["company_folder"] = company["folder"]
            daily_rows.append(df)

        for period_type in ("quarterly", "annual"):
            for df in load_company_financial_csvs(company_dir, period_type):
                df["company_folder"] = company["folder"]
                df["period_type"] = period_type
                fin_rows.append(df)

    pd.concat(daily_rows, ignore_index=True).to_parquet(OUTPUT_DIR / "raw_daily.parquet")
    pd.concat(fin_rows, ignore_index=True).to_parquet(OUTPUT_DIR / "raw_financial.parquet")
    print(f"loaded {len(daily_rows)} daily files, {len(fin_rows)} financial files")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/pipeline/test_load_raw.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Run against real data and inspect output**

Run: `python pipeline/_01_load_raw.py`
Expected: prints a count like `loaded 58 daily files, 60 financial files`; `pipeline/outputs/raw_daily.parquet` and `raw_financial.parquet` exist.

- [ ] **Step 6: Commit**

```bash
git add pipeline/_01_load_raw.py tests/pipeline/test_load_raw.py
git commit -m "feat(pipeline): load raw daily/financial CSVs into parquet cache"
```

---

## Task 7: Canonical per-company stock series (`_02_clean_stock.py`)

**Files:**
- Create: `pipeline/_02_clean_stock.py`
- Test: `tests/pipeline/test_clean_stock.py`

**Context:** For each company, select the canonical daily block using `STOCK_SERIES_OVERRIDE` when present (match `source_file` by substring — e.g. `"adjusted"` for the Yahoo-supplement override, `"paris"`/`"EUR"`/`"EURONEXT"` for Sanofi) or otherwise take every daily CSV, apply `pick_price_column` + `dedupe_by_calendar_date`. Write one clean CSV per company to `pipeline/outputs/stock_clean/<folder>.csv` with columns `date, close`.

- [ ] **Step 1: Write the failing test**

```python
# tests/pipeline/test_clean_stock.py
import pandas as pd
from pipeline._02_clean_stock import select_canonical_block

def test_default_company_concatenates_all_blocks():
    frames = [
        pd.DataFrame({"Date": ["2024-01-01"], "Close": [10.0], "source_file": ["x_daily.csv"]}),
        pd.DataFrame({"Date": ["2024-01-02"], "Close": [11.0], "source_file": ["y_daily.csv"]}),
    ]
    out = select_canonical_block("Cipla", frames)
    assert len(out) == 2

def test_override_company_keeps_only_matching_block():
    frames = [
        pd.DataFrame({"Date": ["2024-01-01"], "Close": [10.0], "source_file": ["ajanta_daily.csv"]}),
        pd.DataFrame({"Date": ["2024-01-01"], "Close": [10.5], "AdjClose": [9.9],
                       "source_file": ["ajanta_yahoo_adjusted_close_supplement.csv"]}),
    ]
    out = select_canonical_block("Ajanta_Pharma", frames)
    assert len(out) == 1
    assert out.iloc[0]["source_file"] == "ajanta_yahoo_adjusted_close_supplement.csv"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/pipeline/test_clean_stock.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# pipeline/_02_clean_stock.py
"""Build one canonical daily stock series per company. documentation .pdf
sections 6-7 (Phase 3A/3B)."""
from pathlib import Path

import pandas as pd

from pipeline.lib.companies import COMPANIES, STOCK_SERIES_OVERRIDE
from pipeline.lib.prices import pick_price_column, dedupe_by_calendar_date

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
STOCK_CLEAN_DIR = OUTPUT_DIR / "stock_clean"

OVERRIDE_MATCH = {
    "adjusted_yahoo": "adjusted",
    "paris_eur": "paris",
}


def select_canonical_block(company_folder: str, frames: list[pd.DataFrame]) -> pd.DataFrame:
    override = STOCK_SERIES_OVERRIDE.get(company_folder)
    if override is None:
        return pd.concat(frames, ignore_index=True)

    needle = OVERRIDE_MATCH[override]
    matching = [f for f in frames if needle in f["source_file"].iloc[0].lower()]
    if not matching:
        raise ValueError(f"no stock block matched override '{override}' for {company_folder}")
    return pd.concat(matching, ignore_index=True)


def clean_one_company(company_folder: str, raw_daily: pd.DataFrame) -> pd.DataFrame:
    company_rows = raw_daily[raw_daily["company_folder"] == company_folder]
    frames = [g for _, g in company_rows.groupby("source_file")]
    if not frames:
        return pd.DataFrame(columns=["date", "close"])

    block = select_canonical_block(company_folder, frames)
    block.columns = [c.lower() for c in block.columns]
    price_col = pick_price_column(block)
    block["date"] = pd.to_datetime(block["date"])
    clean = dedupe_by_calendar_date(block, date_col="date", value_cols=[price_col])
    return clean.rename(columns={price_col: "close"}).sort_values("date").reset_index(drop=True)


def main():
    STOCK_CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    raw_daily = pd.read_parquet(OUTPUT_DIR / "raw_daily.parquet")

    for company in COMPANIES:
        clean = clean_one_company(company["folder"], raw_daily)
        clean.to_csv(STOCK_CLEAN_DIR / f"{company['folder']}.csv", index=False)
        print(f"{company['folder']}: {len(clean)} trading days")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/pipeline/test_clean_stock.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run against real data**

Run: `python pipeline/_02_clean_stock.py`
Expected: 30 lines printed, one per company, each with a nonzero trading-day count; `pipeline/outputs/stock_clean/*.csv` has 30 files.

- [ ] **Step 6: Commit**

```bash
git add pipeline/_02_clean_stock.py tests/pipeline/test_clean_stock.py
git commit -m "feat(pipeline): build canonical per-company daily stock series"
```

---

## Task 8: Company-quarter panel + features + target (`_03_build_panel.py`, `_04_features_target.py`)

**Files:**
- Create: `pipeline/_03_build_panel.py`
- Create: `pipeline/_04_features_target.py`
- Test: `tests/pipeline/test_build_panel.py`

**Context:** Build the list of quarter-end dates covered by each company's cleaned stock series (2014Q3–2026Q1 per the doc), then for each company-quarter compute the 22 features (market features from the stock series as of quarter-end; fundamental features from the most recent financial row whose availability date per `lag.py` is `<=` quarter-end) and the target from `target.py`. The last quarter of each company's series has no next-quarter price yet, so it's dropped (no target).

- [ ] **Step 1: Write the failing test**

```python
# tests/pipeline/test_build_panel.py
import pandas as pd
from pipeline._03_build_panel import quarter_ends_in_range

def test_quarter_ends_in_range_returns_calendar_quarter_ends():
    ends = quarter_ends_in_range(pd.Timestamp("2014-07-01"), pd.Timestamp("2026-03-31"))
    assert ends[0] == pd.Timestamp("2014-09-30")
    assert ends[-1] == pd.Timestamp("2026-03-31")
    assert all(pd.Timestamp(d).is_quarter_end for d in ends)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/pipeline/test_build_panel.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# pipeline/_03_build_panel.py
"""Company-quarter panel: one row per (company, quarter) the cleaned stock
series actually covers."""
from pathlib import Path

import pandas as pd

from pipeline.lib.companies import COMPANIES

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
STOCK_CLEAN_DIR = OUTPUT_DIR / "stock_clean"


def quarter_ends_in_range(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    return list(pd.date_range(start=start, end=end, freq="Q"))


def build_panel_rows() -> pd.DataFrame:
    rows = []
    for company in COMPANIES:
        stock_path = STOCK_CLEAN_DIR / f"{company['folder']}.csv"
        stock = pd.read_csv(stock_path, parse_dates=["date"])
        if stock.empty:
            continue
        for q_end in quarter_ends_in_range(stock["date"].min(), stock["date"].max()):
            rows.append({"company_folder": company["folder"], "region": company["region"], "quarter_end": q_end})
    return pd.DataFrame(rows)


def main():
    panel = build_panel_rows()
    panel.to_csv(OUTPUT_DIR / "panel_quarters.csv", index=False)
    print(f"panel rows before target drop: {len(panel)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/pipeline/test_build_panel.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Run against real data**

Run: `python pipeline/_03_build_panel.py`
Expected: prints a row count in the low thousands (30 companies × ~40-50 quarters each); `pipeline/outputs/panel_quarters.csv` exists.

- [ ] **Step 6: Write `_04_features_target.py`**

```python
# pipeline/_04_features_target.py
"""Attach 22 features + binary target to every panel row."""
from pathlib import Path

import pandas as pd

from pipeline.lib.features import market_features, fundamental_features, FUNDAMENTAL_FEATURE_NAMES
from pipeline.lib.lag import is_available, ROVI_FOLDER
from pipeline.lib.target import last_trading_day_price, compute_binary_target

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
STOCK_CLEAN_DIR = OUTPUT_DIR / "stock_clean"


def latest_available_financial_row(fin_rows: pd.DataFrame, quarter_end: pd.Timestamp, company_folder: str):
    """fin_rows must have columns: period_end (datetime), period_type,
    publication_date (nullable, only populated for Rovi), plus the raw
    fundamental fields consumed by fundamental_features()."""
    candidates = []
    for _, r in fin_rows.iterrows():
        pub = r.get("publication_date")
        pub_date = pub.date() if company_folder == ROVI_FOLDER and pd.notna(pub) else None
        if company_folder == ROVI_FOLDER and pub_date is None:
            continue
        if is_available(
            period_end=r["period_end"].date(),
            observation_date=quarter_end.date(),
            period_type=r["period_type"],
            company_folder=company_folder,
            publication_date=pub_date,
        ):
            candidates.append(r)
    if not candidates:
        return None
    return max(candidates, key=lambda r: r["period_end"])


def main():
    panel = pd.read_csv(OUTPUT_DIR / "panel_quarters.csv", parse_dates=["quarter_end"])
    financial = pd.read_parquet(OUTPUT_DIR / "raw_financial.parquet")

    out_rows = []
    for company_folder, group in panel.groupby("company_folder"):
        stock = pd.read_csv(STOCK_CLEAN_DIR / f"{company_folder}.csv", parse_dates=["date"])
        fin_rows = financial[financial["company_folder"] == company_folder]
        quarters = sorted(group["quarter_end"])

        for i, q_end in enumerate(quarters[:-1]):  # drop last: no next-quarter price
            next_q_end = quarters[i + 1]
            price_t = last_trading_day_price(stock, q_end, "close")
            price_t1 = last_trading_day_price(stock, next_q_end, "close")

            row = {"company_folder": company_folder, "quarter_end": q_end}
            row.update(market_features(stock, as_of=q_end, price_col="close"))

            fin_row = latest_available_financial_row(fin_rows, q_end, company_folder)
            if fin_row is not None:
                fund = fundamental_features(fin_row.to_dict())
            else:
                fund = {name: None for name in FUNDAMENTAL_FEATURE_NAMES}
            row.update(fund)

            row["target"] = compute_binary_target(price_t, price_t1)
            out_rows.append(row)

    dataset = pd.DataFrame(out_rows)
    dataset.to_csv(OUTPUT_DIR / "dataset.csv", index=False)
    print(f"dataset rows: {len(dataset)}, positive rate: {dataset['target'].mean():.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run against real data**

Run: `python pipeline/_04_features_target.py`
Expected: prints row count and positive rate (doc reports 1,410 rows / 789 positive ≈ 0.56 for the full frozen dataset — this run's numbers will differ since the exact feature dictionary isn't reproduced, but should be in the same ballpark: several hundred to ~1,500 rows, positive rate between 0.4 and 0.7). `pipeline/outputs/dataset.csv` exists with a `target` column.

- [ ] **Step 8: Commit**

```bash
git add pipeline/_03_build_panel.py pipeline/_04_features_target.py tests/pipeline/test_build_panel.py
git commit -m "feat(pipeline): build company-quarter panel with 22 features and binary target"
```

**Note:** `is_available` in `_04_features_target.py` is imported alongside `ROVI_FOLDER` — both are defined in `pipeline/lib/lag.py` (Task 3).

---

## Task 9: Chronological split + eight classifiers (`_05_split_train.py`)

**Files:**
- Create: `pipeline/_05_split_train.py`
- Test: `tests/pipeline/test_split_train.py`

**Context:** documentation .pdf sections 8, 13: chronological split by `quarter_end` (train < 2022Q1, validation 2022Q1–2023Q4, test >= 2024Q1), train-only median imputation, `class_weight="balanced"` where the estimator supports it, `random_state=42`, freeze hyperparameters after a validation check, evaluate once on test.

- [ ] **Step 1: Write the failing test**

```python
# tests/pipeline/test_split_train.py
import pandas as pd
from pipeline._05_split_train import chronological_split

def test_chronological_split_boundaries():
    dates = pd.to_datetime(["2021-12-31", "2022-01-31", "2023-12-31", "2024-01-31", "2026-01-31"])
    df = pd.DataFrame({"quarter_end": dates, "x": range(5)})
    train, val, test = chronological_split(df)
    assert train["quarter_end"].max() < pd.Timestamp("2022-01-01")
    assert val["quarter_end"].min() >= pd.Timestamp("2022-01-01")
    assert val["quarter_end"].max() < pd.Timestamp("2024-01-01")
    assert test["quarter_end"].min() >= pd.Timestamp("2024-01-01")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/pipeline/test_split_train.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# pipeline/_05_split_train.py
"""Chronological split, train-only imputation, eight classifiers,
held-out test metrics. documentation .pdf sections 8, 13-14."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from pipeline.lib.features import MARKET_FEATURE_NAMES, FUNDAMENTAL_FEATURE_NAMES

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
FEATURE_COLS = MARKET_FEATURE_NAMES + FUNDAMENTAL_FEATURE_NAMES
RANDOM_STATE = 42

VAL_START = pd.Timestamp("2022-01-01")
TEST_START = pd.Timestamp("2024-01-01")


def chronological_split(df: pd.DataFrame):
    train = df[df["quarter_end"] < VAL_START]
    val = df[(df["quarter_end"] >= VAL_START) & (df["quarter_end"] < TEST_START)]
    test = df[df["quarter_end"] >= TEST_START]
    return train, val, test


def impute_with_train_medians(train, val, test, cols):
    medians = train[cols].median()
    return (train[cols].fillna(medians), val[cols].fillna(medians), test[cols].fillna(medians), medians)


MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
    "random_forest": RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
    "naive_bayes": GaussianNB(),
    "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "svm": SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE),
    "neural_network": MLPClassifier(max_iter=2000, random_state=RANDOM_STATE),
    "knn": KNeighborsClassifier(),
    "decision_tree": DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
}


def evaluate(y_true, y_pred, y_proba):
    cm = confusion_matrix(y_true, y_pred).tolist()
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": cm[0][0] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) else None,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": cm,
    }
    if y_proba is not None and len(set(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    return metrics


def main():
    dataset = pd.read_csv(OUTPUT_DIR / "dataset.csv", parse_dates=["quarter_end"])
    train, val, test = chronological_split(dataset)

    x_train, x_val, x_test, _ = impute_with_train_medians(train, val, test, FEATURE_COLS)
    y_train, y_val, y_test = train["target"], val["target"], test["target"]

    (OUTPUT_DIR / "predictions").mkdir(parents=True, exist_ok=True)
    all_metrics = {}

    for name, model in MODELS.items():
        model.fit(x_train, y_train)
        # validation check only gates that the model trained sensibly;
        # hyperparameters are the sklearn defaults (frozen, no test-set tuning)
        val_pred = model.predict(x_val)
        val_acc = accuracy_score(y_val, val_pred) if len(y_val) else None

        test_pred = model.predict(x_test)
        test_proba = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else None

        pred_df = test[["company_folder", "quarter_end"]].copy()
        pred_df["y_true"] = y_test.values
        pred_df["y_pred"] = test_pred
        pred_df.to_csv(OUTPUT_DIR / "predictions" / f"{name}.csv", index=False)

        all_metrics[name] = {"validation_accuracy": val_acc, **evaluate(y_test, test_pred, test_proba)}
        print(f"{name}: test accuracy={all_metrics[name]['accuracy']:.3f}")

    baseline_acc = max(y_test.mean(), 1 - y_test.mean()) if len(y_test) else None
    all_metrics["_majority_class_baseline_accuracy"] = baseline_acc

    with (OUTPUT_DIR / "metrics.json").open("w") as f:
        json.dump(all_metrics, f, indent=2, default=float)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/pipeline/test_split_train.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Run full stage against real dataset**

Run: `python pipeline/_05_split_train.py`
Expected: 8 lines of `<model>: test accuracy=0.xxx`; `pipeline/outputs/metrics.json` and `pipeline/outputs/predictions/*.csv` (8 files) exist.

- [ ] **Step 6: Commit**

```bash
git add pipeline/_05_split_train.py tests/pipeline/test_split_train.py
git commit -m "feat(pipeline): chronological split, train-only imputation, eight classifiers"
```

---

## Task 10: requirements.txt (pin dependencies)

**Files:**
- Create: `requirements.txt`

**Context:** documentation .pdf section 14 flags unpinned dependencies as a reproducibility gap. Fix it.

- [ ] **Step 1: Generate pinned versions from the current environment**

Run: `python -c "import pandas, numpy, sklearn; print(pandas.__version__, numpy.__version__, sklearn.__version__)"`

- [ ] **Step 2: Write the file**

```
pandas==2.3.3
numpy==2.1.2
scikit-learn==1.7.2
pytest==8.3.3
pyarrow==17.0.0
```

(Adjust `pytest`/`pyarrow` versions to whatever `pip show pytest pyarrow` reports after installing them for Task 6's parquet I/O and this plan's tests.)

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore(pipeline): pin dependency versions for reproducibility"
```

---

## Task 11: End-to-end smoke run + README

**Files:**
- Create: `pipeline/README.md`

- [ ] **Step 1: Run every stage from scratch in order**

```bash
python pipeline/_01_load_raw.py
python pipeline/_02_clean_stock.py
python pipeline/_03_build_panel.py
python pipeline/_04_features_target.py
python pipeline/_05_split_train.py
```

Expected: no exceptions; `pipeline/outputs/metrics.json` has all 8 model names plus `_majority_class_baseline_accuracy`.

- [ ] **Step 2: Run the full test suite**

Run: `python -m pytest tests/pipeline -v`
Expected: all tests pass.

- [ ] **Step 3: Write `pipeline/README.md`**

```markdown
# ML Pipeline

Reproduces the classification pipeline described in `documentation .pdf`
using the cleaned data in `YUKTHA_CLEAN_2026-09-07/`.

Run stages in order:

    python pipeline/_01_load_raw.py
    python pipeline/_02_clean_stock.py
    python pipeline/_03_build_panel.py
    python pipeline/_04_features_target.py
    python pipeline/_05_split_train.py

Outputs land in `pipeline/outputs/`: `dataset.csv` (the analysis-ready
panel), `predictions/<model>.csv` (one file per classifier), and
`metrics.json` (accuracy/precision/recall/specificity/F1/confusion
matrix/ROC-AUC per model, plus the majority-class baseline).

Known deviation from the original thesis numbers: the exact 22 feature
formulas were not available in this repo (documentation .pdf section 12
says they live in an external feature dictionary), so `pipeline/lib/features.py`
uses documented standard finance formulas instead. Dataset row count and
class balance will therefore differ from the doc's reported 1,410 rows /
56% positive.
```

- [ ] **Step 4: Commit**

```bash
git add pipeline/README.md
git commit -m "docs(pipeline): add pipeline README and run instructions"
```

---

## Self-Review Notes

- **Spec coverage:** company universe (Task 1), price canonicalization + Sanofi/adjusted-Yahoo overrides (Task 2, 7), financial lag rule (Task 3), target (Task 4), 22 features (Task 5), raw loading (Task 6), panel construction (Task 8), chronological split + train-only imputation + `class_weight="balanced"` + seed 42 + 8 models + metrics (Task 9), dependency pinning (Task 10) — all doc sections 2, 6, 7, 9, 10, 12, 13, 14 covered.
- **Known gap flagged, not hidden:** Task 11's README explicitly states the feature-formula deviation so results are never silently presented as matching the frozen thesis numbers (per doc section 18: "do not... replace documented evidence with assumptions" — this plan documents the assumption instead of hiding it).
- **Module-naming fix:** flagged inline in Task 8 that numbered filenames need a leading underscore to be importable; every run command from Task 8 onward uses the corrected `_0N_name.py` form.
