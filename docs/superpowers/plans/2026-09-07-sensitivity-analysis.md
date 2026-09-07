# Sensitivity Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a one-factor-at-a-time sensitivity sweep (12 runs: 1 base + 11 variants across financial-lag, missing-data-handling, up/down-threshold, feature-scope, and horizon/frequency dimensions) reusing the existing `pipeline/` code, plus a `/sensitivity/` dashboard page showing whether "no model beats baseline" holds across all variants. Per `docs/superpowers/specs/2026-09-07-sensitivity-analysis-design.md`.

**Architecture:** New `sensitivity/` package parallel to `pipeline/`. Two small, additive, backward-compatible parameters added to already-frozen pipeline code (`lag.is_available`, `_04_features_target.latest_available_financial_row`) so lag can be overridden without duplicating logic. Everything else in `sensitivity/` reuses `pipeline.lib.*` and imports specific helper functions from `pipeline._04_features_target`/`pipeline._05_split_train` rather than re-deriving ~250 lines of financial-format parsing and 8-classifier setup. `pipeline/outputs/` is never touched.

**Tech Stack:** Same as `pipeline/` — pandas, scikit-learn, pytest. No new dependencies.

---

## File Structure

```
sensitivity/
  __init__.py
  configs.py              # 12 RunConfig dicts
  lib/
    __init__.py
    dataset_builder.py     # period_ends_in_range, dead-zone target, feature-scope nulling,
                            # build_dataset_for_config() (reuses pipeline._04_features_target internals)
    train_evaluate.py       # missing-data-strategy variants + train_and_evaluate()
                             # (reuses pipeline._05_split_train.MODELS/evaluate)
  run_all.py                # loops all 12 configs, writes outputs + summary.json
  outputs/
    <run_id>/dataset.csv
    <run_id>/metrics.json
    summary.json
tests/
  sensitivity/
    test_configs.py
    test_dataset_builder.py
    test_train_evaluate.py
```

Dashboard additions (existing `dashboard/resultsboard/`):
```
resultsboard/data.py                                    # + load_sensitivity_summary()
resultsboard/views.py                                    # + sensitivity view
resultsboard/urls.py                                      # + /sensitivity/ route
resultsboard/templates/resultsboard/sensitivity.html      # new page
resultsboard/templates/resultsboard/base.html             # + nav link
```

---

## Task 1: Additive lag-override parameter on frozen pipeline code

**Files:**
- Modify: `pipeline/lib/lag.py`
- Modify: `pipeline/_04_features_target.py`
- Test: `tests/pipeline/test_lag.py` (add one test)

**Context:** `is_available()` currently reads `FINANCIAL_LAG_DAYS` from `pipeline.lib.companies` directly with no way to use a different lag value. Add an optional override, defaulting to `None` so every existing call site and test is completely unaffected.

- [ ] **Step 1: Write the failing test**

Add to `tests/pipeline/test_lag.py`:

```python
def test_lag_days_override_replaces_default_financial_lag_days():
    period_end = dt.date(2024, 3, 31)
    # default (60d) would make 2024-04-15 unavailable; a 10-day override makes it available
    assert is_available(
        period_end, dt.date(2024, 4, 15), "quarterly", "generic",
        lag_days_override={"quarterly": 10, "annual": 20},
    ) is True
    # default 60d lag still applies when no override given
    assert is_available(period_end, dt.date(2024, 4, 15), "quarterly", "generic") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/pipeline/test_lag.py -v`
Expected: FAIL — `TypeError: is_available() got an unexpected keyword argument 'lag_days_override'`

- [ ] **Step 3: Modify `is_available`**

In `pipeline/lib/lag.py`, change the signature and lag lookup:

```python
def is_available(
    period_end: dt.date,
    observation_date: dt.date,
    period_type: str,
    company_folder: str,
    publication_date: dt.date | None = None,
    lag_days_override: dict[str, int] | None = None,
) -> bool:
    """True if a financial observation for `period_end` may be used when
    building features as of `observation_date`. `lag_days_override`, when
    given, replaces the module's FINANCIAL_LAG_DAYS for this call only --
    used by sensitivity/ to test alternative lag assumptions without
    duplicating this function."""
    if company_folder == ROVI_FOLDER:
        if publication_date is None:
            raise ValueError("Rovi requires an actual publication_date, not a lag rule")
        return observation_date >= publication_date

    lag_days_table = lag_days_override if lag_days_override is not None else FINANCIAL_LAG_DAYS
    lag_days = lag_days_table[period_type]
    available_from = period_end + dt.timedelta(days=lag_days)
    return observation_date > available_from
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/pipeline/test_lag.py -v`
Expected: PASS (4 tests: 3 original + 1 new)

- [ ] **Step 5: Thread the same override through `latest_available_financial_row`**

In `pipeline/_04_features_target.py`, change:

```python
def latest_available_financial_row(fin_rows: pd.DataFrame, quarter_end: pd.Timestamp, company_folder: str):
```

to:

```python
def latest_available_financial_row(
    fin_rows: pd.DataFrame,
    quarter_end: pd.Timestamp,
    company_folder: str,
    lag_days_override: dict[str, int] | None = None,
):
```

and inside, change the `is_available(...)` call to pass `lag_days_override=lag_days_override`. Leave everything else in the function (including the Rovi dead-code comment) unchanged. The real pipeline's `main()` calls this function with no `lag_days_override` argument, so its behavior is completely unchanged.

- [ ] **Step 6: Run full pipeline test suite and the real pipeline stage to confirm zero behavior change**

Run: `python -m pytest tests/pipeline -v` — expect all passing (39 tests: 38 previous + 1 new).
Run: `python pipeline/_04_features_target.py` — expect identical output to before (`dataset rows: 3439, positive rate: 0.587`, same per-company fundamental coverage lines).

- [ ] **Step 7: Commit**

```bash
git add pipeline/lib/lag.py pipeline/_04_features_target.py tests/pipeline/test_lag.py
git commit -m "feat(pipeline): add optional lag_days_override for sensitivity analysis"
```

---

## Task 2: Run configs

**Files:**
- Create: `sensitivity/__init__.py` (empty)
- Create: `sensitivity/configs.py`
- Test: `tests/sensitivity/test_configs.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/sensitivity/test_configs.py
from sensitivity.configs import RUN_CONFIGS

REQUIRED_KEYS = {
    "run_id", "description", "dimension", "value_label",
    "quarterly_lag_days", "annual_lag_days", "missing_strategy",
    "threshold_mode", "deadzone_pct", "feature_scope", "horizon_freq",
}


def test_twelve_configs_total():
    assert len(RUN_CONFIGS) == 12


def test_exactly_one_base_config():
    base = [c for c in RUN_CONFIGS if c["dimension"] == "base"]
    assert len(base) == 1
    base_config = base[0]
    assert base_config["quarterly_lag_days"] == 60
    assert base_config["annual_lag_days"] == 120
    assert base_config["missing_strategy"] == "impute"
    assert base_config["threshold_mode"] == "simple"
    assert base_config["feature_scope"] == "all"
    assert base_config["horizon_freq"] == "QE"


def test_all_configs_have_required_keys_and_unique_ids():
    ids = set()
    for config in RUN_CONFIGS:
        assert REQUIRED_KEYS.issubset(config.keys())
        assert config["run_id"] not in ids
        ids.add(config["run_id"])


def test_dimension_counts_match_design():
    from collections import Counter
    counts = Counter(c["dimension"] for c in RUN_CONFIGS)
    assert counts["base"] == 1
    assert counts["lag"] == 4
    assert counts["missing_strategy"] == 2
    assert counts["threshold"] == 1
    assert counts["feature_scope"] == 2
    assert counts["horizon"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/sensitivity/test_configs.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sensitivity'`

- [ ] **Step 3: Write `sensitivity/configs.py`**

```python
# sensitivity/configs.py
"""One-factor-at-a-time run configs for the sensitivity sweep. See
docs/superpowers/specs/2026-09-07-sensitivity-analysis-design.md."""

BASE = {
    "run_id": "base",
    "description": "Base case: current pipeline defaults",
    "dimension": "base",
    "value_label": "current pipeline defaults",
    "quarterly_lag_days": 60,
    "annual_lag_days": 120,
    "missing_strategy": "impute",
    "threshold_mode": "simple",
    "deadzone_pct": 0.0,
    "feature_scope": "all",
    "horizon_freq": "QE",
}


def _variant(run_id: str, description: str, dimension: str, value_label: str, **overrides) -> dict:
    config = dict(BASE)
    config.update(run_id=run_id, description=description, dimension=dimension, value_label=value_label)
    config.update(overrides)
    return config


RUN_CONFIGS = [
    BASE,
    _variant("lag_30", "Financial lag: 30/60 days", "lag", "30d quarterly / 60d annual",
             quarterly_lag_days=30, annual_lag_days=60),
    _variant("lag_45", "Financial lag: 45/90 days", "lag", "45d quarterly / 90d annual",
             quarterly_lag_days=45, annual_lag_days=90),
    _variant("lag_90", "Financial lag: 90/180 days", "lag", "90d quarterly / 180d annual",
             quarterly_lag_days=90, annual_lag_days=180),
    _variant("lag_120", "Financial lag: 120/240 days", "lag", "120d quarterly / 240d annual",
             quarterly_lag_days=120, annual_lag_days=240),
    _variant("missing_drop_any", "Missing data: drop rows with any missing feature",
             "missing_strategy", "drop rows with any missing feature",
             missing_strategy="drop_any"),
    _variant("missing_drop_majority", "Missing data: drop rows >50% missing, impute rest",
             "missing_strategy", "drop rows >50% missing, impute rest",
             missing_strategy="drop_majority"),
    _variant("threshold_deadzone", "Up/down rule: +/-1% dead zone",
             "threshold", "dead zone: |return| < 1% excluded",
             threshold_mode="deadzone", deadzone_pct=0.01),
    _variant("features_market_only", "Feature scope: market features only",
             "feature_scope", "market-only",
             feature_scope="market_only"),
    _variant("features_fundamental_only", "Feature scope: fundamental features only",
             "feature_scope", "fundamental-only",
             feature_scope="fundamental_only"),
    _variant("horizon_monthly", "Horizon: monthly panel", "horizon", "monthly",
             horizon_freq="ME"),
    _variant("horizon_weekly", "Horizon: weekly panel", "horizon", "weekly",
             horizon_freq="W-FRI"),
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/sensitivity/test_configs.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add sensitivity/__init__.py sensitivity/configs.py tests/sensitivity/test_configs.py
git commit -m "feat(sensitivity): add one-factor-at-a-time run configs"
```

---

## Task 3: Dataset builder (period frequency, dead-zone target, feature-scope nulling)

**Files:**
- Create: `sensitivity/lib/__init__.py` (empty)
- Create: `sensitivity/lib/dataset_builder.py`
- Test: `tests/sensitivity/test_dataset_builder.py`

**Context:** Reuses `pipeline.lib.companies.COMPANIES`, `pipeline.lib.features.{market_features,fundamental_features,MARKET_FEATURE_NAMES,FUNDAMENTAL_FEATURE_NAMES}`, `pipeline.lib.target.{last_trading_day_price,compute_binary_target}`, and — to avoid re-deriving ~150 lines of financial-statement-format parsing — imports the private helpers `_load_stock_clean`, `_normalize_financial_long`, `_pivot_fundamentals`, and the now-parameterized `latest_available_financial_row` directly from `pipeline._04_features_target`. This is intentional, explicit reuse of another module's "private" functions (Python allows `from module import _name`), not a design smell — the alternative is duplicating well-tested, non-trivial parsing logic.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/sensitivity/test_dataset_builder.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sensitivity.lib.dataset_builder'`

- [ ] **Step 3: Write `sensitivity/lib/dataset_builder.py`**

```python
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
    return pd.DataFrame(out_rows)[columns]


def build_fundamentals_table() -> pd.DataFrame:
    """Loads and pivots pipeline/outputs/raw_financial.parquet once, shared
    across every config's build_dataset_for_config() call."""
    financial = pd.read_parquet(PIPELINE_OUTPUTS_DIR / "raw_financial.parquet")
    financial_long = _normalize_financial_long(financial)
    fundamentals = _pivot_fundamentals(financial_long)
    fundamentals["publication_date"] = pd.NaT
    return fundamentals
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/sensitivity/test_dataset_builder.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Smoke-test against real data for the base config**

Run this ad-hoc check (not a permanent script, just verify manually):
```bash
python -c "
from sensitivity.configs import RUN_CONFIGS
from sensitivity.lib.dataset_builder import build_dataset_for_config, build_fundamentals_table
base = [c for c in RUN_CONFIGS if c['dimension'] == 'base'][0]
fundamentals = build_fundamentals_table()
df = build_dataset_for_config(base, fundamentals)
print(len(df), df['target'].mean())
"
```
Expected: `3439 0.5865...` — should exactly match `pipeline/outputs/dataset.csv`'s row count and positive rate, since the base config reproduces the real pipeline's own settings. If the numbers don't match exactly, investigate before proceeding — the base config must be a faithful reproduction of the real pipeline.

- [ ] **Step 6: Commit**

```bash
git add sensitivity/lib/__init__.py sensitivity/lib/dataset_builder.py tests/sensitivity/test_dataset_builder.py
git commit -m "feat(sensitivity): add config-driven dataset builder"
```

---

## Task 4: Train/evaluate with missing-data-strategy variants

**Files:**
- Create: `sensitivity/lib/train_evaluate.py`
- Test: `tests/sensitivity/test_train_evaluate.py`

**Context:** Reuses `pipeline._05_split_train.{MODELS, evaluate, chronological_split, RANDOM_STATE, VAL_START, TEST_START}` directly (same 8 classifiers, same split boundaries, same metric computation) rather than redefining them. Only the missing-data handling before fitting is new, config-driven logic.

- [ ] **Step 1: Write the failing test**

```python
# tests/sensitivity/test_train_evaluate.py
import pandas as pd

from sensitivity.lib.train_evaluate import apply_missing_strategy


def _sample_df():
    return pd.DataFrame({
        "a": [1.0, None, 3.0, 4.0],
        "b": [10.0, 20.0, None, 40.0],
        "target": [1, 0, 1, 0],
    })


def test_impute_strategy_fills_with_train_median_keeps_all_rows():
    train = _sample_df()
    out_train, medians = apply_missing_strategy(train, ["a", "b"], "impute")
    assert len(out_train) == 4
    assert out_train["a"].isna().sum() == 0
    assert medians["a"] == 3.0  # median of [1,3,4] (train's own non-null values)


def test_drop_any_strategy_removes_rows_with_any_missing_feature():
    train = _sample_df()
    out_train, medians = apply_missing_strategy(train, ["a", "b"], "drop_any")
    assert len(out_train) == 2  # rows 0 and 3 have no missing values
    assert out_train["a"].isna().sum() == 0


def test_drop_majority_strategy_drops_rows_missing_over_half_features_then_imputes_rest():
    df = pd.DataFrame({
        "a": [1.0, None, None, 4.0],
        "b": [10.0, None, 30.0, 40.0],
        "target": [1, 0, 1, 0],
    })
    # row 1 is missing both of 2 features (100% > 50%) -> dropped
    # row 2 is missing 1 of 2 features (50%, not over 50%) -> kept, imputed
    out_train, medians = apply_missing_strategy(df, ["a", "b"], "drop_majority")
    assert len(out_train) == 3
    assert out_train["a"].isna().sum() == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/sensitivity/test_train_evaluate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sensitivity.lib.train_evaluate'`

- [ ] **Step 3: Write `sensitivity/lib/train_evaluate.py`**

```python
# sensitivity/lib/train_evaluate.py
"""Config-driven missing-data handling, reusing pipeline._05_split_train's
classifiers/metrics/split logic directly."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from pipeline._05_split_train import MODELS, chronological_split, evaluate


def apply_missing_strategy(df: pd.DataFrame, feature_cols: list[str], strategy: str):
    """Returns (df_with_features_imputed_or_filtered, medians_used).
    Operates on ONE split (call separately for train/val/test, always
    computing medians from the TRAIN split and reusing them for val/test --
    see train_and_evaluate() below)."""
    if strategy == "impute":
        medians = df[feature_cols].median()
        out = df.copy()
        out[feature_cols] = out[feature_cols].fillna(medians)
        return out, medians

    if strategy == "drop_any":
        out = df.dropna(subset=feature_cols).copy()
        return out, None

    if strategy == "drop_majority":
        missing_frac = df[feature_cols].isna().mean(axis=1)
        kept = df[missing_frac <= 0.5].copy()
        medians = kept[feature_cols].median()
        kept[feature_cols] = kept[feature_cols].fillna(medians)
        return kept, medians

    raise ValueError(f"unknown missing strategy: {strategy}")


def train_and_evaluate(dataset: pd.DataFrame, config: dict) -> dict:
    """Full train/val/test pipeline for one sensitivity config's dataset.
    dataset must have a "period_end" column (renamed here to "quarter_end"
    to reuse pipeline._05_split_train.chronological_split unchanged, since
    that function's split boundaries are date-based and don't care what the
    column is literally named as long as it's called quarter_end)."""
    df = dataset.rename(columns={"period_end": "quarter_end"})
    feature_cols = [c for c in df.columns if c not in ("company_folder", "quarter_end", "target")]

    train, val, test = chronological_split(df)

    all_nan_cols = [c for c in feature_cols if train[c].isna().all()]
    active_cols = [c for c in feature_cols if c not in all_nan_cols]

    train_processed, medians = apply_missing_strategy(train, active_cols, config["missing_strategy"])
    if medians is not None:
        val_processed = val.copy()
        val_processed[active_cols] = val_processed[active_cols].fillna(medians)
        test_processed = test.copy()
        test_processed[active_cols] = test_processed[active_cols].fillna(medians)
    else:
        # drop_any: no medians to reuse, so val/test also drop incomplete rows
        val_processed = val.dropna(subset=active_cols).copy()
        test_processed = test.dropna(subset=active_cols).copy()

    x_train, y_train = train_processed[active_cols], train_processed["target"]
    x_val, y_val = val_processed[active_cols], val_processed["target"]
    x_test, y_test = test_processed[active_cols], test_processed["target"]

    if len(x_test) == 0 or x_test[active_cols].isna().any().any() or y_test.nunique() < 1:
        return {
            "run_id": config["run_id"], "row_count": len(df),
            "train_rows": len(x_train), "val_rows": len(x_val), "test_rows": len(x_test),
            "active_feature_count": len(active_cols), "dropped_feature_count": len(all_nan_cols),
            "models": {}, "baseline_accuracy": None, "best_model": None, "best_accuracy": None,
            "note": "test set too small/degenerate for this config to evaluate",
        }

    # Reuses the same MODELS instances across all 12 configs (run_all.py
    # calls train_and_evaluate once per config, sequentially). Each
    # model.fit() call fully overwrites that estimator's previously fitted
    # state -- safe to re-fit sklearn estimators repeatedly like this, and
    # avoids duplicating the 8-classifier definitions from pipeline/_05_split_train.py.
    model_metrics = {}
    for name, model in MODELS.items():
        model.fit(x_train, y_train)
        test_pred = model.predict(x_test)
        test_proba = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else None
        model_metrics[name] = evaluate(y_test, test_pred, test_proba)

    baseline_accuracy = max(y_test.mean(), 1 - y_test.mean())
    best_name = max(model_metrics, key=lambda n: model_metrics[n]["accuracy"])

    return {
        "run_id": config["run_id"],
        "row_count": len(df),
        "train_rows": len(x_train), "val_rows": len(x_val), "test_rows": len(x_test),
        "active_feature_count": len(active_cols), "dropped_feature_count": len(all_nan_cols),
        "models": model_metrics,
        "baseline_accuracy": baseline_accuracy,
        "best_model": best_name,
        "best_accuracy": model_metrics[best_name]["accuracy"],
        "beats_baseline": model_metrics[best_name]["accuracy"] > baseline_accuracy,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/sensitivity/test_train_evaluate.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add sensitivity/lib/train_evaluate.py tests/sensitivity/test_train_evaluate.py
git commit -m "feat(sensitivity): add missing-data-strategy variants and train/evaluate"
```

---

## Task 5: Run all 12 configs, aggregate summary

**Files:**
- Create: `sensitivity/run_all.py`

- [ ] **Step 1: Write `sensitivity/run_all.py`**

```python
# sensitivity/run_all.py
"""Runs all 12 sensitivity configs and writes sensitivity/outputs/summary.json."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sensitivity.configs import RUN_CONFIGS
from sensitivity.lib.dataset_builder import build_dataset_for_config, build_fundamentals_table
from sensitivity.lib.train_evaluate import train_and_evaluate

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fundamentals = build_fundamentals_table()

    summary = []
    for config in RUN_CONFIGS:
        run_dir = OUTPUT_DIR / config["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)

        dataset = build_dataset_for_config(config, fundamentals)
        dataset.to_csv(run_dir / "dataset.csv", index=False)

        result = train_and_evaluate(dataset, config)
        with (run_dir / "metrics.json").open("w") as f:
            json.dump(result, f, indent=2, default=float)

        summary.append({
            "run_id": config["run_id"],
            "description": config["description"],
            "dimension": config["dimension"],
            "value_label": config["value_label"],
            "row_count": result["row_count"],
            "best_model": result["best_model"],
            "best_accuracy": result["best_accuracy"],
            "baseline_accuracy": result["baseline_accuracy"],
            "beats_baseline": result.get("beats_baseline"),
        })
        print(f"{config['run_id']}: rows={result['row_count']} "
              f"best={result['best_model']} acc={result['best_accuracy']} "
              f"baseline={result['baseline_accuracy']}")

    with (OUTPUT_DIR / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nWrote summary for {len(summary)} runs to {OUTPUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run against real data**

Run: `python sensitivity/run_all.py`

Expected: 12 lines of per-run output, no exceptions. This will take real time (weekly/monthly horizon variants build much larger panels — expect the full run to take several minutes, not seconds; that's normal, not a bug). Report every line of actual output.

Verify the `base` run's numbers match `pipeline/outputs/dataset.csv`/`metrics.json` exactly (same row count 3439, same best model, same accuracy) — this is the critical sanity check that the sensitivity harness faithfully reproduces the real pipeline before trusting any of its variant results.

- [ ] **Step 3: Inspect `sensitivity/outputs/summary.json`**

Confirm it has exactly 12 entries, one per `run_id` in `sensitivity/configs.py`, each with `best_accuracy` and `baseline_accuracy` populated (or a clear `note` explaining why not, for any degenerate run).

- [ ] **Step 4: Commit**

```bash
git add sensitivity/run_all.py
git commit -m "feat(sensitivity): add run-all script producing aggregated summary"
```

(`sensitivity/outputs/` should NOT be committed — add it to `.gitignore` alongside `pipeline/outputs/` in this same commit if not already covered by a broader pattern.)

---

## Task 6: Sensitivity dashboard page

**Files:**
- Modify: `dashboard/resultsboard/data.py`
- Modify: `dashboard/resultsboard/views.py`
- Modify: `dashboard/resultsboard/urls.py`
- Modify: `dashboard/resultsboard/templates/resultsboard/base.html`
- Create: `dashboard/resultsboard/templates/resultsboard/sensitivity.html`
- Test: `tests/resultsboard/test_data.py` (add tests)

- [ ] **Step 1: Add `.gitignore` entry if needed**

Confirm `sensitivity/outputs/` is gitignored (add `sensitivity/outputs/` to the root `.gitignore` if the existing `pipeline/outputs/` line doesn't already cover it via a broader pattern — check the current `.gitignore` content first).

- [ ] **Step 2: Write failing tests**

Add to `tests/resultsboard/test_data.py`:

```python
def test_load_sensitivity_summary_returns_runs_list(tmp_path):
    import json
    from resultsboard.data import load_sensitivity_summary

    d = tmp_path / "sensitivity_outputs"
    d.mkdir()
    summary = [
        {"run_id": "base", "description": "Base case", "dimension": "base",
         "value_label": "current pipeline defaults", "row_count": 3439,
         "best_model": "neural_network", "best_accuracy": 0.575,
         "baseline_accuracy": 0.579, "beats_baseline": False},
        {"run_id": "lag_30", "description": "Financial lag: 30/60 days", "dimension": "lag",
         "value_label": "30d quarterly / 60d annual", "row_count": 3439,
         "best_model": "knn", "best_accuracy": 0.60,
         "baseline_accuracy": 0.579, "beats_baseline": True},
    ]
    (d / "summary.json").write_text(json.dumps(summary))

    runs = load_sensitivity_summary(d)
    assert len(runs) == 2
    assert runs[0]["run_id"] == "base"
    assert runs[1]["beats_baseline"] is True


def test_load_sensitivity_summary_missing_raises_clear_error(tmp_path):
    import pytest
    from resultsboard.data import PipelineOutputsMissing, load_sensitivity_summary

    with pytest.raises(PipelineOutputsMissing):
        load_sensitivity_summary(tmp_path / "does_not_exist")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/resultsboard/test_data.py -v`
Expected: FAIL — `ImportError: cannot import name 'load_sensitivity_summary'`

- [ ] **Step 4: Add `load_sensitivity_summary` to `data.py`**

Append to `dashboard/resultsboard/data.py`:

```python
def load_sensitivity_summary(sensitivity_outputs_dir: Path) -> list[dict]:
    summary_path = sensitivity_outputs_dir / "summary.json"
    _require(summary_path)
    return json.loads(summary_path.read_text())
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/resultsboard/test_data.py -v`
Expected: PASS (all tests, including the 2 new ones)

- [ ] **Step 6: Add the view**

In `dashboard/resultsboard/views.py`, add:

```python
SENSITIVITY_OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "sensitivity" / "outputs"


def sensitivity(request):
    def build_context():
        runs = data.load_sensitivity_summary(SENSITIVITY_OUTPUTS_DIR)
        base = next((r for r in runs if r["dimension"] == "base"), None)
        return {
            "runs": runs,
            "base": base,
            "chart_labels": [r["run_id"] for r in runs],
            "chart_best": [r["best_accuracy"] for r in runs],
            "chart_baseline": [r["baseline_accuracy"] for r in runs],
        }

    return _render_or_missing(request, "resultsboard/sensitivity.html", build_context)
```

- [ ] **Step 7: Add the URL**

In `dashboard/resultsboard/urls.py`:

```python
    path("sensitivity/", views.sensitivity, name="sensitivity"),
```

- [ ] **Step 8: Add the nav link**

In `dashboard/resultsboard/templates/resultsboard/base.html`, add a fourth nav link after "Full Report":

```html
        <a href="{% url 'resultsboard:sensitivity' %}" class="{% block nav_sensitivity %}{% endblock %}">Sensitivity Analysis</a>
```

- [ ] **Step 9: Write `sensitivity.html`**

```html
{% extends "resultsboard/base.html" %}
{% block title %}Sensitivity Analysis{% endblock %}
{% block nav_sensitivity %}active{% endblock %}
{% block content %}

<h1>Sensitivity Analysis</h1>
<p>
  The methodology choices below (financial lag, missing-data handling,
  up/down threshold, feature scope, prediction horizon) are the pipeline's
  own working assumptions, not yet confirmed by the thesis supervisor. This
  page varies each one independently &mdash; holding everything else at the
  base pipeline's defaults &mdash; to show whether "no model beats the
  majority-class baseline" is a robust finding or an artifact of one
  specific choice.
</p>

<div class="card table-scroll">
  <table>
    <thead>
      <tr><th>Run</th><th>Dimension varied</th><th>Value</th><th>Rows</th><th>Best model</th><th>Best accuracy</th><th>Baseline</th><th>Beats baseline?</th></tr>
    </thead>
    <tbody>
      {% for run in runs %}
      <tr{% if run.dimension == "base" %} class="baseline-row"{% endif %}>
        <td>{{ run.run_id }}</td>
        <td>{{ run.dimension }}</td>
        <td>{{ run.value_label }}</td>
        <td>{{ run.row_count }}</td>
        <td>{{ run.best_model|default:"&mdash;" }}</td>
        <td>{{ run.best_accuracy|floatformat:3 }}</td>
        <td>{{ run.baseline_accuracy|floatformat:3 }}</td>
        <td>
          {% if run.beats_baseline %}<span class="badge red">YES</span>{% else %}<span class="badge green">no</span>{% endif %}
        </td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
</div>

<h2>Best model vs. baseline, all runs</h2>
<div class="chart-wrap">
  <canvas id="sensitivityChart" width="800" height="350"></canvas>
</div>
<script>
  const ctx = document.getElementById('sensitivityChart');
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: {{ chart_labels|safe }},
      datasets: [
        { label: 'Best model accuracy', data: {{ chart_best|safe }}, backgroundColor: '#2f5fb8' },
        { label: 'Majority-class baseline', data: {{ chart_baseline|safe }}, backgroundColor: '#b8452f' },
      ]
    },
    options: { scales: { y: { beginAtZero: true, max: 1 } } }
  });
</script>

<div class="callout">
  <strong>Reproduce this sweep:</strong> <span class="formula">python sensitivity/run_all.py</span>
  (requires <span class="formula">pipeline/outputs/</span> to already exist &mdash; run the main pipeline first).
</div>

{% endblock %}
```

- [ ] **Step 10: Verify manually against real data**

Run `python sensitivity/run_all.py` if not already done in Task 5, then `cd dashboard && python manage.py runserver`, visit `/sensitivity/`. Expected: 12-row table, chart with 12 pairs of bars, base run highlighted. Report the actual `beats_baseline` values seen for all 12 runs — if ANY run shows `beats_baseline: true`, that's a real, notable finding to flag explicitly, not something to double-check away.

- [ ] **Step 11: Run full test suite**

Run: `python -m pytest tests/ -v` from repo root. Expect all passing.

- [ ] **Step 12: Commit**

```bash
git add dashboard/resultsboard/data.py dashboard/resultsboard/views.py dashboard/resultsboard/urls.py dashboard/resultsboard/templates/resultsboard/base.html dashboard/resultsboard/templates/resultsboard/sensitivity.html tests/resultsboard/test_data.py .gitignore
git commit -m "feat(dashboard): add sensitivity analysis page"
```

---

## Task 7: Final polish and documentation

**Files:**
- Create: `sensitivity/README.md`
- Modify: `dashboard/README.md`

- [ ] **Step 1: Write `sensitivity/README.md`**

```markdown
# Sensitivity Analysis

One-factor-at-a-time robustness sweep over the pipeline's unconfirmed
methodology assumptions (financial lag, missing-data handling, up/down
threshold, feature scope, prediction horizon). See
`docs/superpowers/specs/2026-09-07-sensitivity-analysis-design.md` for the
full design rationale.

## Run

Requires `pipeline/outputs/` to already exist (run the main pipeline first,
see `../pipeline/README.md`):

    python sensitivity/run_all.py

Outputs land in `sensitivity/outputs/` (gitignored): one `<run_id>/` folder
per config with `dataset.csv`/`metrics.json`, plus an aggregated
`summary.json` across all 12 runs.

## View results

    cd dashboard
    python manage.py runserver

Visit http://127.0.0.1:8000/sensitivity/

## Tests

    python -m pytest tests/sensitivity -v
```

- [ ] **Step 2: Add a line to `dashboard/README.md`'s page list**

Add `- /sensitivity/ — one-factor-at-a-time robustness sweep across unconfirmed methodology assumptions` to the "Pages" list.

- [ ] **Step 3: Full end-to-end smoke test**

```bash
python -m pytest tests/ -v
python sensitivity/run_all.py
cd dashboard && python manage.py runserver
```
Visit `/`, `/models/`, `/report/`, `/sensitivity/` — confirm all 4 pages render with no errors.

- [ ] **Step 4: Commit**

```bash
git add sensitivity/README.md dashboard/README.md
git commit -m "docs(sensitivity): add README and dashboard page list entry"
```

---

## Self-Review Notes

- **Spec coverage:** all 5 dimensions from the design doc covered (lag: 4 variants, missing-data: 2 variants, threshold: 1 variant, feature-scope: 2 variants, horizon: 2 variants = 11 + base = 12); the additive `lag_days_override` change to already-frozen pipeline code is backward-compatible and tested; `pipeline/outputs/` is never written to by any sensitivity code (`sensitivity/lib/dataset_builder.py` only reads from it); the dashboard page is covered (Task 6).
- **Known scope call flagged, not hidden:** the design doc explicitly rules out a full cross-product sweep as uninterpretable — this plan implements one-factor-at-a-time only, per the approved design.
- **Type/name consistency:** `RunConfig` keys (`run_id`, `dimension`, `value_label`, `quarterly_lag_days`, `annual_lag_days`, `missing_strategy`, `threshold_mode`, `deadzone_pct`, `feature_scope`, `horizon_freq`) are used identically across `configs.py`, `dataset_builder.py`, `train_evaluate.py`, `run_all.py`, and the dashboard template.
- **Runtime expectation flagged:** Task 5 notes the weekly/monthly horizon variants will make the full sweep take noticeably longer than the base pipeline's own run (larger panels, same per-row feature-computation cost) — this is expected, not a bug to investigate.
