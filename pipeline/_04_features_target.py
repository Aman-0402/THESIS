"""Attach 22 features + binary target to every panel row.

The 89-column `raw_financial.parquet` built by Task 6 is a wide concat of
heterogeneous source formats -- there is no single column literally named
`period_end`/`total_revenue`/etc for most companies. Three real formats were
found by inspecting `pipeline/outputs/raw_financial.parquet` and the source
CSVs under `pipeline/YUKTHA_CLEAN_2026-09-07/**/01_FINANCIAL_DATA/{Quarterly,
Annual}/`:

1. "derived" tidy long format (Laboratorios_Rovi, Formycon only): the raw
   parquet already carries usable `period_end`/`statement`/`line_item`/
   `value` columns for these rows (Task 6 built them that way for these two
   companies' `*_Derived.csv` files).
2. "Capital IQ as-reported" wide format (28 of 29 companies with any
   financial data): metadata rows, then a `Period Ended` marker row whose
   cells (in the various `Unnamed: N` columns) are the period-end dates for
   each fiscal-period column, then one row per line item with the line-item
   name in `Unnamed: 0` and the value for a given period in that period's
   `Unnamed: N` column.
3. "yfinance" wide format (an annual-income supplement for ~14 companies):
   a proper header row whose column names ARE the period-end dates, one row
   per line item with the line-item name in `Unnamed: 0`.

`_normalize_financial_long()` melts all three formats into one tidy
(company_folder, period_end, period_type, statement, line_item, value)
table; `_pivot_fundamentals()` then maps line items to the raw fields
`fundamental_features()` expects, via best-effort alias lists checked
against real line-item names observed in the data (see FIELD_ALIASES).
Fields with no matching line item anywhere in the data (market_cap --
these are financial-statement extracts, not market-data files; total_debt
-- no source file carries a single clean "Total Debt" balance-sheet line,
only dozens of cash-flow "repayment/issuance of debt" activity lines that
are NOT the same thing as a balance) come out None for every row. That is
a real data-coverage gap, not a bug: fundamental_features() is null-safe
and any ratio depending on those fields (price_to_book,
price_to_earnings_proxy, debt_to_equity) is simply None for all rows.

Zenotech_Laboratories has zero rows in raw_financial.parquet (confirmed by
Task 6: its financial data exists only as .xlsx files under
01_FINANCIAL_DATA/Other, which Task 6's loader does not read) so it
naturally gets None fundamentals for every quarter -- its market features
still compute fine from its 5567-row stock series.

A second, unrelated real-data defect was found in `pipeline/outputs/
stock_clean/*.csv` (Task 7 output) while sanity-checking this task's
numbers: 23 of the 29 non-empty stock series carry TWO columns literally
named "close" on disk (e.g. `date,close,close` as the literal CSV header
for Cipla.csv). Root cause: `_02_clean_stock.py::_normalize_columns()`
lowercases column names from every concatenated source file without
deduping collisions -- when one source CSV's price column was cased
"Close" and another (different) source CSV for the same company was cased
"close", concatenating them and then lowercasing produces two same-named
"close" columns instead of one. Confirmed the two columns are perfectly
row-complementary (never both non-null for the same row, across every
affected company) so coalescing them loses no data. Left unhandled, naive
`pd.read_csv` silently reads only the FIRST "close" column (pandas renames
the second to "close.1"), which for most affected companies is almost
entirely NaN -- this was previously making ~78% of market features None
and (worse) silently defaulting `compute_binary_target` to 0 whenever
price_t was NaN (NaN comparisons are falsy in Python), massively
depressing the positive rate. `_load_stock_clean()` below works around
this at load time rather than editing already-committed Task 7 code.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from pipeline.lib.features import market_features, fundamental_features, FUNDAMENTAL_FEATURE_NAMES
from pipeline.lib.lag import is_available, ROVI_FOLDER
from pipeline.lib.target import last_trading_day_price, compute_binary_target

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
STOCK_CLEAN_DIR = OUTPUT_DIR / "stock_clean"

# statement -> ordered list of real line-item names seen in the source data
# (Capital IQ "as reported" statements and the yfinance income supplement).
# First alias found for a given (company, period_end, period_type) wins.
FIELD_ALIASES: dict[str, tuple[str | None, list[str]]] = {
    "total_revenue": ("Income Statement", [
        "Total Revenues", "Total Revenue", "Revenue", "Operating Revenue", "Operating Revenues",
        "Net Sales", "Net Revenues", "Sales",
        "Net Sales to Third Parties from Continuing Operations",
        "Net Sales from Continuing Operations", "Net Sales To: Third Parties",
        "Sales Revenue",
        # "Revenues" is often a section HEADER row (no value) rather than a
        # line item in the Capital IQ "as reported" statements (e.g. Merck,
        # Baxter, Novartis, Sanofi, Teva) -- kept last since a real "Revenue"/
        # "Total Revenues" line always wins when present.
        "Revenues",
    ]),
    "net_income": ("Income Statement", [
        "Net Income (Loss)", "Net Income", "Net Income / Comprehensive Income",
        "Net Income/Comprehensive Income",
    ]),
    "operating_income": ("Income Statement", [
        "Operating Income (Loss)", "Operating Income", "Operating Profit",
        "Operating Profit/Loss (EBIT)",
    ]),
    "total_assets": ("Balance Sheet", ["Total Assets"]),
    "total_equity": ("Balance Sheet", [
        "Total Shareholders Equity", "Total Shareholders' Equity",
        "Total Stockholders Equity", "Stockholders Equity", "Total Equity",
    ]),
    "current_assets": ("Balance Sheet", ["Total Current Assets"]),
    "current_liabilities": ("Balance Sheet", ["Total Current Liabilities"]),
    # No source file carries a clean single "Total Debt" balance-sheet line
    # (only many cash-flow "repayment of debt" activity lines, which are not
    # balances). Kept as a real alias in case a company happens to report
    # one, but expect this to be None for effectively every row.
    "total_debt": ("Balance Sheet", ["Total Debt"]),
    # Market cap never appears in these financial-statement extracts (it is
    # a market-data figure, not a statement line item) -- expect None for
    # every row across every company.
    "market_cap": (None, ["Market Cap", "Market Capitalization"]),
}

_CAPIQ_SKIP_LABELS = {"Currency", "Units", "Period Ended"}
_NON_DATE_COLUMNS = {
    "company_folder", "period_type", "source_file", "period_end", "statement",
    "line_item", "value", "value_period", "currency", "units", "scope", "source",
    "source_url", "report_date", "page", "original_file", "sha256",
    "direct_derived", "notes", "period_label", "company",
}


def _parse_date_maybe(x) -> pd.Timestamp | None:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    try:
        ts = pd.to_datetime(x, errors="raise")
    except (ValueError, TypeError):
        return None
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts).normalize()


def _to_float_maybe(x) -> float | None:
    if x is None:
        return None
    if isinstance(x, str):
        x = x.replace(",", "").strip()
        if x in ("", "-", "NM", "NA", "N/A"):
            return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if pd.isna(v):
        return None
    return v


def _statement_from_source_file(source_file: str) -> str | None:
    name = str(source_file).lower()
    if "balance_sheet" in name or "balance" in name:
        return "Balance Sheet"
    if "income_statement" in name or "income" in name:
        return "Income Statement"
    if "cash_flow" in name or "cashflow" in name:
        return "Cash Flow"
    return None


def _melt_capiq_block(group: pd.DataFrame, marker_idx, company_folder: str, source_file: str, period_type: str) -> list[dict]:
    marker = group.loc[marker_idx]
    date_cols = {}
    for col in group.columns:
        if not str(col).startswith("Unnamed: ") or col == "Unnamed: 0":
            continue
        parsed = _parse_date_maybe(marker.get(col))
        if parsed is not None:
            date_cols[col] = parsed
    if not date_cols:
        return []

    statement = _statement_from_source_file(source_file)
    out = []
    item_rows = group.loc[group.index > marker_idx]
    for _, row in item_rows.iterrows():
        label = row.get("Unnamed: 0")
        if not isinstance(label, str):
            continue
        label = label.strip()
        if not label or label in _CAPIQ_SKIP_LABELS or len(label) > 200:
            continue
        for col, period_end in date_cols.items():
            num = _to_float_maybe(row.get(col))
            if num is None:
                continue
            out.append({
                "company_folder": company_folder, "period_end": period_end,
                "period_type": period_type, "statement": statement,
                "line_item": label, "value": num, "source_file": source_file,
            })
    return out


def _melt_yfinance_block(group: pd.DataFrame, company_folder: str, source_file: str, period_type: str) -> list[dict]:
    if "Unnamed: 0" not in group.columns:
        return []
    date_cols = {}
    for col in group.columns:
        if col in _NON_DATE_COLUMNS or str(col).startswith("Unnamed: "):
            continue
        parsed = _parse_date_maybe(col)
        if parsed is not None and group[col].notna().any():
            date_cols[col] = parsed
    if not date_cols:
        return []

    statement = _statement_from_source_file(source_file)
    out = []
    for _, row in group.iterrows():
        label = row.get("Unnamed: 0")
        if not isinstance(label, str) or not label.strip():
            continue
        label = label.strip()
        for col, period_end in date_cols.items():
            num = _to_float_maybe(row.get(col))
            if num is None:
                continue
            out.append({
                "company_folder": company_folder, "period_end": period_end,
                "period_type": period_type, "statement": statement,
                "line_item": label, "value": num, "source_file": source_file,
            })
    return out


# also imported directly by sensitivity/lib/dataset_builder.py -- keep this signature/behavior stable
def _normalize_financial_long(financial: pd.DataFrame) -> pd.DataFrame:
    """Melt every (company_folder, source_file) block -- regardless of which
    of the three real raw formats it's in -- into one tidy long table."""
    records = []
    for (company_folder, source_file), group in financial.groupby(["company_folder", "source_file"], sort=False):
        group = group.sort_index()
        period_type = group["period_type"].iloc[0]

        if "period_end" in group.columns and group["period_end"].notna().any():
            derived = group[group["period_end"].notna()]
            for _, r in derived.iterrows():
                records.append({
                    "company_folder": company_folder,
                    "period_end": pd.Timestamp(r["period_end"]),
                    "period_type": period_type,
                    "statement": r.get("statement"),
                    "line_item": r.get("line_item"),
                    "value": _to_float_maybe(r.get("value")),
                    "source_file": source_file,
                })
            continue

        marker_mask = (group.get("Unnamed: 0") == "Period Ended") if "Unnamed: 0" in group.columns else None
        if marker_mask is not None and marker_mask.any():
            marker_idx = group.index[marker_mask][0]
            records.extend(_melt_capiq_block(group, marker_idx, company_folder, source_file, period_type))
            continue

        records.extend(_melt_yfinance_block(group, company_folder, source_file, period_type))

    if not records:
        return pd.DataFrame(columns=["company_folder", "period_end", "period_type", "statement", "line_item", "value", "source_file"])
    return pd.DataFrame.from_records(records)


# also imported directly by sensitivity/lib/dataset_builder.py -- keep this signature/behavior stable
def _pivot_fundamentals(long_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (company_folder, period_end, period_type) with the raw
    fundamental fields fundamental_features() expects, plus prior_revenue
    (nearest revenue ~1 year earlier, same period_type, within 45 days)."""
    columns = ["company_folder", "period_end", "period_type"] + list(FIELD_ALIASES) + ["prior_revenue"]
    if long_df.empty:
        return pd.DataFrame(columns=columns)

    base = long_df[["company_folder", "period_end", "period_type"]].drop_duplicates()
    result = base.set_index(["company_folder", "period_end", "period_type"])

    for field, (statement, aliases) in FIELD_ALIASES.items():
        sub = long_df
        if statement is not None:
            sub = sub[sub["statement"] == statement]
        sub = sub[sub["line_item"].isin(aliases)]
        if sub.empty:
            result[field] = None
            continue
        rank = {name: i for i, name in enumerate(aliases)}
        sub = sub.assign(_rank=sub["line_item"].map(rank))
        sub = sub.sort_values(["company_folder", "period_end", "period_type", "_rank"])
        picked = sub.drop_duplicates(subset=["company_folder", "period_end", "period_type"], keep="first")
        picked = picked.set_index(["company_folder", "period_end", "period_type"])["value"]
        result[field] = picked

    result = result.reset_index().sort_values(["company_folder", "period_type", "period_end"]).reset_index(drop=True)
    result["prior_revenue"] = _compute_prior_revenue(result)
    return result


def _compute_prior_revenue(wide: pd.DataFrame) -> pd.Series:
    prior = pd.Series(None, index=wide.index, dtype=object)
    for (_, _), g in wide.groupby(["company_folder", "period_type"], sort=False):
        g = g.sort_values("period_end")
        target_dates = g["period_end"] - pd.Timedelta(days=365)
        for pos, (idx, td) in enumerate(zip(g.index, target_dates)):
            candidates = g.iloc[:pos]
            if candidates.empty:
                continue
            diffs = (candidates["period_end"] - td).abs()
            best_pos = diffs.values.argmin()
            if diffs.iloc[best_pos] <= pd.Timedelta(days=45):
                prior.loc[idx] = candidates["total_revenue"].iloc[best_pos]
    return prior


# also imported directly by sensitivity/lib/dataset_builder.py -- keep this signature/behavior stable
def _load_stock_clean(path: Path) -> pd.DataFrame:
    """Read a pipeline/outputs/stock_clean/<folder>.csv, coalescing the
    duplicate on-disk "close" columns some companies have (see module
    docstring) into a single clean "close" column."""
    df = pd.read_csv(path, parse_dates=["date"])
    close_cols = [c for c in df.columns if c == "close" or re.fullmatch(r"close\.\d+", str(c))]
    if len(close_cols) > 1:
        df["close"] = df[close_cols].bfill(axis=1).iloc[:, 0]
        df = df.drop(columns=[c for c in close_cols if c != "close"])
    return df[["date", "close"]]


# also imported directly by sensitivity/lib/dataset_builder.py -- keep this signature/behavior stable
def latest_available_financial_row(
    fin_rows: pd.DataFrame,
    quarter_end: pd.Timestamp,
    company_folder: str,
    lag_days_override: dict[str, int] | None = None,
):
    """fin_rows must have columns: period_end (datetime), period_type,
    publication_date (nullable, only populated for Rovi), plus the raw
    fundamental fields consumed by fundamental_features()."""
    candidates = []
    for _, r in fin_rows.iterrows():
        pub = r.get("publication_date")
        pub_date = pub.date() if company_folder == ROVI_FOLDER and pd.notna(pub) else None
        # NOTE: this branch is currently dead code in practice. `main()`
        # sets fundamentals["publication_date"] = pd.NaT for every company
        # (including Rovi), so pub_date is always None here and every Rovi
        # row gets skipped -- Rovi's fundamentals always resolve to None.
        # That's harmless today because Rovi has zero rows in stock_clean
        # (an empty series, confirmed in Task 7) and therefore contributes
        # zero rows to panel_quarters.csv -- this function is never even
        # called with company_folder == ROVI_FOLDER right now. It's kept
        # (rather than deleted) for forward-compatibility: if Rovi's stock
        # data is ever backfilled and it starts appearing in the panel,
        # real publication dates would need to be wired into `fundamentals`
        # before this branch could produce a result instead of always None.
        if company_folder == ROVI_FOLDER and pub_date is None:
            continue
        if is_available(
            period_end=r["period_end"].date(),
            observation_date=quarter_end.date(),
            period_type=r["period_type"],
            company_folder=company_folder,
            publication_date=pub_date,
            lag_days_override=lag_days_override,
        ):
            candidates.append(r)
    if not candidates:
        return None
    return max(candidates, key=lambda r: r["period_end"])


def main():
    panel = pd.read_csv(OUTPUT_DIR / "panel_quarters.csv", parse_dates=["quarter_end"])
    financial = pd.read_parquet(OUTPUT_DIR / "raw_financial.parquet")

    financial_long = _normalize_financial_long(financial)
    fundamentals = _pivot_fundamentals(financial_long)
    fundamentals["publication_date"] = pd.NaT

    coverage = {}
    for company_folder in panel["company_folder"].unique():
        n_rows = (fundamentals["company_folder"] == company_folder).sum()
        n_with_revenue = ((fundamentals["company_folder"] == company_folder) & fundamentals["total_revenue"].notna()).sum()
        coverage[company_folder] = (n_rows, n_with_revenue)

    out_rows = []
    for company_folder, group in panel.groupby("company_folder"):
        stock = _load_stock_clean(STOCK_CLEAN_DIR / f"{company_folder}.csv")
        fin_rows = fundamentals[fundamentals["company_folder"] == company_folder]
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
    print("\nfundamental coverage (financial period-rows with a mapped total_revenue) per company:")
    for company_folder, (n_rows, n_with_revenue) in sorted(coverage.items()):
        print(f"  {company_folder}: {n_with_revenue}/{n_rows} periods with total_revenue mapped")


if __name__ == "__main__":
    main()
