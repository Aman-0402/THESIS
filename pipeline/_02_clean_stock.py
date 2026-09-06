"""Build one canonical daily stock series per company. documentation .pdf
sections 6-7 (Phase 3A/3B)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from pipeline.lib.companies import COMPANIES, STOCK_SERIES_OVERRIDE
from pipeline.lib.prices import pick_price_column, dedupe_by_calendar_date

CLEAN_ROOT = Path(__file__).resolve().parents[1] / "YUKTHA_CLEAN_2026-09-07"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
STOCK_CLEAN_DIR = OUTPUT_DIR / "stock_clean"
REGION_DIR = {"IN": "01_INDIAN_COMPANIES", "NON_IN": "02_NON_INDIAN_COMPANIES"}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop leftover all-NaN columns (artifacts of slicing one company out
    of a wide multi-company parquet) before lowercasing, so two differently
    -cased empty columns can't collide into one duplicate label."""
    df = df.dropna(axis=1, how="all")
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _load_other_source_material_daily(company_folder: str, region: str) -> list[pd.DataFrame]:
    """For override companies, the canonical block sometimes lives outside
    02_MARKET_DATA/Daily (e.g. a *_yahoo_adjusted_close_supplement.csv under
    04_OTHER_SOURCE_MATERIAL). Load candidate stock-like CSVs from there."""
    company_dir = CLEAN_ROOT / REGION_DIR[region] / company_folder
    other_dir = company_dir / "04_OTHER_SOURCE_MATERIAL"
    frames = []
    if not other_dir.is_dir():
        return frames
    for csv_path in sorted(other_dir.glob("*.csv")):
        name = csv_path.name.lower()
        if "yahoo_adjusted_close_supplement" in name or ("stock" in name and "daily" in name):
            df = pd.read_csv(csv_path)
            df["source_file"] = csv_path.name
            frames.append(df)
    return frames


def select_canonical_block(company_folder: str, frames: list[pd.DataFrame], region: str | None = None) -> pd.DataFrame:
    override = STOCK_SERIES_OVERRIDE.get(company_folder)
    if override is None:
        return pd.concat(frames, ignore_index=True)

    candidates = list(frames)
    normalized_pairs = [(_normalize_columns(f.copy()), f) for f in candidates]

    if override == "adjusted_yahoo":
        if region is not None:
            extra = _load_other_source_material_daily(company_folder, region)
            candidates = candidates + extra
            normalized_pairs = [(_normalize_columns(f.copy()), f) for f in candidates]
        supplement = [
            orig for _, orig in normalized_pairs
            if "yahoo_adjusted_close_supplement" in str(orig["source_file"].iloc[0]).lower()
        ]
        if supplement:
            return pd.concat(supplement, ignore_index=True)
        adj = [orig for norm, orig in normalized_pairs if "adj_close" in norm.columns]
        if adj:
            return pd.concat(adj, ignore_index=True)
        raise ValueError(f"no adjusted-close block found for override company {company_folder}")

    if override == "paris_eur":
        with_country = [orig for norm, orig in normalized_pairs if "country" in norm.columns]
        if with_country:
            return pd.concat(with_country, ignore_index=True)
        raise ValueError(f"no country-tagged (Paris/EUR) block found for {company_folder}")

    raise ValueError(f"unknown override '{override}' for {company_folder}")


def clean_one_company(company_folder: str, raw_daily: pd.DataFrame, region: str | None = None) -> pd.DataFrame:
    company_rows = raw_daily[raw_daily["company_folder"] == company_folder]
    if company_rows.empty:
        return pd.DataFrame(columns=["date", "close"])

    company_rows = company_rows.dropna(axis=1, how="all")
    frames = [g for _, g in company_rows.groupby("source_file")]
    if not frames:
        return pd.DataFrame(columns=["date", "close"])

    block = select_canonical_block(company_folder, frames, region=region)
    block = _normalize_columns(block)
    price_col = pick_price_column(block)
    block["date"] = pd.to_datetime(block["date"], format="mixed", utc=True)
    clean = dedupe_by_calendar_date(block, date_col="date", value_cols=[price_col])
    return clean.rename(columns={price_col: "close"}).sort_values("date").reset_index(drop=True)


def main():
    STOCK_CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    raw_daily = pd.read_parquet(OUTPUT_DIR / "raw_daily.parquet")

    for company in COMPANIES:
        try:
            clean = clean_one_company(company["folder"], raw_daily, region=company["region"])
        except Exception as exc:
            print(f"{company['folder']}: ERROR - {exc}")
            continue
        clean.to_csv(STOCK_CLEAN_DIR / f"{company['folder']}.csv", index=False)
        print(f"{company['folder']}: {len(clean)} trading days")


if __name__ == "__main__":
    main()
