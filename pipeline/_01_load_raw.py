"""Load every company's daily-price and financial-statement CSVs from the
cleaned data folder into two parquet caches, so later stages parse once."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
