"""Build one canonical daily stock series per company. documentation .pdf
sections 6-7 (Phase 3A/3B)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
