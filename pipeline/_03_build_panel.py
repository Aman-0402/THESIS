"""Company-quarter panel: one row per (company, quarter) the cleaned stock
series actually covers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from pipeline.lib.companies import COMPANIES

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
STOCK_CLEAN_DIR = OUTPUT_DIR / "stock_clean"


def quarter_ends_in_range(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    # pandas >= 2.2 deprecated freq="Q" in favor of "QE" (quarter-end); both
    # anchor to the same calendar quarter-end dates (Mar/Jun/Sep/Dec 31/30),
    # "QE" is just the non-deprecated spelling in this environment (pandas 2.3.3).
    return list(pd.date_range(start=start, end=end, freq="QE"))


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
