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
    result = out[["_calendar_date"] + value_cols].rename(columns={"_calendar_date": date_col})
    return result.reset_index(drop=True)
