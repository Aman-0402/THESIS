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
