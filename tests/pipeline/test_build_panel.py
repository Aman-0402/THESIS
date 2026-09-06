import pandas as pd
from pipeline._03_build_panel import quarter_ends_in_range


def test_quarter_ends_in_range_returns_calendar_quarter_ends():
    ends = quarter_ends_in_range(pd.Timestamp("2014-07-01"), pd.Timestamp("2026-03-31"))
    assert ends[0] == pd.Timestamp("2014-09-30")
    assert ends[-1] == pd.Timestamp("2026-03-31")
    assert all(pd.Timestamp(d).is_quarter_end for d in ends)
