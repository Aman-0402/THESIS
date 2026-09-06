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
