import pandas as pd

from pipeline._04_features_target import (
    _parse_date_maybe,
    _to_float_maybe,
    _statement_from_source_file,
    _normalize_financial_long,
    _pivot_fundamentals,
    _load_stock_clean,
)


def test_load_stock_clean_coalesces_duplicate_close_columns(tmp_path):
    # Reproduces the real on-disk defect: a literal "date,close,close" header
    # (pandas parses the second "close" as "close.1"), with the two columns
    # row-complementary (only one populated per row) as confirmed in the
    # actual stock_clean/*.csv files for 23 of 29 companies.
    csv_path = tmp_path / "Dup.csv"
    csv_path.write_text("date,close,close\n2024-01-01,,10.0\n2024-01-02,11.0,\n")

    df = _load_stock_clean(csv_path)

    assert list(df.columns) == ["date", "close"]
    assert df["close"].tolist() == [10.0, 11.0]


def test_load_stock_clean_passes_through_a_clean_single_close_column(tmp_path):
    csv_path = tmp_path / "Clean.csv"
    csv_path.write_text("date,close\n2024-01-01,10.0\n2024-01-02,11.0\n")

    df = _load_stock_clean(csv_path)

    assert list(df.columns) == ["date", "close"]
    assert df["close"].tolist() == [10.0, 11.0]


def test_parse_date_maybe_parses_real_dates_and_rejects_text():
    assert _parse_date_maybe("2015-04-01 00:00:00") == pd.Timestamp("2015-04-01")
    assert _parse_date_maybe("Total Assets") is None
    assert _parse_date_maybe(None) is None
    assert _parse_date_maybe(float("nan")) is None


def test_to_float_maybe_handles_thousands_separators_and_placeholders():
    assert _to_float_maybe("1,234.5") == 1234.5
    assert _to_float_maybe("-") is None
    assert _to_float_maybe("NM") is None
    assert _to_float_maybe(None) is None
    assert _to_float_maybe(42) == 42.0


def test_statement_from_source_file_classifies_known_patterns():
    assert _statement_from_source_file("Ajanta_Pharma_Balance_Sheet_Quarters.csv") == "Balance Sheet"
    assert _statement_from_source_file("Ajanta_Pharma_Income_Statement_Quarters.csv") == "Income Statement"
    assert _statement_from_source_file("Ajanta_Pharma_Cash_Flow_Years.csv") == "Cash Flow"
    assert _statement_from_source_file("something_weird.csv") is None


def _capiq_income_block(company_folder="Acme"):
    """A minimal Capital IQ 'as reported' Income Statement block: a marker
    row ('Period Ended') mapping Unnamed:1/Unnamed:2 to two period-end
    dates, followed by line-item rows."""
    rows = [
        {"Unnamed: 0": "Acme Inc | Income Statement (As Reported)", "Unnamed: 1": None, "Unnamed: 2": None},
        {"Unnamed: 0": "Period Ended", "Unnamed: 1": "2023-03-31 00:00:00", "Unnamed: 2": "2023-06-30 00:00:00"},
        {"Unnamed: 0": "Currency", "Unnamed: 1": "INR", "Unnamed: 2": "INR"},
        {"Unnamed: 0": "Total Revenues", "Unnamed: 1": "1000", "Unnamed: 2": "1100"},
        {"Unnamed: 0": "Net Income (Loss)", "Unnamed: 1": "100", "Unnamed: 2": "120"},
        {"Unnamed: 0": "Operating Income (Loss)", "Unnamed: 1": "150", "Unnamed: 2": "160"},
    ]
    df = pd.DataFrame(rows)
    df["company_folder"] = company_folder
    df["source_file"] = "Acme_Income_Statement_Quarters.csv"
    df["period_type"] = "quarterly"
    df["period_end"] = pd.NaT
    return df


def test_normalize_financial_long_melts_capiq_wide_block():
    financial = _capiq_income_block()
    long_df = _normalize_financial_long(financial)

    revenues = long_df[long_df["line_item"] == "Total Revenues"].sort_values("period_end")
    assert list(revenues["value"]) == [1000.0, 1100.0]
    assert list(revenues["period_end"]) == [pd.Timestamp("2023-03-31"), pd.Timestamp("2023-06-30")]
    assert (long_df["statement"] == "Income Statement").all()
    # metadata rows (title, "Period Ended", "Currency") must not leak through as line items
    assert "Period Ended" not in set(long_df["line_item"])
    assert "Currency" not in set(long_df["line_item"])


def test_pivot_fundamentals_maps_aliases_and_computes_prior_revenue():
    financial = _capiq_income_block()
    long_df = _normalize_financial_long(financial)
    wide = _pivot_fundamentals(long_df)

    row = wide[wide["period_end"] == pd.Timestamp("2023-06-30")].iloc[0]
    assert row["total_revenue"] == 1100.0
    assert row["net_income"] == 120.0
    assert row["operating_income"] == 160.0
    # no balance-sheet rows in this fixture -> those fields stay None
    assert pd.isna(row["total_assets"])


def test_pivot_fundamentals_prior_revenue_uses_prior_year_same_period_type():
    rows = [
        {"company_folder": "Acme", "period_end": pd.Timestamp("2022-06-30"), "period_type": "quarterly",
         "statement": "Income Statement", "line_item": "Total Revenues", "value": 900.0},
        {"company_folder": "Acme", "period_end": pd.Timestamp("2023-06-30"), "period_type": "quarterly",
         "statement": "Income Statement", "line_item": "Total Revenues", "value": 1100.0},
    ]
    long_df = pd.DataFrame(rows)
    wide = _pivot_fundamentals(long_df)
    row2023 = wide[wide["period_end"] == pd.Timestamp("2023-06-30")].iloc[0]
    assert row2023["prior_revenue"] == 900.0


def test_normalize_financial_long_passes_through_already_tidy_derived_rows():
    df = pd.DataFrame([{
        "company_folder": "Laboratorios_Rovi", "source_file": "Rovi_Income_Quarterly_Derived.csv",
        "period_type": "quarterly", "period_end": pd.Timestamp("2024-03-31"),
        "statement": "Income Statement", "line_item": "Revenue", "value": 151175.0,
    }])
    long_df = _normalize_financial_long(df)
    assert len(long_df) == 1
    assert long_df.iloc[0]["line_item"] == "Revenue"
    assert long_df.iloc[0]["value"] == 151175.0
