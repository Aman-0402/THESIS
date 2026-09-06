import pandas as pd

from pipeline._02_clean_stock import select_canonical_block


def test_default_company_concatenates_all_blocks():
    frames = [
        pd.DataFrame({"Date": ["2024-01-01"], "Close": [10.0], "source_file": ["x_daily.csv"]}),
        pd.DataFrame({"Date": ["2024-01-02"], "Close": [11.0], "source_file": ["y_daily.csv"]}),
    ]
    out = select_canonical_block("Cipla", frames)
    assert len(out) == 2


def test_override_company_keeps_only_matching_block():
    frames = [
        pd.DataFrame({"Date": ["2024-01-01"], "Close": [10.0], "source_file": ["ajanta_daily.csv"]}),
        pd.DataFrame({"Date": ["2024-01-01"], "Close": [10.5], "AdjClose": [9.9],
                       "source_file": ["ajanta_yahoo_adjusted_close_supplement.csv"]}),
    ]
    out = select_canonical_block("Ajanta_Pharma", frames)
    assert len(out) == 1
    assert out.iloc[0]["source_file"] == "ajanta_yahoo_adjusted_close_supplement.csv"


def test_paris_eur_override_prefers_country_tagged_block():
    frames = [
        pd.DataFrame({"Date": ["2024-01-01"], "Close": [100.0], "source_file": ["sanofi_sny_daily.csv"]}),
        pd.DataFrame({"Date": ["2024-01-01"], "Close": [90.0], "Country": ["France"],
                       "source_file": ["sanofi_daily.csv"]}),
    ]
    out = select_canonical_block("Sanofi", frames)
    assert len(out) == 1
    assert out.iloc[0]["source_file"] == "sanofi_daily.csv"


def test_normalize_columns_coalesces_duplicate_labels():
    from pipeline._02_clean_stock import _normalize_columns
    df = pd.DataFrame({
        "Close": [10.0, None, 12.0],
        "close": [None, 11.0, None],
    })
    out = _normalize_columns(df)
    assert out.columns.tolist() == ["close"]
    assert out["close"].tolist() == [10.0, 11.0, 12.0]
