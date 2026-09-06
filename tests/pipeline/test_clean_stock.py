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
