import pandas as pd

from pipeline._05_split_train import chronological_split


def test_chronological_split_boundaries():
    dates = pd.to_datetime(["2021-12-31", "2022-01-31", "2023-12-31", "2024-01-31", "2026-01-31"])
    df = pd.DataFrame({"quarter_end": dates, "x": range(5)})
    train, val, test = chronological_split(df)
    assert train["quarter_end"].max() < pd.Timestamp("2022-01-01")
    assert val["quarter_end"].min() >= pd.Timestamp("2022-01-01")
    assert val["quarter_end"].max() < pd.Timestamp("2024-01-01")
    assert test["quarter_end"].min() >= pd.Timestamp("2024-01-01")
