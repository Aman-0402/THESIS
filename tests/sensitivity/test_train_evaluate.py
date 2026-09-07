# tests/sensitivity/test_train_evaluate.py
import pandas as pd

from sensitivity.lib.train_evaluate import apply_missing_strategy


def _sample_df():
    return pd.DataFrame({
        "a": [1.0, None, 3.0, 4.0],
        "b": [10.0, 20.0, None, 40.0],
        "target": [1, 0, 1, 0],
    })


def test_impute_strategy_fills_with_train_median_keeps_all_rows():
    train = _sample_df()
    out_train, medians = apply_missing_strategy(train, ["a", "b"], "impute")
    assert len(out_train) == 4
    assert out_train["a"].isna().sum() == 0
    assert medians["a"] == 3.0  # median of [1,3,4] (train's own non-null values)


def test_drop_any_strategy_removes_rows_with_any_missing_feature():
    train = _sample_df()
    out_train, medians = apply_missing_strategy(train, ["a", "b"], "drop_any")
    assert len(out_train) == 2  # rows 0 and 3 have no missing values
    assert out_train["a"].isna().sum() == 0


def test_drop_majority_strategy_drops_rows_missing_over_half_features_then_imputes_rest():
    df = pd.DataFrame({
        "a": [1.0, None, None, 4.0],
        "b": [10.0, None, 30.0, 40.0],
        "target": [1, 0, 1, 0],
    })
    # row 1 is missing both of 2 features (100% > 50%) -> dropped
    # row 2 is missing 1 of 2 features (50%, not over 50%) -> kept, imputed
    out_train, medians = apply_missing_strategy(df, ["a", "b"], "drop_majority")
    assert len(out_train) == 3
    assert out_train["a"].isna().sum() == 0
