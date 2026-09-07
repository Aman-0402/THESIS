# tests/sensitivity/test_train_evaluate.py
import pandas as pd

from sensitivity.lib.train_evaluate import apply_missing_strategy, train_and_evaluate


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


def _dates(*iso_strings):
    return [pd.Timestamp(s) for s in iso_strings]


def test_train_and_evaluate_drop_majority_filters_val_and_test_rows_too():
    """Regression test: apply_missing_strategy's >50%-missing row filter must
    apply to val/test using their OWN missingness pattern, not just to train.
    Previously train_and_evaluate branched on `medians is not None`, which is
    true for both "impute" and "drop_majority" -- so drop_majority's val/test
    rows only got an unconditional fillna, never the row-drop, defeating the
    strategy for 2 of its 3 splits."""
    train_rows = pd.DataFrame({
        "period_end": _dates(
            "2020-01-01", "2020-04-01", "2020-07-01",
            "2020-10-01", "2021-01-01", "2021-04-01",
        ),
        "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "f2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "target": [0, 1, 0, 1, 0, 1],
    })
    val_rows = pd.DataFrame({
        "period_end": _dates("2022-02-01", "2022-03-01"),
        "f1": [1.0, None],
        "f2": [2.0, None],  # this row is missing both features (100% > 50%) -> must be dropped
        "target": [0, 1],
    })
    test_rows = pd.DataFrame({
        "period_end": _dates("2024-02-01", "2024-03-01", "2024-04-01"),
        "f1": [3.0, None, 5.0],
        "f2": [4.0, None, 6.0],  # middle row missing both features -> must be dropped
        "target": [0, 1, 1],
    })
    dataset = pd.concat([train_rows, val_rows, test_rows], ignore_index=True)
    config = {"run_id": "test_drop_majority", "missing_strategy": "drop_majority"}

    result = train_and_evaluate(dataset, config)

    assert "note" not in result  # confirms the non-degenerate path ran (models actually fit)
    assert result["val_rows"] == 1  # the all-missing val row was dropped, not just filled
    assert result["test_rows"] == 2  # the all-missing test row was dropped, not just filled
    assert len(result["models"]) == 8


def test_train_and_evaluate_returns_degenerate_result_for_single_class_test_split():
    """y_test.nunique() < 2 (not < 1, which is dead code since len(x_test) == 0
    already covers the empty case) must catch a single-class test split, since
    max(y_test.mean(), 1 - y_test.mean()) degenerates to 1.0 and ROC-AUC is
    undefined for one class."""
    train_rows = pd.DataFrame({
        "period_end": _dates(
            "2020-01-01", "2020-04-01", "2020-07-01",
            "2020-10-01", "2021-01-01", "2021-04-01",
        ),
        "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "f2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "target": [0, 1, 0, 1, 0, 1],
    })
    val_rows = pd.DataFrame({
        "period_end": _dates("2022-02-01"),
        "f1": [1.5], "f2": [15.0], "target": [0],
    })
    test_rows = pd.DataFrame({
        "period_end": _dates("2024-02-01", "2024-03-01"),
        "f1": [3.0, 3.5], "f2": [4.0, 4.5],
        "target": [0, 0],  # single class in test -> degenerate
    })
    dataset = pd.concat([train_rows, val_rows, test_rows], ignore_index=True)
    config = {"run_id": "test_degenerate", "missing_strategy": "impute"}

    result = train_and_evaluate(dataset, config)

    assert result["test_rows"] == 2
    assert result["models"] == {}
    assert result["baseline_accuracy"] is None
    assert result["best_model"] is None
    assert result["best_accuracy"] is None
    assert result["beats_baseline"] is None
    assert "note" in result
