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


def test_predictions_csv_has_y_proba_column(tmp_path, monkeypatch):
    """Run main() against a tiny synthetic dataset and confirm the written
    predictions CSV has a y_proba column with values in [0, 1]."""
    import pandas as pd
    from pipeline import _05_split_train as mod

    rows = []
    for i in range(40):
        rows.append({
            "company_folder": "TestCo",
            "quarter_end": pd.Timestamp("2020-01-01") + pd.Timedelta(days=90 * i),
            **{c: float(i % 5) for c in mod.FEATURE_COLS},
            "target": i % 2,
        })
    dataset = pd.DataFrame(rows)

    outputs_dir = tmp_path / "outputs"
    (outputs_dir / "predictions").mkdir(parents=True)
    dataset.to_csv(outputs_dir / "dataset.csv", index=False)

    # NOTE: plan specified VAL_START=2020-06-01 / TEST_START=2021-01-01, but
    # with 90-day spacing that leaves only 2 training rows, which
    # KNeighborsClassifier (default n_neighbors=5) cannot predict against
    # regardless of the y_proba change under test. Shifted forward so the
    # training fold has >=5 rows; intent (small train/val, most rows in
    # test) is unchanged.
    monkeypatch.setattr(mod, "OUTPUT_DIR", outputs_dir)
    monkeypatch.setattr(mod, "VAL_START", pd.Timestamp("2021-01-01"))
    monkeypatch.setattr(mod, "TEST_START", pd.Timestamp("2021-06-01"))

    mod.main()

    pred = pd.read_csv(outputs_dir / "predictions" / "logistic_regression.csv")
    assert "y_proba" in pred.columns
    assert pred["y_proba"].between(0, 1).all()
