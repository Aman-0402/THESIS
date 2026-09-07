import json
import pandas as pd
import pytest

from resultsboard.data import (
    PipelineOutputsMissing,
    load_dataset_summary,
    load_model_metrics,
    load_model_names,
    load_roc_curve,
)


@pytest.fixture
def outputs_dir(tmp_path):
    d = tmp_path / "outputs"
    d.mkdir()
    (d / "predictions").mkdir()

    dataset = pd.DataFrame({
        "company_folder": ["A", "A", "B"],
        "quarter_end": pd.to_datetime(["2020-01-01", "2020-04-01", "2020-01-01"]),
        "return_1q": [0.1, None, 0.2],
        "volume_change_60d": [None, None, None],  # all-NaN column
        "target": [1, 0, 1],
    })
    dataset.to_csv(d / "dataset.csv", index=False)

    metrics = {
        "logistic_regression": {
            "accuracy": 0.5, "precision": 0.6, "recall": 0.4, "specificity": 0.7,
            "f1": 0.48, "roc_auc": 0.55, "validation_accuracy": 0.52,
            "confusion_matrix": [[10, 5], [8, 12]],
        },
        "_majority_class_baseline_accuracy": 0.579,
        "_dropped_all_nan_feature_columns": ["volume_change_60d"],
    }
    (d / "metrics.json").write_text(json.dumps(metrics))

    pd.DataFrame({
        "company_folder": ["A", "B"],
        "quarter_end": ["2024-01-01", "2024-01-01"],
        "y_true": [1, 0],
        "y_pred": [1, 1],
        "y_proba": [0.8, 0.6],
    }).to_csv(d / "predictions" / "logistic_regression.csv", index=False)

    return d


def test_load_dataset_summary_reports_row_and_feature_counts(outputs_dir):
    summary = load_dataset_summary(outputs_dir)
    assert summary["row_count"] == 3
    assert "return_1q" in summary["active_features"]
    assert "volume_change_60d" not in summary["active_features"]
    assert summary["positive_rate"] == pytest.approx(2 / 3)


def test_load_model_names_lists_models_excluding_meta_keys(outputs_dir):
    names = load_model_names(outputs_dir)
    assert names == ["logistic_regression"]


def test_load_model_metrics_returns_metrics_and_baseline(outputs_dir):
    metrics, baseline = load_model_metrics(outputs_dir)
    assert metrics["logistic_regression"]["accuracy"] == 0.5
    assert baseline == pytest.approx(0.579)


def test_load_roc_curve_computes_fpr_tpr_from_predictions(outputs_dir):
    fpr, tpr, auc = load_roc_curve(outputs_dir, "logistic_regression")
    assert len(fpr) == len(tpr)
    assert 0.0 <= auc <= 1.0


def test_missing_outputs_dir_raises_clear_error(tmp_path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(PipelineOutputsMissing):
        load_dataset_summary(missing)
