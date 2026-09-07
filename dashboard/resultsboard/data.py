"""Reads pipeline/outputs/* into plain Python data for the results
dashboard. No Django models, no database -- always reflects the latest
files on disk."""
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import auc as auc_fn
from sklearn.metrics import roc_curve

_META_KEYS = {"_majority_class_baseline_accuracy", "_dropped_all_nan_feature_columns"}


class PipelineOutputsMissing(Exception):
    """Raised when pipeline/outputs/ (or a required file inside it) doesn't exist yet."""


def _require(path: Path):
    if not path.exists():
        raise PipelineOutputsMissing(f"expected pipeline output not found: {path}")


def load_dataset_summary(outputs_dir: Path) -> dict:
    dataset_path = outputs_dir / "dataset.csv"
    _require(dataset_path)
    df = pd.read_csv(dataset_path)

    non_feature_cols = {"company_folder", "quarter_end", "target"}
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    active_features = [c for c in feature_cols if not df[c].isna().all()]
    dropped_features = [c for c in feature_cols if df[c].isna().all()]

    return {
        "row_count": len(df),
        "feature_count": len(feature_cols),
        "active_features": active_features,
        "dropped_features": dropped_features,
        "positive_rate": float(df["target"].mean()),
        "company_count": df["company_folder"].nunique(),
    }


def load_model_metrics(outputs_dir: Path):
    metrics_path = outputs_dir / "metrics.json"
    _require(metrics_path)
    all_metrics = json.loads(metrics_path.read_text())

    baseline = all_metrics.get("_majority_class_baseline_accuracy")
    model_metrics = {k: v for k, v in all_metrics.items() if k not in _META_KEYS}
    return model_metrics, baseline


def load_model_names(outputs_dir: Path) -> list[str]:
    model_metrics, _ = load_model_metrics(outputs_dir)
    return list(model_metrics.keys())


def load_roc_curve(outputs_dir: Path, model_name: str):
    pred_path = outputs_dir / "predictions" / f"{model_name}.csv"
    _require(pred_path)
    pred = pd.read_csv(pred_path)
    fpr, tpr, _ = roc_curve(pred["y_true"], pred["y_proba"])
    return fpr.tolist(), tpr.tolist(), float(auc_fn(fpr, tpr))
