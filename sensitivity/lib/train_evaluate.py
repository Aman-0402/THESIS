# sensitivity/lib/train_evaluate.py
"""Config-driven missing-data handling, reusing pipeline._05_split_train's
classifiers/metrics/split logic directly."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from pipeline._05_split_train import MODELS, chronological_split, evaluate


def apply_missing_strategy(df: pd.DataFrame, feature_cols: list[str], strategy: str):
    """Returns (df_with_features_imputed_or_filtered, medians_used).
    Operates on ONE split (call separately for train/val/test, always
    computing medians from the TRAIN split and reusing them for val/test --
    see train_and_evaluate() below)."""
    if strategy == "impute":
        medians = df[feature_cols].median()
        out = df.copy()
        out[feature_cols] = out[feature_cols].fillna(medians)
        return out, medians

    if strategy == "drop_any":
        out = df.dropna(subset=feature_cols).copy()
        return out, None

    if strategy == "drop_majority":
        missing_frac = df[feature_cols].isna().mean(axis=1)
        kept = df[missing_frac <= 0.5].copy()
        medians = kept[feature_cols].median()
        kept[feature_cols] = kept[feature_cols].fillna(medians)
        return kept, medians

    raise ValueError(f"unknown missing strategy: {strategy}")


def train_and_evaluate(dataset: pd.DataFrame, config: dict) -> dict:
    """Full train/val/test pipeline for one sensitivity config's dataset.
    dataset must have a "period_end" column (renamed here to "quarter_end"
    to reuse pipeline._05_split_train.chronological_split unchanged, since
    that function's split boundaries are date-based and don't care what the
    column is literally named as long as it's called quarter_end)."""
    df = dataset.rename(columns={"period_end": "quarter_end"})
    feature_cols = [c for c in df.columns if c not in ("company_folder", "quarter_end", "target")]

    train, val, test = chronological_split(df)

    all_nan_cols = [c for c in feature_cols if train[c].isna().all()]
    active_cols = [c for c in feature_cols if c not in all_nan_cols]

    train_processed, medians = apply_missing_strategy(train, active_cols, config["missing_strategy"])

    if config["missing_strategy"] == "drop_any":
        # no medians to reuse, so val/test also drop incomplete rows
        val_processed = val.dropna(subset=active_cols).copy()
        test_processed = test.dropna(subset=active_cols).copy()
    elif config["missing_strategy"] == "drop_majority":
        # apply the same >50%-missing row filter to val/test (using each split's
        # own missingness pattern), then fill the rest with train-derived medians
        val_missing_frac = val[active_cols].isna().mean(axis=1)
        val_processed = val[val_missing_frac <= 0.5].copy()
        val_processed[active_cols] = val_processed[active_cols].fillna(medians)
        test_missing_frac = test[active_cols].isna().mean(axis=1)
        test_processed = test[test_missing_frac <= 0.5].copy()
        test_processed[active_cols] = test_processed[active_cols].fillna(medians)
    else:  # "impute"
        val_processed = val.copy()
        val_processed[active_cols] = val_processed[active_cols].fillna(medians)
        test_processed = test.copy()
        test_processed[active_cols] = test_processed[active_cols].fillna(medians)

    x_train, y_train = train_processed[active_cols], train_processed["target"]
    x_val, y_val = val_processed[active_cols], val_processed["target"]
    x_test, y_test = test_processed[active_cols], test_processed["target"]

    if len(x_test) == 0 or x_test[active_cols].isna().any().any() or y_test.nunique() < 2:
        return {
            "run_id": config["run_id"], "row_count": len(df),
            "train_rows": len(x_train), "val_rows": len(x_val), "test_rows": len(x_test),
            "active_feature_count": len(active_cols), "dropped_feature_count": len(all_nan_cols),
            "models": {}, "baseline_accuracy": None, "best_model": None, "best_accuracy": None,
            "beats_baseline": None,
            "note": "test set too small/degenerate for this config to evaluate",
        }

    # Reuses the same MODELS instances across all 12 configs (run_all.py
    # calls train_and_evaluate once per config, sequentially). Each
    # model.fit() call fully overwrites that estimator's previously fitted
    # state -- safe to re-fit sklearn estimators repeatedly like this, and
    # avoids duplicating the 8-classifier definitions from pipeline/_05_split_train.py.
    model_metrics = {}
    for name, model in MODELS.items():
        model.fit(x_train, y_train)
        test_pred = model.predict(x_test)
        test_proba = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else None
        model_metrics[name] = evaluate(y_test, test_pred, test_proba)

    baseline_accuracy = max(y_test.mean(), 1 - y_test.mean())
    best_name = max(model_metrics, key=lambda n: model_metrics[n]["accuracy"])

    return {
        "run_id": config["run_id"],
        "row_count": len(df),
        "train_rows": len(x_train), "val_rows": len(x_val), "test_rows": len(x_test),
        "active_feature_count": len(active_cols), "dropped_feature_count": len(all_nan_cols),
        "models": model_metrics,
        "baseline_accuracy": baseline_accuracy,
        "best_model": best_name,
        "best_accuracy": model_metrics[best_name]["accuracy"],
        "beats_baseline": model_metrics[best_name]["accuracy"] > baseline_accuracy,
    }
