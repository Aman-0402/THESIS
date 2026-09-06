"""Chronological split, train-only imputation, eight classifiers,
held-out test metrics. documentation .pdf sections 8, 13-14.

NaN-handling note (investigated via `pd.read_csv(...).isna().mean()` on the
real `pipeline/outputs/dataset.csv` built by Task 8): four feature columns
are entirely NaN for all 3439 rows -- `volume_change_60d` (no volume data
survived into stock_clean), and `debt_to_equity`, `price_to_book`,
`price_to_earnings_proxy` (all three depend on `market_cap`/`total_debt`,
which Task 6 found have no clean source field across any of the 29
companies' financial statements -- a real data-coverage gap, not a bug;
see the docstring of `_04_features_target.py`). Train-only median
imputation cannot fix an all-NaN column: `median()` of an all-NaN series is
NaN, so `.fillna(NaN)` is a no-op and the column stays entirely NaN, which
none of the eight specified sklearn estimators can `.fit()` on. The other
fundamental columns have real but PARTIAL missingness (up to ~82%) and are
handled fine by ordinary train-median imputation -- only the four fully-NaN
columns need special handling.

Chosen approach: drop the four all-NaN columns from the modeling feature
set right before fitting, and report exactly which ones were dropped. This
is done locally within `main()` (`ACTIVE_FEATURE_COLS`) rather than by
mutating the module-level `FEATURE_COLS` constant, since `FEATURE_COLS` is
part of this module's public contract (mirrors `MARKET_FEATURE_NAMES` +
`FUNDAMENTAL_FEATURE_NAMES` from `pipeline.lib.features`) and other code /
tests may reasonably expect it to list the full 22 features regardless of
which happen to be all-NaN in any one run of the real dataset.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from pipeline.lib.features import MARKET_FEATURE_NAMES, FUNDAMENTAL_FEATURE_NAMES

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
FEATURE_COLS = MARKET_FEATURE_NAMES + FUNDAMENTAL_FEATURE_NAMES
RANDOM_STATE = 42

VAL_START = pd.Timestamp("2022-01-01")
TEST_START = pd.Timestamp("2024-01-01")


def chronological_split(df: pd.DataFrame):
    train = df[df["quarter_end"] < VAL_START]
    val = df[(df["quarter_end"] >= VAL_START) & (df["quarter_end"] < TEST_START)]
    test = df[df["quarter_end"] >= TEST_START]
    return train, val, test


def impute_with_train_medians(train, val, test, cols):
    medians = train[cols].median()
    return (train[cols].fillna(medians), val[cols].fillna(medians), test[cols].fillna(medians), medians)


# class_weight="balanced" only where the estimator supports it (GaussianNB,
# GradientBoostingClassifier, KNeighborsClassifier, MLPClassifier don't
# expose that param)
MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
    "random_forest": RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
    "naive_bayes": GaussianNB(),
    "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "svm": SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE),
    "neural_network": MLPClassifier(max_iter=2000, random_state=RANDOM_STATE),
    "knn": KNeighborsClassifier(),
    "decision_tree": DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
}


def evaluate(y_true, y_pred, y_proba):
    cm = confusion_matrix(y_true, y_pred).tolist()
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        # cm rows/cols ordered [0,1] (sklearn default sorted labels); row 0 =
        # actual-negative -> specificity = TN/(TN+FP)
        "specificity": cm[0][0] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) else None,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": cm,
    }
    if y_proba is not None and len(set(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    return metrics


def main():
    dataset = pd.read_csv(OUTPUT_DIR / "dataset.csv", parse_dates=["quarter_end"])
    train, val, test = chronological_split(dataset)

    # Drop feature columns that are entirely NaN in the training split --
    # train-median imputation cannot fill an all-NaN column (median of an
    # all-NaN series is NaN), and sklearn estimators cannot fit on NaN.
    all_nan_cols = [c for c in FEATURE_COLS if train[c].isna().all()]
    active_feature_cols = [c for c in FEATURE_COLS if c not in all_nan_cols]
    if all_nan_cols:
        print(f"Dropping all-NaN feature columns (no train-set data to impute from): {all_nan_cols}")

    x_train, x_val, x_test, _ = impute_with_train_medians(train, val, test, active_feature_cols)
    y_train, y_val, y_test = train["target"], val["target"], test["target"]

    print(f"Row counts: train={len(train)}, val={len(val)}, test={len(test)}")

    (OUTPUT_DIR / "predictions").mkdir(parents=True, exist_ok=True)
    all_metrics = {}

    for name, model in MODELS.items():
        model.fit(x_train, y_train)
        val_pred = model.predict(x_val)
        val_acc = accuracy_score(y_val, val_pred) if len(y_val) else None

        test_pred = model.predict(x_test)
        test_proba = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else None

        pred_df = test[["company_folder", "quarter_end"]].copy()
        pred_df["y_true"] = y_test.values
        pred_df["y_pred"] = test_pred
        pred_df.to_csv(OUTPUT_DIR / "predictions" / f"{name}.csv", index=False)

        all_metrics[name] = {"validation_accuracy": val_acc, **evaluate(y_test, test_pred, test_proba)}
        print(f"{name}: test accuracy={all_metrics[name]['accuracy']:.3f}")

    baseline_acc = max(y_test.mean(), 1 - y_test.mean()) if len(y_test) else None
    all_metrics["_majority_class_baseline_accuracy"] = baseline_acc
    all_metrics["_dropped_all_nan_feature_columns"] = all_nan_cols

    with (OUTPUT_DIR / "metrics.json").open("w") as f:
        json.dump(all_metrics, f, indent=2, default=float)


if __name__ == "__main__":
    main()
