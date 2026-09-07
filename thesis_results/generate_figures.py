"""Generate figures for the thesis results write-up from pipeline/outputs/.
One-off analysis script, not part of the pipeline itself -- run after
pipeline/_05_split_train.py has produced metrics.json and predictions/*.csv.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "pipeline" / "outputs"
FIGURES = Path(__file__).resolve().parent / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "naive_bayes": "Naive Bayes",
    "gradient_boosting": "Gradient Boosting",
    "svm": "SVM",
    "neural_network": "Neural Network (MLP)",
    "knn": "KNN",
    "decision_tree": "Decision Tree",
}


def load_metrics():
    all_metrics = json.loads((OUTPUTS / "metrics.json").read_text())
    baseline = all_metrics["_majority_class_baseline_accuracy"]
    models = {k: v for k, v in all_metrics.items() if not k.startswith("_")}
    return models, baseline


def plot_model_comparison(models: dict, baseline: float):
    names = list(models.keys())
    accs = [models[n]["accuracy"] for n in names]
    order = sorted(range(len(names)), key=lambda i: accs[i], reverse=True)
    names = [names[i] for i in order]
    accs = [accs[i] for i in order]
    labels = [MODEL_LABELS[n] for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, accs, color="#4C72B0")
    ax.axhline(baseline, color="#C44E52", linestyle="--", label=f"Majority-class baseline ({baseline:.3f})")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Test accuracy by model")
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.01, f"{acc:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(FIGURES / "model_comparison.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrices(models: dict):
    names = list(models.keys())
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, name in zip(axes.flat, names):
        cm = models[name]["confusion_matrix"]
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=11)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["Actual 0", "Actual 1"])
        ax.set_title(MODEL_LABELS[name], fontsize=10)
    plt.tight_layout()
    fig.savefig(FIGURES / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def plot_roc_curves(models: dict):
    fig, ax = plt.subplots(figsize=(7, 7))
    for name in models:
        pred = pd.read_csv(OUTPUTS / "predictions" / f"{name}.csv")
        fpr, tpr, _ = roc_curve(pred["y_true"], pred["y_proba"])
        auc = models[name]["roc_auc"]
        ax.plot(fpr, tpr, label=f"{MODEL_LABELS[name]} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random baseline (AUC=0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves, all 8 models (test set)")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    fig.savefig(FIGURES / "roc_curves.png", dpi=150)
    plt.close(fig)


def main():
    models, baseline = load_metrics()
    plot_model_comparison(models, baseline)
    plot_confusion_matrices(models)
    plot_roc_curves(models)
    print(f"Figures written to {FIGURES}")


if __name__ == "__main__":
    main()
