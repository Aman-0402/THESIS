import json
from pathlib import Path

from django.http import Http404
from django.shortcuts import render

from . import data

PIPELINE_OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "pipeline" / "outputs"


def _render_or_missing(request, template_name, context_fn):
    try:
        context = context_fn()
    except data.PipelineOutputsMissing as exc:
        return render(request, "resultsboard/missing_outputs.html", {"error": str(exc)}, status=503)
    return render(request, template_name, context)


def overview(request):
    def build_context():
        summary = data.load_dataset_summary(PIPELINE_OUTPUTS_DIR)
        companies = data.load_company_breakdown(PIPELINE_OUTPUTS_DIR)
        return {"summary": summary, "companies": companies}

    return _render_or_missing(request, "resultsboard/overview.html", build_context)


def model_comparison(request):
    def build_context():
        metrics, baseline = data.load_model_metrics(PIPELINE_OUTPUTS_DIR)
        rows = [{"name": name, **m} for name, m in metrics.items()]
        rows.sort(key=lambda r: r["accuracy"] if r["accuracy"] is not None else -1, reverse=True)
        return {
            "rows": rows,
            "baseline": baseline,
            "chart_labels": [r["name"] for r in rows],
            "chart_accuracies": [r["accuracy"] for r in rows],
        }

    return _render_or_missing(request, "resultsboard/model_comparison.html", build_context)


def model_detail(request, model_name):
    def build_context():
        metrics, baseline = data.load_model_metrics(PIPELINE_OUTPUTS_DIR)
        if model_name not in metrics:
            raise Http404(f"unknown model: {model_name}")
        fpr, tpr, auc = data.load_roc_curve(PIPELINE_OUTPUTS_DIR, model_name)
        cm = metrics[model_name]["confusion_matrix"]
        roc_points = [{"x": x, "y": y} for x, y in zip(fpr, tpr)]
        return {
            "model_name": model_name,
            "metrics": metrics[model_name],
            "baseline": baseline,
            "cm": cm,
            "roc_points": json.dumps(roc_points),
            "roc_auc": auc,
        }

    return _render_or_missing(request, "resultsboard/model_detail.html", build_context)
