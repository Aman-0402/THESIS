from pathlib import Path

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
        return {"summary": summary}

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


# TODO(Task 6): replace with the real model-detail view (metrics, confusion matrix, ROC curve for one model)
def model_detail(request, model_name):
    return render(request, "resultsboard/model_detail_placeholder.html", {"model_name": model_name})
