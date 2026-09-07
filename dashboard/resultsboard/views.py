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


# TODO(Task 5): replace with the real model-comparison view
def model_comparison(request):
    return render(request, "resultsboard/model_comparison_placeholder.html")
