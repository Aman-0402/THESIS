# Results Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local Django app that displays the ML pipeline's results (dataset summary, 8-model comparison, per-model confusion matrix + ROC curve) for the thesis defense, per `docs/superpowers/specs/2026-09-07-results-dashboard-design.md`.

**Architecture:** One Django project (`dashboard/`) with one app (`resultsboard/`). No database/models/migrations — `resultsboard/data.py` reads `pipeline/outputs/*` directly into plain dicts/lists on each request; views stay thin; templates render with Chart.js for charts. A small pipeline addition (`y_proba` column in prediction CSVs) is done first so real ROC curves can be drawn.

**Tech Stack:** Django 4.2 (already installed), Chart.js (CDN), Python's `csv`/`json` stdlib + `sklearn.metrics.roc_curve` for ROC computation. No REST framework, no ORM.

---

## File Structure

```
dashboard/                     # Django project root (manage.py lives here)
  manage.py
  dashboard/
    __init__.py
    settings.py
    urls.py
    wsgi.py
  resultsboard/
    __init__.py
    apps.py
    data.py                    # all pipeline-output parsing (pure functions)
    views.py                   # thin views calling data.py
    urls.py
    templates/resultsboard/
      base.html
      overview.html
      model_comparison.html
      model_detail.html
      missing_outputs.html     # shown if pipeline/outputs/ doesn't exist
    static/resultsboard/
      style.css
tests/
  resultsboard/
    test_data.py
pipeline/
  _05_split_train.py           # modified: add y_proba column to predictions
```

---

## Task 1: Add `y_proba` to prediction output

**Files:**
- Modify: `pipeline/_05_split_train.py`
- Test: `tests/pipeline/test_split_train.py` (add one test)

**Context:** `predictions/<model>.csv` currently has `company_folder, quarter_end, y_true, y_pred`. Need `y_proba` (positive-class probability) added so the dashboard can compute real ROC curves via `sklearn.metrics.roc_curve`, instead of only showing the scalar AUC already in `metrics.json`. All 8 configured estimators support `predict_proba` (`SVC` is constructed with `probability=True`), so this column will always be populated — no `None`/missing case to handle.

- [ ] **Step 1: Read the current `main()` prediction-writing block**

Find this block in `pipeline/_05_split_train.py` (inside the `for name, model in MODELS.items():` loop):

```python
        pred_df = test[["company_folder", "quarter_end"]].copy()
        pred_df["y_true"] = y_test.values
        pred_df["y_pred"] = test_pred
        pred_df.to_csv(OUTPUT_DIR / "predictions" / f"{name}.csv", index=False)
```

- [ ] **Step 2: Write a failing test**

Add to `tests/pipeline/test_split_train.py`:

```python
def test_predictions_csv_has_y_proba_column(tmp_path, monkeypatch):
    """End-to-end-lite: run main() against a tiny synthetic dataset and
    confirm the written predictions CSV has a y_proba column with values
    in [0, 1]."""
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

    monkeypatch.setattr(mod, "OUTPUT_DIR", outputs_dir)
    monkeypatch.setattr(mod, "VAL_START", pd.Timestamp("2020-06-01"))
    monkeypatch.setattr(mod, "TEST_START", pd.Timestamp("2021-01-01"))

    mod.main()

    pred = pd.read_csv(outputs_dir / "predictions" / "logistic_regression.csv")
    assert "y_proba" in pred.columns
    assert pred["y_proba"].between(0, 1).all()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/pipeline/test_split_train.py -v`
Expected: FAIL — `AssertionError` (no `y_proba` column) or `KeyError`.

- [ ] **Step 4: Modify `main()` to add the column**

Replace the block from Step 1 with:

```python
        test_proba_col = (
            test_proba if test_proba is not None
            else model.decision_function(x_test)
        )
        pred_df = test[["company_folder", "quarter_end"]].copy()
        pred_df["y_true"] = y_test.values
        pred_df["y_pred"] = test_pred
        pred_df["y_proba"] = test_proba_col
        pred_df.to_csv(OUTPUT_DIR / "predictions" / f"{name}.csv", index=False)
```

(`test_proba` is already computed a few lines earlier in the existing loop via `model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else None` — this reuses it, falling back to `decision_function` only in the theoretical case an estimator lacks `predict_proba`, which doesn't currently happen with this exact `MODELS` dict but keeps the code correct if that ever changes.)

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/pipeline/test_split_train.py -v`
Expected: PASS (2 tests: the original boundary test + this new one)

- [ ] **Step 6: Re-run the real pipeline stage and confirm no behavior change**

Run: `python pipeline/_05_split_train.py`
Expected: same 8 accuracy lines as before (0.498, 0.544, 0.425, 0.533, 0.421, 0.575, 0.552, 0.429) — only the CSV gains a column, nothing else changes. Then run `python -m pytest tests/pipeline -v` to confirm the full suite (38 tests now) still passes.

- [ ] **Step 7: Commit**

```bash
git add pipeline/_05_split_train.py tests/pipeline/test_split_train.py
git commit -m "feat(pipeline): add y_proba to prediction CSVs for ROC curve plotting"
```

---

## Task 2: Django project scaffold

**Files:**
- Create: `dashboard/manage.py`
- Create: `dashboard/dashboard/__init__.py`, `settings.py`, `urls.py`, `wsgi.py`
- Create: `dashboard/resultsboard/__init__.py`, `apps.py`, `urls.py`, `views.py` (stub)

- [ ] **Step 1: Scaffold via django-admin**

Run from `D:\code\GITHUB\THESIS`:
```bash
django-admin startproject dashboard
cd dashboard
python manage.py startapp resultsboard
cd ..
```

- [ ] **Step 2: Register the app**

In `dashboard/dashboard/settings.py`, add `"resultsboard"` to `INSTALLED_APPS`.

- [ ] **Step 3: Point `DATABASES` at nothing meaningful (no DB used)**

Leave the default SQLite config as Django generates it — don't remove it (Django's admin/auth apps still reference it internally even though we won't run migrations for our own app or use the DB in views). Do NOT run `python manage.py migrate` — this project has zero custom models and we're not using Django auth/sessions/admin, so an unmigrated DB file is fine and simpler than deciding what to do with one.

- [ ] **Step 4: Wire up empty routing so the server starts**

`dashboard/resultsboard/urls.py`:
```python
from django.urls import path
from . import views

app_name = "resultsboard"
urlpatterns = [
    path("", views.overview, name="overview"),
]
```

`dashboard/resultsboard/views.py` (stub for now, filled in Task 4):
```python
from django.http import HttpResponse


def overview(request):
    return HttpResponse("placeholder")
```

`dashboard/dashboard/urls.py` — add the include:
```python
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("resultsboard.urls")),
]
```

- [ ] **Step 5: Verify the server starts**

Run: `cd dashboard && python manage.py runserver` (in background or with a short timeout), then `curl http://127.0.0.1:8000/` from another terminal — expect to see `placeholder`. Stop the server.

- [ ] **Step 6: Commit**

```bash
git add dashboard/
git commit -m "chore(dashboard): scaffold Django project and resultsboard app"
```

---

## Task 3: `data.py` — pipeline-output parsing

**Files:**
- Create: `dashboard/resultsboard/data.py`
- Test: `tests/resultsboard/test_data.py`

**Context:** This is the one module with real logic — everything else in this plan is thin views/templates. All functions take a `pipeline_outputs_dir: Path` argument (don't hardcode the path inside `data.py`) so tests can point at a `tmp_path` fixture instead of the real `pipeline/outputs/`.

- [ ] **Step 1: Write failing tests**

```python
# tests/resultsboard/test_data.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd dashboard && python -m pytest ../tests/resultsboard/test_data.py -v` (or configure `pytest.ini`/`conftest.py` so `resultsboard` is importable — see Step 3 note below)

Expected: FAIL — `ModuleNotFoundError: No module named 'resultsboard'`

**Note on running these tests:** `resultsboard` is a Django app living under `dashboard/`. Add a `dashboard/conftest.py`:
```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
```
and run tests from the repo root as `python -m pytest tests/resultsboard dashboard/resultsboard -v` — or simpler, just run `cd dashboard && python -m pytest ../tests/resultsboard -v` after adding that `conftest.py` inside `dashboard/`. Pick whichever actually works in your environment and verify it before moving on; report if pytest's rootdir/import mechanics need a different fix.

- [ ] **Step 3: Write `data.py`**

```python
# dashboard/resultsboard/data.py
"""Reads pipeline/outputs/* into plain Python data for the results
dashboard. No Django models, no database -- always reflects the latest
files on disk."""
import json
from pathlib import Path

import pandas as pd
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
    from sklearn.metrics import auc as auc_fn
    return fpr.tolist(), tpr.tolist(), float(auc_fn(fpr, tpr))
```

- [ ] **Step 4: Run test to verify it passes**

Run the same command as Step 2. Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add dashboard/resultsboard/data.py tests/resultsboard/test_data.py dashboard/conftest.py
git commit -m "feat(dashboard): add pipeline-output parsing module with tests"
```

---

## Task 4: Overview page

**Files:**
- Modify: `dashboard/resultsboard/views.py`
- Create: `dashboard/resultsboard/templates/resultsboard/base.html`
- Create: `dashboard/resultsboard/templates/resultsboard/overview.html`
- Create: `dashboard/resultsboard/templates/resultsboard/missing_outputs.html`

- [ ] **Step 1: Define the outputs-dir constant and error-handling pattern**

At the top of `dashboard/resultsboard/views.py`:
```python
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
```

- [ ] **Step 2: Implement `overview` view**

```python
def overview(request):
    def build_context():
        summary = data.load_dataset_summary(PIPELINE_OUTPUTS_DIR)
        return {"summary": summary}

    return _render_or_missing(request, "resultsboard/overview.html", build_context)
```

- [ ] **Step 3: Write `base.html`**

```html
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{% block title %}Thesis Results{% endblock %}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    body { font-family: system-ui, sans-serif; margin: 2rem; max-width: 900px; }
    nav a { margin-right: 1rem; }
    table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
    th, td { border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: right; }
    th:first-child, td:first-child { text-align: left; }
    .baseline-row { background: #fff3cd; font-weight: bold; }
  </style>
</head>
<body>
  <nav>
    <a href="{% url 'resultsboard:overview' %}">Overview</a>
    <a href="{% url 'resultsboard:model_comparison' %}">Model Comparison</a>
  </nav>
  <hr>
  {% block content %}{% endblock %}
</body>
</html>
```

- [ ] **Step 4: Write `overview.html`**

```html
{% extends "resultsboard/base.html" %}
{% block title %}Overview{% endblock %}
{% block content %}
  <h1>Dataset Overview</h1>
  <ul>
    <li>Rows: {{ summary.row_count }}</li>
    <li>Companies: {{ summary.company_count }}</li>
    <li>Positive rate: {{ summary.positive_rate|floatformat:3 }}</li>
    <li>Active features: {{ summary.active_features|length }} / {{ summary.feature_count }}</li>
  </ul>

  <h2>Active features</h2>
  <p>{{ summary.active_features|join:", " }}</p>

  <h2>Dropped (always-missing) features</h2>
  <p>{{ summary.dropped_features|join:", " }}</p>

  <h2>Known data limitations</h2>
  <ul>
    <li>Laboratorios_Rovi has zero daily-price rows in the source archive and contributes no dataset rows.</li>
    <li>Zenotech_Laboratories has zero financial-statement rows; its market features still compute, but fundamentals are always None.</li>
    <li>market_cap and total_debt have no clean source field anywhere, so debt_to_equity, price_to_book, and price_to_earnings_proxy are None for every row.</li>
    <li>Dataset size (this run) differs from the original thesis's reported 1,410 rows because several companies' real cleaned stock series start decades earlier than the thesis's assumed 2014Q3 start.</li>
  </ul>
{% endblock %}
```

- [ ] **Step 5: Write `missing_outputs.html`**

```html
{% extends "resultsboard/base.html" %}
{% block title %}Pipeline not run{% endblock %}
{% block content %}
  <h1>Pipeline outputs not found</h1>
  <p>{{ error }}</p>
  <p>Run the pipeline first:</p>
  <pre>python pipeline/_01_load_raw.py
python pipeline/_02_clean_stock.py
python pipeline/_03_build_panel.py
python pipeline/_04_features_target.py
python pipeline/_05_split_train.py</pre>
{% endblock %}
```

- [ ] **Step 6: Verify manually**

Run: `cd dashboard && python manage.py runserver`, visit `http://127.0.0.1:8000/` in a browser (or `curl`). Expected: renders the overview page with real numbers from `pipeline/outputs/dataset.csv` (3,439 rows etc.) — this requires the real pipeline outputs to already exist on disk, which they do from the earlier merged work. Stop the server after checking.

- [ ] **Step 7: Commit**

```bash
git add dashboard/resultsboard/views.py dashboard/resultsboard/templates/
git commit -m "feat(dashboard): add overview page"
```

---

## Task 5: Model comparison page

**Files:**
- Modify: `dashboard/resultsboard/views.py`
- Modify: `dashboard/resultsboard/urls.py`
- Create: `dashboard/resultsboard/templates/resultsboard/model_comparison.html`

- [ ] **Step 1: Add the URL**

In `dashboard/resultsboard/urls.py`, add:
```python
    path("models/", views.model_comparison, name="model_comparison"),
```

- [ ] **Step 2: Implement the view**

```python
def model_comparison(request):
    def build_context():
        metrics, baseline = data.load_model_metrics(PIPELINE_OUTPUTS_DIR)
        rows = [{"name": name, **m} for name, m in metrics.items()]
        rows.sort(key=lambda r: r["accuracy"], reverse=True)
        return {
            "rows": rows,
            "baseline": baseline,
            "chart_labels": [r["name"] for r in rows],
            "chart_accuracies": [r["accuracy"] for r in rows],
        }

    return _render_or_missing(request, "resultsboard/model_comparison.html", build_context)
```

- [ ] **Step 3: Write the template**

```html
{% extends "resultsboard/base.html" %}
{% block title %}Model Comparison{% endblock %}
{% block content %}
  <h1>Model Comparison</h1>
  <p>Majority-class baseline accuracy: <strong>{{ baseline|floatformat:3 }}</strong></p>

  <table>
    <tr>
      <th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th>
      <th>Specificity</th><th>F1</th><th>ROC-AUC</th><th>Val. Accuracy</th>
    </tr>
    {% for row in rows %}
    <tr>
      <td><a href="{% url 'resultsboard:model_detail' row.name %}">{{ row.name }}</a></td>
      <td>{{ row.accuracy|floatformat:3 }}</td>
      <td>{{ row.precision|floatformat:3 }}</td>
      <td>{{ row.recall|floatformat:3 }}</td>
      <td>{{ row.specificity|floatformat:3 }}</td>
      <td>{{ row.f1|floatformat:3 }}</td>
      <td>{{ row.roc_auc|default_if_none:"-"|floatformat:3 }}</td>
      <td>{{ row.validation_accuracy|floatformat:3 }}</td>
    </tr>
    {% endfor %}
  </table>

  <canvas id="accuracyChart" width="700" height="300"></canvas>
  <script>
    const ctx = document.getElementById('accuracyChart');
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: {{ chart_labels|safe }},
        datasets: [{ label: 'Test Accuracy', data: {{ chart_accuracies|safe }} }]
      },
      options: {
        scales: { y: { beginAtZero: true, max: 1 } },
        plugins: {
          annotation: {}
        }
      }
    });
  </script>
  <p><em>Baseline accuracy ({{ baseline|floatformat:3 }}) is shown in the table above; no model in this run exceeds it.</em></p>
{% endblock %}
```

(Note: a horizontal baseline reference line on the chart itself would need the `chartjs-plugin-annotation` package — skip that for simplicity per the design's "no model exceeds baseline" being stated in text already; the table's majority-baseline line plus the caption is sufficient for the defense.)

- [ ] **Step 4: Verify manually**

Run: `cd dashboard && python manage.py runserver`, visit `http://127.0.0.1:8000/models/`. Expected: table of all 8 models sorted by accuracy descending, plus a bar chart. Stop the server.

- [ ] **Step 5: Commit**

```bash
git add dashboard/resultsboard/views.py dashboard/resultsboard/urls.py dashboard/resultsboard/templates/resultsboard/model_comparison.html
git commit -m "feat(dashboard): add model comparison page"
```

---

## Task 6: Model detail page

**Files:**
- Modify: `dashboard/resultsboard/views.py`
- Modify: `dashboard/resultsboard/urls.py`
- Create: `dashboard/resultsboard/templates/resultsboard/model_detail.html`

- [ ] **Step 1: Add the URL**

```python
    path("models/<str:model_name>/", views.model_detail, name="model_detail"),
```

- [ ] **Step 2: Implement the view**

```python
from django.http import Http404


def model_detail(request, model_name):
    def build_context():
        metrics, baseline = data.load_model_metrics(PIPELINE_OUTPUTS_DIR)
        if model_name not in metrics:
            raise Http404(f"unknown model: {model_name}")
        fpr, tpr, auc = data.load_roc_curve(PIPELINE_OUTPUTS_DIR, model_name)
        cm = metrics[model_name]["confusion_matrix"]
        return {
            "model_name": model_name,
            "metrics": metrics[model_name],
            "baseline": baseline,
            "cm": cm,
            "roc_fpr": fpr,
            "roc_tpr": tpr,
            "roc_auc": auc,
        }

    return _render_or_missing(request, "resultsboard/model_detail.html", build_context)
```

- [ ] **Step 3: Write the template**

```html
{% extends "resultsboard/base.html" %}
{% block title %}{{ model_name }}{% endblock %}
{% block content %}
  <h1>{{ model_name }}</h1>
  <p><a href="{% url 'resultsboard:model_comparison' %}">&larr; back to comparison</a></p>

  <h2>Metrics</h2>
  <ul>
    <li>Accuracy: {{ metrics.accuracy|floatformat:3 }} (baseline: {{ baseline|floatformat:3 }})</li>
    <li>Precision: {{ metrics.precision|floatformat:3 }}</li>
    <li>Recall: {{ metrics.recall|floatformat:3 }}</li>
    <li>Specificity: {{ metrics.specificity|floatformat:3 }}</li>
    <li>F1: {{ metrics.f1|floatformat:3 }}</li>
    <li>ROC-AUC: {{ roc_auc|floatformat:3 }}</li>
  </ul>

  <h2>Confusion Matrix</h2>
  <table>
    <tr><th></th><th>Predicted Negative</th><th>Predicted Positive</th></tr>
    <tr><th>Actual Negative</th><td>{{ cm.0.0 }} (TN)</td><td>{{ cm.0.1 }} (FP)</td></tr>
    <tr><th>Actual Positive</th><td>{{ cm.1.0 }} (FN)</td><td>{{ cm.1.1 }} (TP)</td></tr>
  </table>

  <h2>ROC Curve</h2>
  <canvas id="rocChart" width="500" height="500"></canvas>
  <script>
    const ctx = document.getElementById('rocChart');
    new Chart(ctx, {
      type: 'line',
      data: {
        labels: {{ roc_fpr|safe }},
        datasets: [
          {
            label: 'ROC (AUC = {{ roc_auc|floatformat:3 }})',
            data: {{ roc_tpr|safe }},
            fill: false,
            pointRadius: 0,
          },
          {
            label: 'Random baseline',
            data: [0, 1],
            borderDash: [5, 5],
            pointRadius: 0,
          }
        ]
      },
      options: {
        scales: {
          x: { title: { display: true, text: 'False Positive Rate' }, min: 0, max: 1 },
          y: { title: { display: true, text: 'True Positive Rate' }, min: 0, max: 1 }
        }
      }
    });
  </script>
{% endblock %}
```

- [ ] **Step 4: Verify manually**

Run: `cd dashboard && python manage.py runserver`, visit `http://127.0.0.1:8000/models/random_forest/` (and 2-3 others). Expected: metrics list, confusion matrix table, ROC curve chart rendering a real curve (not a straight diagonal, given AUC values like 0.484-0.541 from the real run). Also visit `/models/not_a_real_model/` and confirm a 404, not a crash. Stop the server.

- [ ] **Step 5: Commit**

```bash
git add dashboard/resultsboard/views.py dashboard/resultsboard/urls.py dashboard/resultsboard/templates/resultsboard/model_detail.html
git commit -m "feat(dashboard): add model detail page with confusion matrix and ROC curve"
```

---

## Task 7: Final polish — settings, requirements, README, full smoke test

**Files:**
- Modify: `dashboard/dashboard/settings.py`
- Modify: `requirements.txt`
- Create: `dashboard/README.md`

- [ ] **Step 1: Point Django at the templates correctly and tidy settings**

Confirm `dashboard/dashboard/settings.py`'s `TEMPLATES` setting has `APP_DIRS: True` (Django's `startapp`/`startproject` default already sets this — just verify, don't need to change it unless it's missing). Set `ALLOWED_HOSTS = ["127.0.0.1", "localhost"]` (fine for local-only use per the design's scope). Leave `SECRET_KEY` as generated by `startproject` — this never leaves your machine.

- [ ] **Step 2: Add Django and scikit-learn's `roc_curve`/`auc` dependency to `requirements.txt`**

`scikit-learn` is already listed (used by the pipeline). Add:
```
django==4.2.21
```
(matching the version already installed, confirmed via `python -c "import django; print(django.get_version())"`.)

- [ ] **Step 3: Write `dashboard/README.md`**

```markdown
# Results Dashboard

Local-only Django app displaying the ML pipeline's results. No database,
no auth, no deployment — reads `pipeline/outputs/` directly on each request.

## Run

Make sure the pipeline has been run first (see `../pipeline/README.md`),
then:

    cd dashboard
    python manage.py runserver

Visit http://127.0.0.1:8000/

## Pages

- `/` — dataset overview, feature list, known data limitations
- `/models/` — 8-model comparison table + accuracy bar chart
- `/models/<name>/` — confusion matrix + ROC curve for one model

## Tests

    python -m pytest ../tests/resultsboard -v
```

- [ ] **Step 4: Full manual smoke test**

Run: `cd dashboard && python manage.py runserver`. Visit all three page types (`/`, `/models/`, `/models/<name>/` for at least 2 different model names) in a browser. Confirm no template errors, no broken charts, numbers match `pipeline/outputs/metrics.json` when cross-checked by hand for at least one model.

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ -v` from repo root (covers both `tests/pipeline` and `tests/resultsboard`). Expected: all passing.

- [ ] **Step 6: Commit**

```bash
git add dashboard/dashboard/settings.py dashboard/README.md requirements.txt
git commit -m "chore(dashboard): finalize settings, pin Django version, add README"
```

---

## Self-Review Notes

- **Spec coverage**: all 3 pages from the design (overview, model comparison, model detail) covered; no-DB/no-auth/local-only constraints respected; `y_proba` gap identified in the design is fixed in Task 1 before the dashboard needs it; error handling for missing `pipeline/outputs/` covered (Task 4's `missing_outputs.html` + `_render_or_missing` helper used by every view).
- **Known open item flagged, not hidden**: Task 5's comparison chart doesn't draw a literal baseline reference line on the bar chart itself (would need an extra Chart.js plugin) — the design didn't mandate that specific visual, and the table + caption convey the same information, so this is a reasonable scope call, stated explicitly rather than silently simplified.
- **Type/name consistency**: `PIPELINE_OUTPUTS_DIR`, `data.load_*` function names, and template context keys (`summary`, `rows`, `baseline`, `cm`, `roc_fpr`/`roc_tpr`/`roc_auc`) are used identically across every task that touches them.
