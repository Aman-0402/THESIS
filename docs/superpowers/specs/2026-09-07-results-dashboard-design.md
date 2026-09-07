# Results Dashboard — Design

**Date:** 2026-09-07
**Status:** Approved

## Purpose

A local Django app to present the ML pipeline's results (dataset summary,
8-model comparison, per-model confusion matrix + ROC curve) for the thesis
defense. Read-only display of artifacts already produced by `pipeline/`.

## Audience & deployment

Single user (you), local machine only, run via `python manage.py runserver`
during the defense. No authentication, no remote hosting, no production
concerns (DEBUG mode is acceptable here).

## Architecture

- One Django project (`dashboard/`) with one app (`resultsboard/`).
- **No database, no models, no migrations.** Views read directly from
  `pipeline/outputs/` on each request:
  - `pipeline/outputs/dataset.csv` — dataset shape, feature list, positive rate
  - `pipeline/outputs/metrics.json` — per-model metrics + majority baseline
  - `pipeline/outputs/predictions/<model>.csv` — per-row true/pred/proba for
    confusion matrix and ROC curve construction
  - `pipeline/README.md` — "Known deviations" section, parsed or hand-copied
    into a static context dict (simplicity over parsing markdown at runtime)
- A small `resultsboard/data.py` module centralizes all file-reading/parsing
  logic (one clear responsibility: load pipeline outputs into plain Python
  dicts/lists), so views stay thin and templates only render.
- Charts rendered client-side via Chart.js (bar charts for metric comparison,
  line chart for ROC curve) — no server-side plotting library needed.

## Pipeline change required first

`pipeline/_05_split_train.py` currently writes only `y_true`/`y_pred` to
each `predictions/<model>.csv`. To draw real ROC curves (not just show the
scalar ROC-AUC already in `metrics.json`), add a `y_proba` column (the
positive-class probability, already computed in `main()` via
`predict_proba`/`decision_function` where available). This is a small,
additive change to an already-merged, tested file:
- For the 4 models without `predict_proba` (would be none currently — all 8
  specified estimators support `predict_proba` given `SVC(probability=True)`
  is used), `y_proba` is always populated.
- Re-run `_05_split_train.py` after the change and confirm accuracy/AUC
  numbers are unchanged (only an extra output column, no behavior change).

## Pages

### 1. Overview (`/`)
- Dataset shape (row count, feature count active vs. dropped)
- Train/val/test row counts and date ranges
- Target positive rate
- The 22 feature names, with the 4 always-null ones visibly marked
- "Known deviations" notes (Rovi/Zenotech gaps, dataset-size difference from
  the original thesis, etc.) — static text sourced from `pipeline/README.md`

### 2. Model comparison (`/models/`)
- Table: 8 rows (models) × columns (accuracy, precision, recall,
  specificity, F1, ROC-AUC, validation accuracy)
- Majority-class baseline shown as a highlighted reference row/line
- Bar chart comparing test accuracy across all 8 models + baseline line

### 3. Model detail (`/models/<name>/`)
- Confusion matrix rendered as a 2×2 table (labeled TN/FP/FN/TP)
- ROC curve (line chart, computed from `y_true`/`y_proba` in that model's
  predictions CSV using a small `sklearn.metrics.roc_curve` call in
  `data.py`, not precomputed/stored)
- The same metric row from the comparison table, repeated for context

## Error handling

If `pipeline/outputs/` doesn't exist yet (pipeline not run), views should
show a clear "run the pipeline first" message rather than crashing — this
is a local dev tool, not a robustness-critical app, but a blank crash page
during a defense would be embarrassing.

## Testing

Given this is a thin read/render layer over already-tested pipeline
outputs, tests focus on `resultsboard/data.py`'s parsing functions (pure
functions, easy to unit test with small crafted files) rather than
full view/template integration tests. A manual run-and-click-through
before the defense is the acceptance test for the UI itself.

## Out of scope

- Authentication/authorization
- Production deployment (WSGI server, static file collection, HTTPS)
- Database persistence of results across pipeline re-runs (always reads
  the latest files on disk)
- Editing/re-running the pipeline from the UI
