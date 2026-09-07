# Results Dashboard

Local-only Django app displaying the ML pipeline's results. No database,
no auth, no deployment — reads `pipeline/outputs/` and `sensitivity/outputs/`
directly on each request.

## Run

Make sure the pipeline has been run first (see `../pipeline/README.md`),
then:

    cd dashboard
    python manage.py runserver

Visit http://127.0.0.1:8000/

## Pages

- `/` — dataset overview, methodology, feature formulas, Indian vs. non-Indian company breakdown
- `/models/` — 8-model comparison table + accuracy bar chart
- `/models/<name>/` — confusion matrix + ROC curve for one model
- `/report/` — full written results report, with embedded figures
- `/sensitivity/` — one-factor-at-a-time robustness sweep across unconfirmed methodology assumptions (only populated after running `python sensitivity/run_all.py`, see `../sensitivity/README.md`)

## Tests

    python -m pytest tests/ -v

(run from the repo root — `pytest.ini` wires `dashboard/` onto the
Python path so `resultsboard` is importable)
