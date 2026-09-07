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

    python -m pytest tests/ -v

(run from the repo root — `pytest.ini` wires `dashboard/` onto the
Python path so `resultsboard` is importable)
