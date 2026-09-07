# Sensitivity Analysis

One-factor-at-a-time robustness sweep over the pipeline's unconfirmed
methodology assumptions (financial lag, missing-data handling, up/down
threshold, feature scope, prediction horizon). See
`docs/superpowers/specs/2026-09-07-sensitivity-analysis-design.md` for the
full design rationale.

## Run

Requires `pipeline/outputs/` to already exist (run the main pipeline first,
see `../pipeline/README.md`):

    python sensitivity/run_all.py

This runs all 12 configs (1 base + 11 variants). The weekly-horizon variant
builds a ~45,000-row dataset and dominates the runtime — expect the full
sweep to take roughly 15-25 minutes, not seconds. That's expected.

Outputs land in `sensitivity/outputs/` (gitignored): one `<run_id>/` folder
per config with `dataset.csv`/`metrics.json`, plus an aggregated
`summary.json` across all 12 runs.

## View results

    cd dashboard
    python manage.py runserver

Visit http://127.0.0.1:8000/sensitivity/

## Result (last run)

The `base` config reproduces `pipeline/outputs/metrics.json` bit-for-bit
(verified across all 8 models). Across the 11 variants, 5 show a model
technically beating its own baseline (lag 45/120 days, fundamental-only
features, monthly and weekly horizon), but every delta is 0.1-1.5
percentage points — consistent with noise, not a specific assumption
recovering real predictive signal. The other 6 stay below baseline like the
base case.

## Tests

    python -m pytest tests/sensitivity -v
