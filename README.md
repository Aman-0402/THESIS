# Pharma 30-Company ML Classification — Thesis Project

Supervised ML classification pipeline for a 30-company pharmaceutical/
biopharma universe (15 Indian, 15 non-Indian), reproducing the analysis
described in `documentation .pdf`. Predicts next-quarter stock direction
(up/down) from market and fundamental features using 8 classifiers.

**Current result: no model exceeds the majority-class baseline.** See
`thesis_results/RESULTS.md` for the full write-up.

## Project layout

| Folder | What it is |
|---|---|
| `YUKTHA_FINAL_MASTER_DATA_2026-09-02_ORGANIZED/` | Frozen original raw-data archive (515MB) — provenance record, not read by the pipeline directly |
| `YUKTHA_CLEAN_2026-09-07/` | Cleaned/reorganized copy of the raw data — this is what `pipeline/` actually reads |
| `pipeline/` | The 5-stage ML pipeline (raw load → stock cleaning → panel → features/target → 8-classifier training) |
| `dashboard/` | Local Django app to view results in a browser |
| `sensitivity/` | One-factor-at-a-time robustness sweep over unconfirmed methodology assumptions |
| `thesis_results/` | Written results report (Markdown + figures), for pasting into the thesis document |
| `docs/superpowers/` | Design specs and implementation plans for each piece of this work |
| `tests/` | All tests (`tests/pipeline/`, `tests/resultsboard/`, `tests/sensitivity/`) |

## Setup (do this once)

```bash
pip install -r requirements.txt
```

## Run everything, in order

### 1. Run the ML pipeline

```bash
python pipeline/_01_load_raw.py
python pipeline/_02_clean_stock.py
python pipeline/_03_build_panel.py
python pipeline/_04_features_target.py
python pipeline/_05_split_train.py
```

Produces `pipeline/outputs/dataset.csv`, `metrics.json`, and `predictions/*.csv`. Takes a few minutes. See `pipeline/README.md` for details on what each stage does and the real-data limitations found while building it.

### 2. Generate the report figures

```bash
python thesis_results/generate_figures.py
```

Produces the PNGs in `thesis_results/figures/`, used by both `thesis_results/RESULTS.md` and the dashboard's Report page.

### 3. Run the sensitivity sweep (optional, takes longer)

```bash
python sensitivity/run_all.py
```

Runs 12 variants of the pipeline (financial lag, missing-data handling, up/down threshold, feature scope, and prediction horizon, one factor at a time) to check whether the "no model beats baseline" result is robust or fragile. See `sensitivity/README.md`.

### 4. View results in the dashboard

```bash
cd dashboard
python manage.py runserver
```

Visit **http://127.0.0.1:8000/**:

- `/` — dataset overview, methodology, feature formulas, Indian vs. non-Indian company breakdown
- `/models/` — 8-model comparison table + accuracy chart
- `/models/<name>/` — confusion matrix + ROC curve for one model
- `/report/` — full written report (same content as `thesis_results/RESULTS.md`), with embedded figures
- `/sensitivity/` — sensitivity-sweep results (only populated after step 3 above)

No database, no login — it's a local-only viewer that reads `pipeline/outputs/` and `sensitivity/outputs/` directly.

## Run the tests

```bash
python -m pytest tests/ -v
```

## Known open items

- The methodology's unconfirmed assumptions (financial-lag rule, target horizon, feature choices) are pipeline defaults from `documentation .pdf`, **not yet confirmed by the thesis supervisor** — see `thesis_results/RESULTS.md` §7 for the full list of open questions.
- The original thesis's exact 22 feature formulas were not available in this repo; `pipeline/lib/features.py` uses standard, documented finance formulas as a stated substitute.
- Full list of known data-coverage gaps (missing companies' data, always-null features) is in `pipeline/README.md` and `thesis_results/RESULTS.md` §7 (Limitations).
