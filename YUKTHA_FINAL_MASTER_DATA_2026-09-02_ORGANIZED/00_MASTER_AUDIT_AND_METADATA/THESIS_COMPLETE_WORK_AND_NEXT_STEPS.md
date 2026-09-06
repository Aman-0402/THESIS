# THESIS — COMPLETE WORK SUMMARY AND NEXT STEPS
Prepared: 06 September 2026

## 1. Purpose of this document

This document is the master plain-language description of the thesis work completed so far, what each stage did, why the stage was necessary, what was found, what was changed, what the current ML results are, what is still subject to supervisor confirmation, and what should happen next. It is intended to be understandable to a second reviewer, a supervisor, and the student who will continue the work.

This document deliberately distinguishes the **initial/first-data preliminary stage** from the **current final 30-company analysis pipeline**. The initial stage is not described using a person's name.

## 2. Thesis objective and required analysis

The study uses 30 pharmaceutical/biopharma companies, split into 15 Indian and 15 non-Indian companies, combining historical market data and financial-statement data for supervised machine-learning classification.

The eight required classifiers are:
1. Logistic Regression
2. Random Forest
3. Naive Bayes
4. Gradient Boosting
5. Support Vector Machine (SVM)
6. Neural Network / MLP
7. K-Nearest Neighbors (KNN)
8. Decision Tree

The required evaluation includes Accuracy, Precision, Recall/Sensitivity, Specificity, F1, confusion matrix, and ROC-AUC where appropriate.

## 3. Final company universe

### Indian
Biocon; Aurobindo Pharma; Glenmark Pharmaceuticals; Sun Pharmaceutical; Torrent Pharmaceuticals; Alembic Pharmaceuticals; Natco Pharma; Ajanta Pharma; Marksans Pharma; Strides Pharma Science; Cipla; Jubilant Pharmova; Lupin; Zydus Lifesciences; Zenotech Laboratories.

### Non-Indian
Amgen; Biogen; Teva Pharmaceutical; Novartis; Laboratorios Rovi; Fresenius SE; Merck & Co.; Baxter International; Viatris; Jiangsu Hengrui Pharma; Shanghai Fosun Pharma; Zhejiang Hisun Pharma; Pfizer; Sanofi; Formycon.

J.B. Chemicals and 3SBio are excluded from the final universe. J.B. Chemicals was replaced by Zenotech Laboratories and 3SBio by Laboratorios Rovi as part of the source-data/data-overlap decisions documented during reconciliation.

## 4. Initial / first-data stage

The project first used an earlier data package for a preliminary ML pipeline validation. This initial stage was deliberately a placeholder/sanity-check exercise rather than the final thesis experiment.

The preliminary setup used weekly stock observations and trailing market variables such as:
- ret_1w
- ret_4w
- ret_12w
- vol_12w
- price_to_ma12
- vol_chg_4w

The placeholder target was weekly direction:
- 1 if next week's close was higher than the current week's close
- 0 otherwise.

The preliminary pipeline demonstrated:
- chronological train/validation/test splitting
- train-only standardization
- validation-only hyperparameter selection
- held-out test evaluation
- majority-class baseline comparison

The preliminary KNN and Gradient Boosting results were close to baseline (roughly 51% test accuracy and AUC near 0.50). Those figures are not the final thesis results because the target, frequency, features, company set and source package were different.

## 5. Why the preliminary stage was not treated as final

The preliminary work exposed two methodological questions that could not responsibly be guessed:
- exactly what the thesis target should mean (UP/DOWN, threshold and horizon);
- exactly which features should be used and how financial information should be timed relative to publication.

The key principle is that a financial quarter-end date is not necessarily the date on which the information became public. Using a financial number before it was publicly known would create look-ahead bias.

The project therefore moved from a preliminary weekly price-only sanity check to a documented data-audit and methodology-construction process.

## 6. Source-data audit

The current master source package was audited before serious modeling.

The final audit processed the raw package and checked:
- archive integrity
- CSV parsing
- XLSX files
- PDF-derived text
- company coverage
- market-data coverage
- financial-data coverage
- duplicate and overlap patterns
- dates
- units and currencies
- annual/quarterly structures

The final audit was reported with zero remaining parsing errors after correcting initial date-format/timezone and empty-table handling problems.

The raw archive contains 7,953 members:
- 4,338 CSV files
- 182 XLSX files
- 62 PDFs
- 3,367 TXT files representing PDF page-text extractions
- 3 JSON files
- 1 Markdown file

## 7. Major data-reconciliation work

The audit found that some apparent gaps were parser/source-organization problems rather than true missing data.

Important examples:

### Zenotech Laboratories
Financial evidence was found despite the initial parser limitation. Annual and quarterly financial material exists, along with stock history. Zenotech was retained, with a documented liquidity/thin-trading caution.

### Laboratorios Rovi
Rovi has substantial daily stock history and annual financial information, but only limited quarterly fundamental coverage. Rovi was retained with that limitation explicitly documented.

### Formycon
Formycon has later-starting quarterly fundamentals and was retained with the limitation recorded.

### Shanghai Fosun Pharma / Zhejiang Hisun Pharma
Targeted investigation established that their source data was present, including stock and financial evidence. They were retained.

## 8. Currency, units and fiscal-period issues

The source data spans several currencies and units, including INR, USD, EUR and CNY, with both thousands and millions used in different source layers.

The analysis consequently emphasizes scale-normalized measures such as ratios, growth rates and returns rather than assuming incompatible raw monetary values are directly comparable.

Indian companies and international companies can also use different fiscal calendars. The current implementation documents the periods used rather than silently forcing all companies into one native fiscal convention.

## 9. Phase 3A — date collision analysis

When building the analytical stock series, many duplicate-looking dates appeared because timestamps differed while representing the same calendar trading day.

The diagnostic found:
- 33,592 collision records
- across 11 companies

Two types were distinguished.

### Mechanical duplicate/date-normalization cases
- Amgen
- Biogen
- Merck & Co.
- Novartis
- Pfizer
- Teva

Their same-day values were effectively identical apart from floating-point noise. These were handled using calendar-date normalization and deterministic deduplication.

### Structural stock-series conflicts
- Biocon
- Ajanta Pharma
- Marksans Pharma
- Strides Pharma Science
- Sanofi

These were not treated as harmless duplicates because they represented different price bases/source blocks.

## 10. Phase 3B — canonical stock-series repair

### Biocon
The merged stock series mixed adjusted and unadjusted blocks. The adjusted canonical block was retained and the unadjusted/redundant material was discarded from the analytical series.

### Ajanta Pharma
The adjusted Yahoo supplement was retained as the canonical stock series; the raw conflicting block was discarded.

### Marksans Pharma
The adjusted Yahoo supplement was retained.

### Strides Pharma Science
The adjusted Yahoo supplement was retained.

### Sanofi
Two different price bases were identified:
- US ADR (SNY, USD)
- Euronext Paris (EUR) data, internally mislabeled SNY.

The Paris/EUR home-market series was selected as the canonical series because it is consistent with the EUR financial reporting basis. The US ADR block was discarded from the analytical series.

## 11. Current analysis-ready dataset

The final analysis-ready dataset reported by the completed Phase 3B / Phase 4 workflow is:

- 1,410 rows
- 36 columns
- 30 companies
- 47 company-quarters per company
- period: 2014Q3–2026Q1

Target distribution:
- 789 positive
- 621 negative

Split:
- Train: 900 observations, 2014Q3–2021Q4
- Validation: 240 observations, 2022Q1–2023Q4
- Test: 270 observations, 2024Q1–2026Q1

There are 22 primary model features. The dataset also retains availability/provenance information for auditability, but those fields were excluded from the primary predictor set.

## 12. Current target implementation

The implemented target is binary:

For company-quarter t:
next-quarter return = (price at end of t+1 / price at end of t) - 1

Target:
- 1 if next-quarter return > 0
- 0 if next-quarter return <= 0

The implementation uses the cleaned canonical daily stock series and strictly uses the last trading day of each quarter for the comparison.

This is the **currently implemented** target. It must not be described as supervisor-approved unless the supervisor has explicitly confirmed it.

## 13. Financial timing and lag

The project deliberately separates:
- financial period end
- information availability

The current implementation uses a conservative availability rule:
- quarterly Capital-IQ-style data: period end + 60 days
- annual Capital-IQ-style data: period end + 120 days
- Rovi curated data: actual publication dates

The feature-building code only uses financial observations whose availability date is no later than the observation date. Additional staleness guards prevent indefinite carry-forward.

The documented Phase 3B leakage audit found zero availability-lag violations.

## 14. Missing financial information

Examples of documented missingness:
- operating margin has substantial missingness
- current ratio has meaningful missingness
- Rovi has limited quarterly fundamentals
- Formycon begins quarterly fundamentals later

The model pipeline handles numerical missing values using training-derived medians. This keeps imputation from using validation/test information.

Whether a feature with substantial missingness should remain in the final thesis is still a supervisor-review question.

## 15. Current feature design

The current implementation uses:
- 11 stock/market features
- 11 fundamental features
- total = 22 model predictors

The exact formulas and source-field mapping are stored in the project's feature dictionary and feature-building scripts. The definitive feature formulas should be taken from those actual artifacts, not from generic examples.

Conceptually, the market features cover historical price/return, momentum, moving-average, volatility and volume information, while fundamental features cover profitability, leverage, liquidity, growth and valuation-type information where available.

## 16. Leakage prevention

The pipeline was designed to prevent future information from entering predictors.

Controls include:
- past-only stock windows
- financial availability dates
- train-only imputation
- train-only winsorization
- train-only scaling
- chronological splitting
- validation-only hyperparameter selection
- one-time held-out test evaluation

The independent code audit found no future-information, target-formula, split-date, or preprocessing-fit leakage defect.

## 17. Phase 4 — eight-model analysis

All eight required classifiers were run against the same analysis-ready dataset.

The process was:
1. train models using the training period;
2. use validation data to select hyperparameters;
3. freeze the settings;
4. evaluate once on the untouched 270-observation test set.

Random seed: 42.

All eight prediction files have 270 test observations in identical company/quarter order.

## 18. Exact current model results

The independently checked final result CSV gives:

| Model | Validation Accuracy | Test Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.467 | 0.548 | 0.650 | 0.484 | 0.637 | 0.555 | 0.561 |
| Random Forest | 0.458 | 0.463 | 0.543 | 0.484 | 0.434 | 0.512 | 0.470 |
| Naive Bayes | 0.475 | 0.396 | 0.375 | 0.057 | 0.867 | 0.099 | 0.475 |
| Gradient Boosting | 0.458 | 0.511 | 0.571 | 0.637 | 0.336 | 0.602 | 0.460 |
| SVM | 0.463 | 0.519 | 0.605 | 0.497 | 0.549 | 0.545 | 0.515 |
| Neural Network | 0.596 | 0.574 | 0.598 | 0.815 | 0.239 | 0.690 | 0.690 |
| KNN | 0.567 | 0.522 | 0.584 | 0.618 | 0.389 | 0.601 | 0.524 |
| Decision Tree | 0.513 | 0.485 | 0.557 | 0.561 | 0.381 | 0.559 | 0.471 |
| Majority Baseline | 0.579 | 0.581 | 0.581 | 1.000 | 0.000 | 0.735 | 0.500 |

The Neural Network ROC-AUC is exactly 0.6900269541778976.

## 19. How to interpret the results

No model beats the majority-class baseline on held-out test accuracy.

The validation-selected model is the Neural Network because it has the highest validation accuracy (0.596).

Its test accuracy is 0.574, below the 0.581 majority baseline.

The Neural Network has the highest test F1 (0.690) and the highest test AUC (0.690), but its recall is high (0.815) while specificity is low (0.239). This shows that it strongly favors the positive class and therefore should not simply be described as a universally successful predictor.

Logistic Regression has the best test AUC among the non-neural models and shows only modest separation (0.561).

Naive Bayes performs poorly, with only 0.057 recall and F1 of 0.099.

The defensible overall conclusion is that the selected features provide, at most, weak out-of-sample predictive signal for next-quarter direction in the current sample. This does not prove that financial markets are perfectly efficient, and it does not prove that stock direction is impossible to predict.

## 20. Baseline

The majority-class baseline is essential because the test set is not exactly 50/50.

The baseline predicts the most common class. Its test accuracy is 0.581.

This means a model producing 0.57 accuracy is not successful merely because 57% sounds high. Its performance must be compared with the baseline and the other models under identical evaluation conditions.

## 21. Important implementation issues identified in independent review

Three YELLOW items were identified:

### Price-basis consistency
The general rule is `adj_close` when available, otherwise `close`. The independent reviewer recommended checking that the 30 cleaned stock series follow a sufficiently consistent adjusted-price basis.

### Class-weight asymmetry
Some classifiers use `class_weight='balanced'` while others do not because the available implementation exposes different interfaces. This should be documented and, ideally, discussed with the supervisor.

### Dependency pinning
The original project did not have a pinned requirements file. Exact reproduction across different machines can therefore vary slightly unless package versions are fixed.

None of these were found to be a data-integrity or leakage failure.

## 22. Professor approval questions still matter

Before declaring the current Phase 4 results the final thesis results, confirm with the supervisor:
- target definition
- binary versus multiclass
- threshold
- prediction horizon
- adjusted versus raw price
- raw versus market-adjusted return
- company-quarter versus another frequency
- exact feature list and formulas
- combined market + fundamentals versus separate analyses
- financial publication-date/lag rule
- YTD cash-flow derivation
- missing-data treatment
- pooled 30-company design versus separate groups
- country/company identity as predictor
- validation metric for hyperparameter selection
- required robustness analyses
- final thesis presentation requirements

## 23. What a second reviewer should check

The independent reviewer should treat the project like a thesis audit.

Check:
1. the frozen source-data package;
2. the 30-company universe;
3. the stock repairs;
4. Sanofi listing choice;
5. quarterly financial data;
6. target calculation;
7. every feature formula;
8. information availability;
9. train/validation/test splitting;
10. imputation/winsorization/scaling;
11. hyperparameter selection;
12. all eight models;
13. baseline;
14. confusion matrices;
15. ROC/AUC;
16. final interpretation;
17. reproducibility from a clean environment.

Classify findings:
- GREEN = okay
- YELLOW = needs clarification
- RED = must fix before finalization.

## 24. Exact reproducibility chain

The project should remain understandable as:

Frozen source data
→ source audit
→ reconciliation
→ stock-series canonicalization
→ calendar-date normalization/deduplication
→ company-quarter construction
→ target construction
→ feature construction
→ financial availability/lag filtering
→ chronological split
→ training-only preprocessing
→ validation hyperparameter selection
→ eight-model training
→ held-out test predictions
→ metrics/confusion matrices/ROC curves
→ thesis interpretation.

## 25. What should NOT be done

Do not:
- chase higher test accuracy;
- pick a model because its test metric looks attractive;
- mix initial/first-data preliminary results with final 30-company results;
- use the test set to tune hyperparameters;
- silently change the target after seeing model performance;
- overwrite the frozen source data;
- silently remove difficult companies;
- silently change financial lag rules;
- replace documented evidence with assumptions.

## 26. What should happen next

### Step 1 — independent brother review
Give the reviewer:
- the authoritative raw-data ZIP;
- the complete current VS Code project ZIP;
- this summary;
- the review checklist;
- the independent technical audit.

### Step 2 — supervisor confirmation
Take the outstanding methodology questions to the supervisor and record the answers.

### Step 3 — compare decisions
Compare the supervisor's answers against the current implementation.

If they match:
- freeze the methodology;
- freeze the Phase 4 model outputs;
- proceed to thesis writing.

If they change a core item such as target, horizon, frequency, feature set or timing:
- change the affected pipeline;
- rerun all affected models consistently;
- regenerate the final comparison.

### Step 4 — thesis outputs
Once the methodology is frozen, prepare:
- Results tables
- model comparison
- confusion matrices
- ROC figures
- feature table
- discussion
- limitations
- conclusion
- appendix / reproducibility material

## 27. Final status in one paragraph

The project has progressed from an initial weekly price-based ML sanity check to a fully audited 30-company, company-quarter classification pipeline. The current technical pipeline includes a frozen raw-data package, extensive source-data reconciliation, canonical stock-series repairs, a 1,410-observation analysis-ready dataset with 22 primary features, explicit financial-information timing controls, a chronological 900/240/270 train-validation-test split, train-only preprocessing, validation-only tuning, and all eight required classification models with held-out predictions, confusion matrices, ROC curves and detailed metrics. The current empirical result is that no model exceeds the majority-class baseline in test accuracy; therefore the current evidence indicates limited predictive signal rather than successful high-accuracy forecasting. The remaining critical step is to have the supervisor confirm the methodological choices before the Phase 4 results are declared the final thesis results.

## 28. Source package organization

A reorganized copy of the raw source data is intended to use:

01_INDIAN_COMPANIES/
  Company/
    01_FINANCIAL_DATA/
      Annual/
      Quarterly/
      Interim/
      Other/
    02_HISTORICAL_MARKET_DATA/
      Daily/
      Weekly/
      Monthly/
      Other/
    03_SOURCE_REPORTS/
      Reports/
    04_OTHER_SOURCE_MATERIAL/

02_NON_INDIAN_COMPANIES/
  Company/
    same structure

00_MASTER_AUDIT_AND_METADATA/
  original audit, coverage, gaps, hash and README files

Each company folder also receives a SOURCE_INDEX.csv linking every reorganized file back to its original archive member path, so organization does not destroy provenance.

The source files themselves are copied byte-for-byte from the raw ZIP into their new locations; this organization does not change the source content.
