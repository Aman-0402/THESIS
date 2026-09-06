# THESIS FINAL MASTER DATA PACKAGE

Created: 2026-09-02T19:29:12.621849

## Frozen 30-company universe

- Total: 30
- Indian: 15
- Non-Indian: 15

### Indian

- Biocon
- Aurobindo Pharma
- Glenmark Pharmaceuticals
- Sun Pharmaceutical
- Torrent Pharmaceuticals
- Alembic Pharmaceuticals
- Natco Pharma
- Ajanta Pharma
- Marksans Pharma
- Strides Pharma Science
- Cipla
- Jubilant Pharmova
- Lupin
- Zydus Lifesciences
- Zenotech Laboratories

### Non-Indian

- Amgen
- Biogen
- Teva Pharmaceutical
- Novartis
- Laboratorios Rovi
- Fresenius SE
- Merck & Co.
- Baxter International
- Viatris
- Jiangsu Hengrui Pharma
- Shanghai Fosun Pharma
- Zhejiang Hisun Pharma
- Pfizer
- Sanofi
- Formycon

## Scan statistics

- Desktop files scanned: 104212
- ZIP archives inspected: 99
- ZIP members inspected: 58626
- Relevant files captured: 7735
- Exact duplicate files: 24252
- Parse/read errors: 78
- Too-large files skipped: 0

## Company validation

- Detected frozen companies: 28/30
- Missing frozen companies: Shanghai Fosun Pharma, Zhejiang Hisun Pharma
- J.B. Chemicals: EXCLUDED BY RULE
- 3SBio: EXCLUDED BY RULE
- Zenotech Laboratories: FROZEN COMPANY
- Laboratorios Rovi: FROZEN COMPANY
- Company evidence directories: 30/30

## Methodology

Relevant raw/source evidence is preserved under 01_EVIDENCE. Compatible stock and financial tables are merged under 02_MERGED_DATA. Primary reports are preserved under 03_SOURCE_REPORTS. Audit, coverage, gap and hash information is retained in the remaining folders.

The script does not blindly concatenate heterogeneous datasets. It groups compatible tables by company, domain and inferred frequency, unions compatible columns, and removes exact duplicate rows.

Dates are treated conservatively. Implausible dates outside 1990-2035 are excluded from valid date coverage to prevent Excel/date artefacts from creating false historical coverage.

A gap means automatic reconciliation did not produce a suitable merged series; it does not by itself prove that no source exists. The file-level audit provides the evidence trail.
