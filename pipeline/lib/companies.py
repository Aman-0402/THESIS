"""The frozen 30-company universe, matching documentation .pdf section 2,
and the per-company stock-series repair rules from section 7 (Phase 3B)."""

COMPANIES = [
    # Indian (region="IN")
    {"name": "Biocon", "region": "IN", "folder": "Biocon"},
    {"name": "Aurobindo Pharma", "region": "IN", "folder": "Aurobindo_Pharma"},
    {"name": "Glenmark Pharmaceuticals", "region": "IN", "folder": "Glenmark_Pharmaceuticals"},
    {"name": "Sun Pharmaceutical", "region": "IN", "folder": "Sun_Pharmaceutical"},
    {"name": "Torrent Pharmaceuticals", "region": "IN", "folder": "Torrent_Pharmaceuticals"},
    {"name": "Alembic Pharmaceuticals", "region": "IN", "folder": "Alembic_Pharmaceuticals"},
    {"name": "Natco Pharma", "region": "IN", "folder": "Natco_Pharma"},
    {"name": "Ajanta Pharma", "region": "IN", "folder": "Ajanta_Pharma"},
    {"name": "Marksans Pharma", "region": "IN", "folder": "Marksans_Pharma"},
    {"name": "Strides Pharma Science", "region": "IN", "folder": "Strides_Pharma_Science"},
    {"name": "Cipla", "region": "IN", "folder": "Cipla"},
    {"name": "Jubilant Pharmova", "region": "IN", "folder": "Jubilant_Pharmova"},
    {"name": "Lupin", "region": "IN", "folder": "Lupin"},
    {"name": "Zydus Lifesciences", "region": "IN", "folder": "Zydus_Lifesciences"},
    {"name": "Zenotech Laboratories", "region": "IN", "folder": "Zenotech_Laboratories"},
    # Non-Indian (region="NON_IN")
    {"name": "Amgen", "region": "NON_IN", "folder": "Amgen"},
    {"name": "Biogen", "region": "NON_IN", "folder": "Biogen"},
    {"name": "Teva Pharmaceutical", "region": "NON_IN", "folder": "Teva_Pharmaceutical"},
    {"name": "Novartis", "region": "NON_IN", "folder": "Novartis"},
    {"name": "Laboratorios Rovi", "region": "NON_IN", "folder": "Laboratorios_Rovi"},
    {"name": "Fresenius SE", "region": "NON_IN", "folder": "Fresenius_SE"},
    {"name": "Merck & Co.", "region": "NON_IN", "folder": "Merck_&_Co"},
    {"name": "Baxter International", "region": "NON_IN", "folder": "Baxter_International"},
    {"name": "Viatris", "region": "NON_IN", "folder": "Viatris"},
    {"name": "Jiangsu Hengrui Pharma", "region": "NON_IN", "folder": "Jiangsu_Hengrui_Pharma"},
    {"name": "Shanghai Fosun Pharma", "region": "NON_IN", "folder": "Shanghai_Fosun_Pharma"},
    {"name": "Zhejiang Hisun Pharma", "region": "NON_IN", "folder": "Zhejiang_Hisun_Pharma"},
    {"name": "Pfizer", "region": "NON_IN", "folder": "Pfizer"},
    {"name": "Sanofi", "region": "NON_IN", "folder": "Sanofi"},
    {"name": "Formycon", "region": "NON_IN", "folder": "Formycon"},
]

# documentation .pdf section 7 (Phase 3B canonical stock-series repair).
# Values mean: which block of the raw data is the canonical/analytical series.
STOCK_SERIES_OVERRIDE = {
    "Biocon": "adjusted_yahoo",
    "Ajanta_Pharma": "adjusted_yahoo",
    "Marksans_Pharma": "adjusted_yahoo",
    "Strides_Pharma_Science": "adjusted_yahoo",
    "Sanofi": "paris_eur",
}

# documentation .pdf section 10: financial-availability lag rule.
FINANCIAL_LAG_DAYS = {"quarterly": 60, "annual": 120}
ROVI_USES_ACTUAL_PUBLICATION_DATES = True
