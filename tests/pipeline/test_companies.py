from pipeline.lib.companies import COMPANIES, STOCK_SERIES_OVERRIDE


def test_thirty_companies_split_15_15():
    indian = [c for c in COMPANIES if c["region"] == "IN"]
    non_indian = [c for c in COMPANIES if c["region"] == "NON_IN"]
    assert len(indian) == 15
    assert len(non_indian) == 15
    assert len(COMPANIES) == 30


def test_folder_names_match_clean_data_dirs():
    names = {c["folder"] for c in COMPANIES}
    assert "Ajanta_Pharma" in names
    assert "Laboratorios_Rovi" in names
    assert "Merck_&_Co" in names


def test_sanofi_uses_paris_override():
    assert STOCK_SERIES_OVERRIDE["Sanofi"] == "paris_eur"


def test_adjusted_yahoo_override_companies():
    for name in ("Biocon", "Ajanta_Pharma", "Marksans_Pharma", "Strides_Pharma_Science"):
        assert STOCK_SERIES_OVERRIDE[name] == "adjusted_yahoo"
