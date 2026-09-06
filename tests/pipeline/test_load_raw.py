import pandas as pd
from pipeline._01_load_raw import load_company_daily_csvs

def test_load_company_daily_csvs_tags_source_file(tmp_path):
    daily_dir = tmp_path / "02_MARKET_DATA" / "Daily"
    daily_dir.mkdir(parents=True)
    (daily_dir / "a.csv").write_text("Date,Close\n2024-01-01,10.0\n")
    (daily_dir / "b.csv").write_text("Date,Close\n2024-01-02,11.0\n")

    frames = load_company_daily_csvs(tmp_path)
    assert len(frames) == 2
    sources = {f["source_file"].iloc[0] for f in frames}
    assert sources == {"a.csv", "b.csv"}
