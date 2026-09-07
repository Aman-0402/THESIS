from sensitivity.configs import RUN_CONFIGS

REQUIRED_KEYS = {
    "run_id", "description", "dimension", "value_label",
    "quarterly_lag_days", "annual_lag_days", "missing_strategy",
    "threshold_mode", "deadzone_pct", "feature_scope", "horizon_freq",
}


def test_twelve_configs_total():
    assert len(RUN_CONFIGS) == 12


def test_exactly_one_base_config():
    base = [c for c in RUN_CONFIGS if c["dimension"] == "base"]
    assert len(base) == 1
    base_config = base[0]
    assert base_config["quarterly_lag_days"] == 60
    assert base_config["annual_lag_days"] == 120
    assert base_config["missing_strategy"] == "impute"
    assert base_config["threshold_mode"] == "simple"
    assert base_config["feature_scope"] == "all"
    assert base_config["horizon_freq"] == "QE"


def test_all_configs_have_required_keys_and_unique_ids():
    ids = set()
    for config in RUN_CONFIGS:
        assert REQUIRED_KEYS.issubset(config.keys())
        assert config["run_id"] not in ids
        ids.add(config["run_id"])


def test_dimension_counts_match_design():
    from collections import Counter
    counts = Counter(c["dimension"] for c in RUN_CONFIGS)
    assert counts["base"] == 1
    assert counts["lag"] == 4
    assert counts["missing_strategy"] == 2
    assert counts["threshold"] == 1
    assert counts["feature_scope"] == 2
    assert counts["horizon"] == 2
