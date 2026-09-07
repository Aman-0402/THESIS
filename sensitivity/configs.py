"""One-factor-at-a-time run configs for the sensitivity sweep. See
docs/superpowers/specs/2026-09-07-sensitivity-analysis-design.md."""

BASE = {
    "run_id": "base",
    "description": "Base case: current pipeline defaults",
    "dimension": "base",
    "value_label": "current pipeline defaults",
    "quarterly_lag_days": 60,
    "annual_lag_days": 120,
    "missing_strategy": "impute",
    "threshold_mode": "simple",
    "deadzone_pct": 0.0,
    "feature_scope": "all",
    "horizon_freq": "QE",
}


def _variant(run_id: str, description: str, dimension: str, value_label: str, **overrides) -> dict:
    config = dict(BASE)
    config.update(run_id=run_id, description=description, dimension=dimension, value_label=value_label)
    config.update(overrides)
    return config


RUN_CONFIGS = [
    BASE,
    _variant("lag_30", "Financial lag: 30/60 days", "lag", "30d quarterly / 60d annual",
             quarterly_lag_days=30, annual_lag_days=60),
    _variant("lag_45", "Financial lag: 45/90 days", "lag", "45d quarterly / 90d annual",
             quarterly_lag_days=45, annual_lag_days=90),
    _variant("lag_90", "Financial lag: 90/180 days", "lag", "90d quarterly / 180d annual",
             quarterly_lag_days=90, annual_lag_days=180),
    _variant("lag_120", "Financial lag: 120/240 days", "lag", "120d quarterly / 240d annual",
             quarterly_lag_days=120, annual_lag_days=240),
    _variant("missing_drop_any", "Missing data: drop rows with any missing feature",
             "missing_strategy", "drop rows with any missing feature",
             missing_strategy="drop_any"),
    _variant("missing_drop_majority", "Missing data: drop rows >50% missing, impute rest",
             "missing_strategy", "drop rows >50% missing, impute rest",
             missing_strategy="drop_majority"),
    _variant("threshold_deadzone", "Up/down rule: +/-1% dead zone",
             "threshold", "dead zone: |return| < 1% excluded",
             threshold_mode="deadzone", deadzone_pct=0.01),
    _variant("features_market_only", "Feature scope: market features only",
             "feature_scope", "market-only",
             feature_scope="market_only"),
    _variant("features_fundamental_only", "Feature scope: fundamental features only",
             "feature_scope", "fundamental-only",
             feature_scope="fundamental_only"),
    _variant("horizon_monthly", "Horizon: monthly panel", "horizon", "monthly",
             horizon_freq="ME"),
    _variant("horizon_weekly", "Horizon: weekly panel", "horizon", "weekly",
             horizon_freq="W-FRI"),
]
