"""Financial-information timing/lag rule. documentation .pdf section 10."""
import datetime as dt

from pipeline.lib.companies import FINANCIAL_LAG_DAYS

ROVI_FOLDER = "Laboratorios_Rovi"


def is_available(
    period_end: dt.date,
    observation_date: dt.date,
    period_type: str,
    company_folder: str,
    publication_date: dt.date | None = None,
    lag_days_override: dict[str, int] | None = None,
) -> bool:
    """True if a financial observation for `period_end` may be used when
    building features as of `observation_date`. `lag_days_override`, when
    given, replaces the module's FINANCIAL_LAG_DAYS for this call only --
    used by sensitivity/ to test alternative lag assumptions without
    duplicating this function."""
    if company_folder == ROVI_FOLDER:
        if publication_date is None:
            raise ValueError("Rovi requires an actual publication_date, not a lag rule")
        # >=: the publication date itself is the day the data becomes public,
        # so that day already counts as available.
        return observation_date >= publication_date

    lag_days_table = lag_days_override if lag_days_override is not None else FINANCIAL_LAG_DAYS
    lag_days = lag_days_table[period_type]
    available_from = period_end + dt.timedelta(days=lag_days)
    # strict >: data becomes available the day AFTER the lag period, not on
    # it (e.g. period_end + 60 days is still unavailable; +61 days is not).
    return observation_date > available_from
