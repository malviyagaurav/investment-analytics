from __future__ import annotations

from datetime import date, datetime


DEFAULT_THRESHOLDS = {
    "nav_staleness_trading_days": 2,
    "holdings_staleness_days": 90,
    "minimum_history_years": 3.0,
    "missing_fields_suppress_ratio": 0.30,
    "etf_avg_daily_value_min": 10_000_000,
}


def trading_days_between(start: date, end: date) -> int:
    if start > end:
        return 0
    days = 0
    cursor = start
    while cursor < end:
        cursor = date.fromordinal(cursor.toordinal() + 1)
        if cursor.weekday() < 5:
            days += 1
    return days


def parse_date(value: str) -> date:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).date()


def nav_staleness_flag(timestamp: str, as_of: date | None = None) -> bool:
    as_of = as_of or datetime.utcnow().date()
    return trading_days_between(parse_date(timestamp), as_of) > DEFAULT_THRESHOLDS["nav_staleness_trading_days"]


def holdings_staleness_flag(timestamp: str, as_of: date | None = None) -> bool:
    as_of = as_of or datetime.utcnow().date()
    return (as_of - parse_date(timestamp)).days > DEFAULT_THRESHOLDS["holdings_staleness_days"]


def evidence_from_history(history_years: float) -> str:
    return "Low" if history_years < DEFAULT_THRESHOLDS["minimum_history_years"] else "Medium"


def should_suppress_for_missing_fields(missing_ratio: float) -> bool:
    return missing_ratio > DEFAULT_THRESHOLDS["missing_fields_suppress_ratio"]


def etf_liquidity_flag(avg_daily_value: float, threshold: float | None = None) -> str:
    limit = threshold if threshold is not None else DEFAULT_THRESHOLDS["etf_avg_daily_value_min"]
    return "Low liquidity" if avg_daily_value < limit else "Liquidity threshold met"

