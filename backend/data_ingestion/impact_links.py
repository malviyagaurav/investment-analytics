"""Link ingestion issues to the analyzer metrics they affect.

Pure function — takes ingestion metadata, returns deterministic impact
annotations.  No heuristics, no scoring, no intelligence.  Just states
which data issues potentially affect which metric families.

Design principles:
    - Deterministic: same input → same output.
    - Informational: describes *what* is affected, not *how badly*.
    - Non-advisory: no judgments like "unreliable" or "degraded".
"""
from __future__ import annotations

from typing import Any


# Metric family labels — stable across analyzer versions.
TRAILING_RETURNS = "trailing returns"
ROLLING_WINDOWS = "rolling excess returns"
DRAWDOWN = "drawdown profile"
EVIDENCE = "evidence strength"

# Cost/tax is independent of time-series quality.
_TIMESERIES_METRICS = [TRAILING_RETURNS, ROLLING_WINDOWS, DRAWDOWN]


def build_impact_links(
    fund_meta: dict[str, Any],
    bench_meta: dict[str, Any],
    fund_anomalies: list[dict[str, Any]],
    bench_anomalies: list[dict[str, Any]],
    fund_events: list[dict[str, Any]],
    bench_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build deterministic links between ingestion issues and affected metrics.

    Returns a list of impact link dicts::

        {
            "issue": str,          # what was detected
            "series": str,         # "fund", "benchmark", or "both"
            "affected_metrics": [str],  # metric families affected
            "explanation": str,    # why this issue matters (non-advisory)
        }
    """
    links: list[dict[str, Any]] = []

    # Ordered broadest-impact-first for consistent reading:
    # short_history → records_rejected → duplicates → gaps → extreme_values

    # ── Short history ──────────────────────────────────────────
    fund_points = fund_meta.get("output_points", 0)
    bench_points = bench_meta.get("output_points", 0)
    min_points = min(fund_points, bench_points)

    if min_points < 20:
        links.append({
            "issue": "short_history",
            "series": "both",
            "affected_metrics": _TIMESERIES_METRICS + [EVIDENCE],
            "explanation": (
                f"Usable series length is {min_points} observation(s). "
                "Longer trailing return periods may be unavailable. "
                "Rolling window count and evidence strength are "
                "limited by the observation count."
            ),
        })

    # ── Rejected records ───────────────────────────────────────
    fund_rejected = fund_meta.get("rejected_count", 0)
    bench_rejected = bench_meta.get("rejected_count", 0)

    if fund_rejected or bench_rejected:
        series = _which_series(fund_rejected, bench_rejected)
        total = fund_rejected + bench_rejected
        links.append({
            "issue": "records_rejected",
            "series": series,
            "affected_metrics": [EVIDENCE],
            "explanation": (
                f"{total} record(s) excluded from {series} series. "
                "Fewer usable observations reduce the aligned dataset "
                "and may lower evidence strength."
            ),
        })

    # ── Duplicates merged ──────────────────────────────────────
    fund_dupes = fund_meta.get("duplicate_dates_merged", 0)
    bench_dupes = bench_meta.get("duplicate_dates_merged", 0)

    if fund_dupes or bench_dupes:
        series = _which_series(fund_dupes, bench_dupes)
        links.append({
            "issue": "duplicates_merged",
            "series": series,
            "affected_metrics": [EVIDENCE],
            "explanation": (
                f"Duplicate date(s) in {series} series resolved by "
                "keeping the last observation per date. Original values "
                "on those dates were discarded."
            ),
        })

    # ── Gaps ───────────────────────────────────────────────────
    fund_gaps = fund_meta.get("gaps_detected", 0)
    bench_gaps = bench_meta.get("gaps_detected", 0)

    if fund_gaps or bench_gaps:
        series = _which_series(fund_gaps, bench_gaps)
        gap_dates = _gap_date_ranges(fund_events, bench_events)
        links.append({
            "issue": "gaps_detected",
            "series": series,
            "affected_metrics": [ROLLING_WINDOWS, DRAWDOWN, TRAILING_RETURNS],
            "explanation": (
                f"Observation gap(s) detected in {series} series"
                f"{_date_suffix(gap_dates)}. "
                "Rolling windows that span a gap use fewer observations. "
                "Drawdown recovery timing may be approximate."
            ),
        })

    # ── Outliers / extreme moves ───────────────────────────────
    fund_anomaly_count = fund_meta.get("anomalies_flagged", 0)
    bench_anomaly_count = bench_meta.get("anomalies_flagged", 0)

    if fund_anomaly_count or bench_anomaly_count:
        series = _which_series(fund_anomaly_count, bench_anomaly_count)
        anomaly_dates = _anomaly_date_list(fund_anomalies, bench_anomalies)
        links.append({
            "issue": "extreme_values",
            "series": series,
            "affected_metrics": [ROLLING_WINDOWS, DRAWDOWN, TRAILING_RETURNS],
            "explanation": (
                f"Extreme value change(s) in {series} series"
                f"{_date_suffix(anomaly_dates)}. "
                "Rolling windows containing these observations reflect "
                "the extreme movement. Drawdown peak or trough may "
                "coincide with the flagged date(s)."
            ),
        })

    return links


# ── Helpers ────────────────────────────────────────────────────

def _which_series(fund_count: int, bench_count: int) -> str:
    if fund_count and bench_count:
        return "both"
    return "fund" if fund_count else "benchmark"


def _gap_date_ranges(
    fund_events: list[dict[str, Any]],
    bench_events: list[dict[str, Any]],
) -> list[str]:
    ranges: list[str] = []
    for ev in fund_events + bench_events:
        if ev.get("type") == "gap_detected":
            ranges.append(f"{ev['start_date']} to {ev['end_date']}")
    return sorted(set(ranges))


def _anomaly_date_list(
    fund_anomalies: list[dict[str, Any]],
    bench_anomalies: list[dict[str, Any]],
) -> list[str]:
    dates: list[str] = []
    for a in fund_anomalies + bench_anomalies:
        if "date" in a:
            dates.append(a["date"])
    return sorted(set(dates))


def _date_suffix(dates: list[str]) -> str:
    if not dates:
        return ""
    if len(dates) <= 3:
        return " (" + ", ".join(dates) + ")"
    return f" ({len(dates)} date(s))"
