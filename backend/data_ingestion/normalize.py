from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

CORPORATE_ACTION_THRESHOLD_PCT = 50.0
GAP_THRESHOLD_CALENDAR_DAYS = 10


@dataclass(frozen=True)
class NormalizedPoint:
    as_of: date
    value: float


@dataclass
class SeriesNormalization:
    points: list[NormalizedPoint]
    events: list[dict[str, Any]]
    rejected: list[dict[str, Any]]
    anomalies: list[dict[str, Any]]
    metadata: dict[str, Any]


def _parse_date_safe(value: str) -> date | None:
    """Parse a date string, returning None on failure."""
    try:
        return datetime.fromisoformat(value.strip()).date()
    except (ValueError, AttributeError):
        pass
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except (ValueError, AttributeError):
            continue
    return None


def _parse_float_safe(value: Any) -> float | None:
    """Parse a numeric value, returning None if missing or non-positive."""
    if value is None:
        return None
    try:
        v = float(str(value).strip())
        return v if v > 0 else None
    except (ValueError, TypeError):
        return None


def normalize_series(
    records: list[dict[str, Any]],
    date_key: str,
    value_key: str,
    label: str,
) -> SeriesNormalization:
    """Normalize a raw series into sorted, deduplicated points with event tracking.

    Steps:
        1. Parse dates and values; reject unparseable/non-positive.
        2. Sort ascending by date.
        3. Deduplicate (keep last observation per date).
        4. Detect extreme moves (corporate actions / bad ticks).
        5. Detect unusual gaps (>10 calendar days).
    """
    events: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []

    # Phase 1: Parse and filter
    parsed: list[tuple[date, float, int]] = []
    for idx, record in enumerate(records):
        raw_date = str(record.get(date_key, "")).strip()
        raw_value = record.get(value_key)

        parsed_date = _parse_date_safe(raw_date)
        if parsed_date is None:
            rejected.append({
                "row_index": idx,
                "reason": "unparseable_date",
                "raw_date": raw_date,
                "series": label,
            })
            continue

        parsed_value = _parse_float_safe(raw_value)
        if parsed_value is None:
            rejected.append({
                "row_index": idx,
                "reason": "missing_or_non_positive_value",
                "date": raw_date,
                "raw_value": str(raw_value) if raw_value is not None else "",
                "series": label,
            })
            continue

        parsed.append((parsed_date, parsed_value, idx))

    # Phase 2: Detect if sorting was needed
    dates_before_sort = [p[0] for p in parsed]
    if dates_before_sort != sorted(dates_before_sort):
        events.append({
            "series": label,
            "type": "sorted_ascending",
            "description": "Input records were not in chronological order.",
        })

    # Sort by date, then by original index (later rows win for same date)
    parsed.sort(key=lambda x: (x[0], x[2]))

    # Phase 3: Deduplicate (keep last per date)
    by_date: dict[date, tuple[float, int]] = {}
    duplicate_dates: set[date] = set()
    for d, v, idx in parsed:
        if d in by_date:
            duplicate_dates.add(d)
        by_date[d] = (v, idx)

    if duplicate_dates:
        events.append({
            "series": label,
            "type": "duplicate_dates_merged",
            "count": len(duplicate_dates),
            "dates": [d.isoformat() for d in sorted(duplicate_dates)],
            "description": (
                f"{len(duplicate_dates)} duplicate date(s) resolved by "
                "keeping last observation."
            ),
        })

    points = [
        NormalizedPoint(as_of=d, value=v)
        for d, (v, _) in sorted(by_date.items())
    ]

    # Phase 4: Detect anomalies (large jumps)
    for prev, curr in zip(points, points[1:]):
        if prev.value > 0:
            move_pct = ((curr.value / prev.value) - 1.0) * 100.0
            if abs(move_pct) >= CORPORATE_ACTION_THRESHOLD_PCT:
                anomalies.append({
                    "series": label,
                    "type": "extreme_move",
                    "date": curr.as_of.isoformat(),
                    "previous_date": prev.as_of.isoformat(),
                    "move_pct": round(move_pct, 2),
                    "threshold_pct": CORPORATE_ACTION_THRESHOLD_PCT,
                    "description": (
                        f"Value changed {move_pct:+.2f}% between "
                        f"{prev.as_of.isoformat()} and {curr.as_of.isoformat()}. "
                        "May indicate corporate action, split, or data error."
                    ),
                })

    # Phase 5: Detect gaps
    gap_events: list[dict[str, Any]] = []
    for prev, curr in zip(points, points[1:]):
        gap_days = (curr.as_of - prev.as_of).days
        if gap_days > GAP_THRESHOLD_CALENDAR_DAYS:
            gap_events.append({
                "series": label,
                "type": "gap_detected",
                "start_date": prev.as_of.isoformat(),
                "end_date": curr.as_of.isoformat(),
                "gap_days": gap_days,
                "description": (
                    f"No observations for {gap_days} calendar days between "
                    f"{prev.as_of.isoformat()} and {curr.as_of.isoformat()}."
                ),
            })
    events.extend(gap_events)

    metadata = {
        "input_records": len(records),
        "output_points": len(points),
        "rejected_count": len(rejected),
        "duplicate_dates_merged": len(duplicate_dates),
        "anomalies_flagged": len(anomalies),
        "gaps_detected": len(gap_events),
        "date_range": {
            "start": points[0].as_of.isoformat() if points else None,
            "end": points[-1].as_of.isoformat() if points else None,
        },
    }

    return SeriesNormalization(
        points=points,
        events=events,
        rejected=rejected,
        anomalies=anomalies,
        metadata=metadata,
    )
