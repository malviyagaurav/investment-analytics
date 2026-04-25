from __future__ import annotations

from typing import Any

from backend.investment_analytics.errors import PolicyError
from .normalize import SeriesNormalization


def validate_series(
    result: SeriesNormalization,
    label: str,
    min_points: int = 2,
) -> None:
    """Raise PolicyError if normalized series is unusable for analysis."""
    if len(result.points) < min_points:
        raise PolicyError(
            "ingestion_validation_error",
            f"{label} series has {len(result.points)} usable point(s) after "
            f"normalization, minimum is {min_points}.",
            {
                "series": label,
                "usable_points": len(result.points),
                "min_required": min_points,
                "rejected_count": result.metadata["rejected_count"],
            },
        )


def build_ingestion_limitations(
    fund_result: SeriesNormalization,
    benchmark_result: SeriesNormalization | None = None,
) -> list[str]:
    """Build human-readable limitations from ingestion events."""
    limitations: list[str] = []

    for result, label in [(fund_result, "Fund"), (benchmark_result, "Benchmark")]:
        if result is None:
            continue
        if result.metadata["rejected_count"] > 0:
            limitations.append(
                f"{result.metadata['rejected_count']} {label.lower()} record(s) "
                "were excluded due to missing or non-positive values."
            )
        if result.metadata["duplicate_dates_merged"] > 0:
            limitations.append(
                f"{result.metadata['duplicate_dates_merged']} duplicate "
                f"{label.lower()} date(s) were resolved by keeping the last "
                "observation."
            )
        if result.metadata["anomalies_flagged"] > 0:
            limitations.append(
                f"{result.metadata['anomalies_flagged']} extreme value "
                f"change(s) detected in {label.lower()} series. "
                "Large value changes were flagged using a fixed threshold "
                "and may include valid market events."
            )
        if result.metadata["gaps_detected"] > 0:
            limitations.append(
                f"{result.metadata['gaps_detected']} gap(s) of more than "
                f"10 calendar days detected in {label.lower()} series. "
                "Gaps are measured in calendar days and may reflect "
                "market closures or data frequency."
            )

    return limitations
