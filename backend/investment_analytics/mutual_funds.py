from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

from .compiler import compile_insight
from .errors import PolicyError
from .lineage import LICENSE_REDIS, make_source


EXPECTED_TRADING_DAYS_PER_YEAR = 252
DEFAULT_ROLLING_WINDOW_POINTS = EXPECTED_TRADING_DAYS_PER_YEAR
DEFAULT_ROLLING_STEP_POINTS = 5
DEFAULT_MIN_ROLLING_WINDOWS = 126
DEFAULT_EXPECTED_WINDOW_SPAN_DAYS = 365
OUTLIER_MOVE_THRESHOLD_PCT = 50.0


@dataclass(frozen=True)
class SeriesPoint:
    as_of: date
    value: float


@dataclass(frozen=True)
class AlignedPoint:
    as_of: date
    fund_value: float
    benchmark_value: float


@dataclass(frozen=True)
class NormalizedSeries:
    points: list[SeriesPoint]
    events: list[dict[str, Any]]
    outliers: list[dict[str, Any]]


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _parse_date(value: str, label: str) -> date:
    try:
        parsed = datetime.fromisoformat(value.strip()).date()
    except ValueError as exc:
        raise PolicyError(
            "data_validation_error",
            f"Invalid {label} date.",
            {"date": value},
        ) from exc
    return parsed


def _business_days_between(start: date, end: date) -> int:
    if end < start:
        return 0
    total = 0
    cursor = start
    while cursor <= end:
        if cursor.weekday() < 5:
            total += 1
        cursor = date.fromordinal(cursor.toordinal() + 1)
    return total


def _years_between(start: date, end: date) -> float:
    return max((end - start).days / 365.25, 0.0)


def _period_return(start_value: float, end_value: float) -> float:
    if start_value <= 0:
        return 0.0
    return (end_value / start_value) - 1.0


def _annualized_return(start_value: float, end_value: float, years: float) -> float:
    period_return = _period_return(start_value, end_value)
    if years <= 0:
        return 0.0
    # Total loss (end_value == 0) gives period_return == -1.0. The
    # mathematically correct annualized return is -1.0 (-100% per year
    # in the abstract limit), not 0.0 — returning 0.0 mislabels a
    # wipeout as "no return" in downstream displays. NAV cannot be
    # negative in practice, so we still guard < -1.0 as malformed.
    if period_return <= -1.0:
        return -1.0 if (1.0 + period_return) == 0.0 else 0.0
    return ((1.0 + period_return) ** (1.0 / years)) - 1.0


def _target_date(latest: date, years: int) -> date:
    try:
        return latest.replace(year=latest.year - years)
    except ValueError:
        return latest.replace(month=2, day=28, year=latest.year - years)


def _point_on_or_before(points: list[AlignedPoint], target: date) -> AlignedPoint | None:
    eligible = [point for point in points if point.as_of <= target]
    if not eligible:
        return None
    return max(eligible, key=lambda point: point.as_of)


def _pct(value: float) -> float:
    return round(value * 100.0, 2)


def _source_key(source: dict[str, Any]) -> str:
    return json.dumps(source, sort_keys=True, separators=(",", ":"))


def _series_source(payload: dict[str, Any], key: str, fallback_name: str) -> dict[str, Any]:
    source = payload.get(key)
    if isinstance(source, dict):
        return dict(source)
    return make_source(fallback_name, _now_iso(), LICENSE_REDIS)


def _source_lineage(fund_source: dict[str, Any], benchmark_source: dict[str, Any]) -> dict[str, Any]:
    lineage: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in (fund_source, benchmark_source):
        key = _source_key(source)
        if key not in seen:
            seen.add(key)
            lineage.append(source)
    return make_source("derived:mutual_fund_metric", _now_iso(), LICENSE_REDIS, lineage)


def _normalize_series(items: list[dict[str, Any]], value_key: str, label: str) -> NormalizedSeries:
    if not isinstance(items, list) or not items:
        raise PolicyError(
            "data_validation_error",
            f"{label} series is required.",
            {"series": label},
        )

    parsed_rows: list[tuple[int, date, float]] = []
    input_dates: list[date] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise PolicyError(
                "data_validation_error",
                f"{label} row must be an object.",
                {"row_index": index},
            )
        row_date = _parse_date(str(item.get("date", "")), label)
        try:
            row_value = float(item.get(value_key))
        except (TypeError, ValueError) as exc:
            raise PolicyError(
                "data_validation_error",
                f"Invalid {label} value.",
                {"date": row_date.isoformat(), "value_key": value_key},
            ) from exc
        if row_value <= 0:
            raise PolicyError(
                "data_validation_error",
                f"Non-positive {label} value detected.",
                {"date": row_date.isoformat(), "value": row_value},
            )
        parsed_rows.append((index, row_date, row_value))
        input_dates.append(row_date)

    events: list[dict[str, Any]] = []
    if input_dates != sorted(input_dates):
        events.append({"series": label, "type": "sorted_ascending"})

    by_date: dict[date, tuple[int, float]] = {}
    duplicate_dates: set[date] = set()
    for index, row_date, row_value in parsed_rows:
        if row_date in by_date:
            duplicate_dates.add(row_date)
        by_date[row_date] = (index, row_value)
    if duplicate_dates:
        events.append(
            {
                "series": label,
                "type": "deduped_by_last_observation",
                "dates": [item.isoformat() for item in sorted(duplicate_dates)],
            }
        )

    points = [
        SeriesPoint(as_of=row_date, value=value)
        for row_date, (_, value) in sorted(by_date.items(), key=lambda item: item[0])
    ]

    outliers: list[dict[str, Any]] = []
    for previous, current in zip(points, points[1:]):
        move_pct = _period_return(previous.value, current.value) * 100.0
        if abs(move_pct) >= OUTLIER_MOVE_THRESHOLD_PCT:
            outliers.append(
                {
                    "series": label,
                    "date": current.as_of.isoformat(),
                    "previous_date": previous.as_of.isoformat(),
                    "move_pct": round(move_pct, 2),
                    "threshold_pct": OUTLIER_MOVE_THRESHOLD_PCT,
                }
            )

    return NormalizedSeries(points=points, events=events, outliers=outliers)


def _align_series(fund: list[SeriesPoint], benchmark: list[SeriesPoint]) -> list[AlignedPoint]:
    fund_by_date = {point.as_of: point.value for point in fund}
    benchmark_by_date = {point.as_of: point.value for point in benchmark}
    common_dates = sorted(set(fund_by_date).intersection(benchmark_by_date))
    return [
        AlignedPoint(as_of=row_date, fund_value=fund_by_date[row_date], benchmark_value=benchmark_by_date[row_date])
        for row_date in common_dates
    ]


def _data_quality(
    aligned: list[AlignedPoint],
    fund_count: int,
    benchmark_count: int,
    normalization_events: list[dict[str, Any]],
    outliers: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(aligned) < 2:
        expected_points = 0
        completeness_ratio = 0.0
        history_years = 0.0
    else:
        expected_points = _business_days_between(aligned[0].as_of, aligned[-1].as_of)
        calendar_density = len(aligned) / expected_points if expected_points else 0.0
        relative_completeness = len(aligned) / max(fund_count, benchmark_count, 1)
        history_years = _years_between(aligned[0].as_of, aligned[-1].as_of)

    if len(aligned) < 2:
        calendar_density = 0.0
        relative_completeness = 0.0

    if relative_completeness >= 0.95 and calendar_density >= 0.70:
        completeness = "High"
    elif relative_completeness >= 0.80 and calendar_density >= 0.40:
        completeness = "Medium"
    else:
        completeness = "Low"

    return {
        "aligned_points": len(aligned),
        "fund_points": fund_count,
        "benchmark_points": benchmark_count,
        "dropped_fund_only_dates": max(fund_count - len(aligned), 0),
        "dropped_benchmark_only_dates": max(benchmark_count - len(aligned), 0),
        "expected_trading_points": expected_points,
        "relative_completeness": round(relative_completeness, 4),
        "calendar_density": round(calendar_density, 4),
        "data_completeness": completeness,
        "history_years": round(history_years, 2),
        "normalization_events": normalization_events,
        "outlier_flags": outliers,
    }


def _evidence_strength(quality: dict[str, Any], rolling_windows: int = 0, min_windows: int = 0) -> str:
    history_years = float(quality["history_years"])
    aligned_points = int(quality["aligned_points"])
    if min_windows and rolling_windows < min_windows:
        return "Low"
    if history_years >= 5.0 and aligned_points >= 5 * EXPECTED_TRADING_DAYS_PER_YEAR:
        return "High"
    if history_years >= 3.0 and aligned_points >= 3 * EXPECTED_TRADING_DAYS_PER_YEAR:
        return "Medium"
    return "Low"


def _downgrade_evidence(current: str, levels: int = 1) -> str:
    order = ["Low", "Medium", "High"]
    index = order.index(current)
    return order[max(0, index - levels)]


def _window_span_summary(windows: list[dict[str, Any]]) -> dict[str, float]:
    spans = [int(item["span_days"]) for item in windows]
    if not spans:
        return {"min_days": 0, "median_days": 0, "max_days": 0}
    return {
        "min_days": min(spans),
        "median_days": round(float(median(spans)), 2),
        "max_days": max(spans),
    }


def _rolling_evidence(
    base: str,
    quality: dict[str, Any],
    windows: list[dict[str, Any]],
    min_windows: int,
    expected_window_span_days: int,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    evidence = base
    if len(windows) < min_windows:
        evidence = _downgrade_evidence(evidence)
        reasons.append(f"Rolling window count {len(windows)} is below threshold {min_windows}.")
    if float(quality["calendar_density"]) < 0.70:
        evidence = _downgrade_evidence(evidence)
        reasons.append(f"Calendar density {quality['calendar_density']:.2f} is below threshold 0.70.")
    span_summary = _window_span_summary(windows)
    min_span = expected_window_span_days * 0.80
    if span_summary["median_days"] < min_span:
        evidence = _downgrade_evidence(evidence)
        reasons.append(
            f"Median rolling window span {span_summary['median_days']:.0f} days is below threshold {min_span:.0f} days."
        )
    return evidence, reasons


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {"median_pct": 0.0, "p25_pct": 0.0, "p75_pct": 0.0}
    return {
        "median_pct": _pct(median(values)),
        "p25_pct": _pct(_percentile(values, 0.25)),
        "p75_pct": _pct(_percentile(values, 0.75)),
    }


def _rolling_windows(points: list[AlignedPoint], window_points: int, step_points: int) -> list[dict[str, Any]]:
    if window_points < 2 or step_points < 1 or len(points) < window_points:
        return []

    windows: list[dict[str, Any]] = []
    for start_index in range(0, len(points) - window_points + 1, step_points):
        start = points[start_index]
        end = points[start_index + window_points - 1]
        fund_return = _period_return(start.fund_value, end.fund_value)
        benchmark_return = _period_return(start.benchmark_value, end.benchmark_value)
        windows.append(
            {
                "start_date": start.as_of.isoformat(),
                "end_date": end.as_of.isoformat(),
                "span_days": (end.as_of - start.as_of).days,
                "fund_return": fund_return,
                "benchmark_return": benchmark_return,
                "excess_return": fund_return - benchmark_return,
            }
        )
    return windows


def _drawdown_profile(points: list[AlignedPoint], value_key: str) -> dict[str, Any]:
    peak_value = getattr(points[0], value_key)
    peak_date = points[0].as_of
    max_drawdown = 0.0
    start_date = points[0].as_of
    trough_date = points[0].as_of
    recovery_date: date | None = None
    recovery_threshold = peak_value

    for point in points:
        value = getattr(point, value_key)
        if value >= peak_value:
            peak_value = value
            peak_date = point.as_of
        drawdown = (value / peak_value) - 1.0
        if drawdown < max_drawdown:
            max_drawdown = drawdown
            start_date = peak_date
            trough_date = point.as_of
            recovery_threshold = peak_value
            recovery_date = None

    if max_drawdown < 0:
        for point in points:
            if point.as_of > trough_date and getattr(point, value_key) >= recovery_threshold:
                recovery_date = point.as_of
                break

    return {
        "max_drawdown_pct": _pct(max_drawdown),
        "start_date": start_date.isoformat(),
        "trough_date": trough_date.isoformat(),
        "recovery_date": recovery_date.isoformat() if recovery_date else None,
        "duration_days": (trough_date - start_date).days,
        "recovery_days": (recovery_date - trough_date).days if recovery_date else None,
    }


def _compile_or_record(insight: dict[str, Any], compiled: list[dict], suppressed: list[dict]) -> None:
    try:
        compiled.append(compile_insight(insight))
    except PolicyError as exc:
        if exc.code == "restricted_lineage":
            suppressed.append(
                {
                    "type": insight.get("type"),
                    "reason": exc.code,
                    "details": exc.details,
                }
            )
            return
        raise


def load_mutual_fund_csv(path: Path) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    if not rows:
        raise PolicyError("data_unavailable", "Mutual fund CSV has no rows.", {"path": str(path)})

    first = rows[0]
    fund_source = make_source(
        source=first.get("source", "sample_csv"),
        timestamp=first.get("source_timestamp", _now_iso()),
        license_value=first.get("license", LICENSE_REDIS),
    )
    benchmark_source = make_source(
        source=first.get("benchmark_source", first.get("source", "sample_csv")),
        timestamp=first.get("source_timestamp", _now_iso()),
        license_value=first.get("benchmark_license", first.get("license", LICENSE_REDIS)),
    )
    return {
        "fund_name": first.get("fund_name", "Sample Mutual Fund"),
        "benchmark_name": first.get("benchmark_name", "Sample Benchmark"),
        "category": first.get("category", "Unknown"),
        "expense_ratio_pct": float(first.get("expense_ratio_pct", 0.0) or 0.0),
        "fund_source": fund_source,
        "benchmark_source": benchmark_source,
        "fund": [{"date": row["date"], "nav": float(row["fund_nav"])} for row in rows],
        "benchmark": [{"date": row["date"], "value": float(row["benchmark_nav"])} for row in rows],
        "rolling_window_points": 4,
        "rolling_step_points": 1,
        "rolling_min_windows": 3,
        "expense_impact": {
            "investment_amount": 100000.0,
            "horizons_years": [3, 5, 10],
        },
    }


def _legacy_observations_to_contract(payload: dict[str, Any]) -> dict[str, Any]:
    observations = payload.get("observations")
    if not observations:
        return payload
    first = observations[0] if isinstance(observations, list) and observations else {}
    updated = dict(payload)
    updated.setdefault("fund_source", dict(first.get("fund_source", {})))
    updated.setdefault("benchmark_source", dict(first.get("benchmark_source", {})))
    updated["fund"] = [
        {"date": item["date"], "nav": item["fund_nav"]}
        for item in observations
        if isinstance(item, dict)
    ]
    updated["benchmark"] = [
        {"date": item["date"], "value": item["benchmark_nav"]}
        for item in observations
        if isinstance(item, dict)
    ]
    return updated


def analyze_mutual_fund(payload: dict[str, Any]) -> dict[str, Any]:
    payload = _legacy_observations_to_contract(payload)
    fund_name = str(payload.get("fund_name", "Mutual Fund"))
    benchmark_name = str(payload.get("benchmark_name", "Benchmark"))
    fund_source = _series_source(payload, "fund_source", "submitted_fund_series")
    benchmark_source = _series_source(payload, "benchmark_source", "submitted_benchmark_series")

    fund_series = _normalize_series(list(payload.get("fund", [])), "nav", "fund")
    benchmark_series = _normalize_series(list(payload.get("benchmark", [])), "value", "benchmark")
    aligned = _align_series(fund_series.points, benchmark_series.points)
    if len(aligned) < 2:
        raise PolicyError(
            "data_validation_error",
            "Fund and benchmark require at least two common dates.",
            {
                "fund_points": len(fund_series.points),
                "benchmark_points": len(benchmark_series.points),
                "intersection_points": len(aligned),
            },
        )

    normalization_events = fund_series.events + benchmark_series.events
    outliers = fund_series.outliers + benchmark_series.outliers
    quality = _data_quality(
        aligned,
        fund_count=len(fund_series.points),
        benchmark_count=len(benchmark_series.points),
        normalization_events=normalization_events,
        outliers=outliers,
    )
    source = _source_lineage(fund_source, benchmark_source)
    compiled: list[dict] = []
    suppressed: list[dict] = []
    unavailable: list[str] = []

    latest = aligned[-1]
    trailing_returns: dict[str, dict[str, float | str]] = {}
    for years in (1, 3, 5):
        start = _point_on_or_before(aligned, _target_date(latest.as_of, years))
        if start is None or start.as_of == latest.as_of:
            unavailable.append(f"{years}Y trailing return")
            continue
        actual_years = _years_between(start.as_of, latest.as_of)
        fund_return = _annualized_return(start.fund_value, latest.fund_value, actual_years)
        benchmark_return = _annualized_return(start.benchmark_value, latest.benchmark_value, actual_years)
        trailing_returns[f"{years}Y"] = {
            "start_date": start.as_of.isoformat(),
            "end_date": latest.as_of.isoformat(),
            "actual_years": round(actual_years, 2),
            "fund_cagr_pct": _pct(fund_return),
            "benchmark_cagr_pct": _pct(benchmark_return),
            "difference_pct_points": _pct(fund_return - benchmark_return),
        }

    base_limitations = ["Aligned to common dates only.", "Historical analytics do not describe future outcomes."]
    if quality["data_completeness"] != "High":
        base_limitations.append("Submitted series has fewer common observations than a daily trading calendar.")
    if outliers:
        outlier_dates = ", ".join(sorted({item["date"] for item in outliers}))
        base_limitations.append(
            f"Observed extreme value changes on {outlier_dates} may affect rolling and drawdown metrics."
        )

    if trailing_returns:
        longest_key = sorted(trailing_returns.keys(), key=lambda item: int(item[:-1]))[-1]
        longest = trailing_returns[longest_key]
        _compile_or_record(
            {
                "type": "benchmark_comparison",
                "observation": (
                    f"{longest_key} CAGR difference versus {benchmark_name} is "
                    f"{longest['difference_pct_points']:.2f} percentage points."
                ),
                "benchmark": {
                    "name": benchmark_name,
                    "methodology": "Simple compounded return annualized over matched common dates.",
                    "source": "submitted_aligned_series",
                },
                "supporting_data": {
                    "fund_name": fund_name,
                    "trailing_returns": trailing_returns,
                    "alignment": quality,
                    "source": source,
                },
                "evidence_strength": _evidence_strength(quality),
                "data_completeness": quality["data_completeness"],
                "limitations": base_limitations,
                "unavailable_components": unavailable,
            },
            compiled,
            suppressed,
        )

    window_points = int(payload.get("rolling_window_points", DEFAULT_ROLLING_WINDOW_POINTS) or DEFAULT_ROLLING_WINDOW_POINTS)
    step_points = int(payload.get("rolling_step_points", DEFAULT_ROLLING_STEP_POINTS) or DEFAULT_ROLLING_STEP_POINTS)
    min_windows = int(payload.get("rolling_min_windows", DEFAULT_MIN_ROLLING_WINDOWS) or DEFAULT_MIN_ROLLING_WINDOWS)
    windows = _rolling_windows(aligned, window_points, step_points)
    if windows:
        fund_returns = [item["fund_return"] for item in windows]
        benchmark_returns = [item["benchmark_return"] for item in windows]
        excess_returns = [item["excess_return"] for item in windows]
        wins = [value for value in excess_returns if value > 0]
        losses = [value for value in excess_returns if value < 0]
        hit_ratio = len(wins) / len(excess_returns) if excess_returns else 0.0
        span_summary = _window_span_summary(windows)
        expected_span = int(payload.get("expected_window_span_days", DEFAULT_EXPECTED_WINDOW_SPAN_DAYS) or DEFAULT_EXPECTED_WINDOW_SPAN_DAYS)
        base_evidence = _evidence_strength(quality)
        rolling_evidence, downgrade_reasons = _rolling_evidence(
            base_evidence,
            quality,
            windows,
            min_windows,
            expected_span,
        )
        excess_stats = {
            "sample_size": len(excess_returns),
            "window_count": len(excess_returns),
            "hit_ratio_pct": _pct(hit_ratio),
            "avg_win_pct": _pct(mean(wins)) if wins else 0.0,
            "avg_loss_pct": _pct(mean(losses)) if losses else 0.0,
            "spread_pct": _pct(pstdev(excess_returns)) if len(excess_returns) > 1 else 0.0,
            "fund_distribution": _distribution(fund_returns),
            "benchmark_distribution": _distribution(benchmark_returns),
            "excess_distribution": _distribution(excess_returns),
            "window_span_summary": span_summary,
            "window_points": window_points,
            "step_points": step_points,
            "expected_window_span_days": expected_span,
            "evidence_downgrade_reasons": downgrade_reasons,
        }
        _compile_or_record(
            {
                "type": "benchmark_comparison",
                "observation": (
                    f"Rolling excess return hit ratio versus {benchmark_name} is "
                    f"{excess_stats['hit_ratio_pct']:.2f}% across {len(excess_returns)} windows."
                ),
                "benchmark": {
                    "name": benchmark_name,
                    "methodology": "Each rolling window uses the same aligned start and end dates for fund and benchmark.",
                    "source": "submitted_aligned_series",
                },
                "supporting_data": {
                    "excess_return_stats": excess_stats,
                    "aligned_points": quality["aligned_points"],
                    "relative_completeness": quality["relative_completeness"],
                    "calendar_density": quality["calendar_density"],
                    "normalization_events": quality["normalization_events"],
                    "source": source,
                },
                "evidence_strength": rolling_evidence,
                "data_completeness": quality["data_completeness"],
                "limitations": base_limitations
                + downgrade_reasons
                + (["Rolling windows span expected duration but contain sparse observations, which may lower reliability of rolling metrics."]
                   if float(quality["calendar_density"]) < 0.70 and span_summary["median_days"] >= expected_span * 0.80
                   else [])
                + ["Rolling windows overlap and are not independent observations."],
                "unavailable_components": [],
            },
            compiled,
            suppressed,
        )
    else:
        unavailable.append("rolling excess return distribution")

    fund_drawdown = _drawdown_profile(aligned, "fund_value")
    benchmark_drawdown = _drawdown_profile(aligned, "benchmark_value")
    _compile_or_record(
        {
            "type": "benchmark_comparison",
            "observation": (
                f"Maximum drawdown over aligned history is {fund_drawdown['max_drawdown_pct']:.2f}% for "
                f"{fund_name} and {benchmark_drawdown['max_drawdown_pct']:.2f}% for {benchmark_name}."
            ),
            "benchmark": {
                "name": benchmark_name,
                "methodology": "Running peak-to-current decline measured identically on aligned observations.",
                "source": "submitted_aligned_series",
            },
            "supporting_data": {
                "fund_drawdown": fund_drawdown,
                "benchmark_drawdown": benchmark_drawdown,
                "source": source,
            },
            "evidence_strength": _evidence_strength(quality),
            "data_completeness": quality["data_completeness"],
            "limitations": base_limitations
            + ["Drawdown duration depends on observation frequency."]
            + (["Drawdown peak and trough identification may be affected by sparse observations."]
               if float(quality["calendar_density"]) < 0.70
               else []),
            "unavailable_components": [],
        },
        compiled,
        suppressed,
    )

    expense_ratio_pct = float(payload.get("expense_ratio_pct", 0.0) or 0.0)
    expense_impact = payload.get("expense_impact") or {}
    investment_amount = float(expense_impact.get("investment_amount", 100000.0) or 100000.0)
    horizons = expense_impact.get("horizons_years") or [3, 5, 10]
    ter = expense_ratio_pct / 100.0
    horizon_rows: list[dict[str, float]] = []
    for horizon in horizons:
        years = float(horizon)
        drag_factor = (1.0 - ter) ** years
        impact = investment_amount * (1.0 - drag_factor)
        horizon_rows.append(
            {
                "horizon_years": years,
                "drag_factor": round(drag_factor, 6),
                "estimated_cost_drag": round(impact, 2),
            }
        )
    impacts = [item["estimated_cost_drag"] for item in horizon_rows]
    expense_source = make_source("submitted_expense_ratio", _now_iso(), LICENSE_REDIS, [fund_source])
    _compile_or_record(
        {
            "type": "cost_tax",
            "scenario_a": {"name": "TER 0.00%", "terminal_value_proxy": round(investment_amount, 2)},
            "scenario_b": {
                "name": f"TER {expense_ratio_pct:.2f}%",
                "terminal_value_proxy_range": [
                    round(investment_amount - max(impacts), 2),
                    round(investment_amount - min(impacts), 2),
                ],
            },
            "assumptions": {
                "calculation_kind": "cost",
                "investment_amount": round(investment_amount, 2),
                "expense_ratio_pct": round(expense_ratio_pct, 2),
                "horizons": horizon_rows,
                "compounding": "annual constant TER drag factor",
                "tax_year": "not_applicable",
                "residency": "not_applicable",
                "rates": {},
                "holding_period": max(float(item) for item in horizons),
                "source": expense_source,
            },
            "estimated_impact": {
                "range": [round(min(impacts), 2), round(max(impacts), 2)],
                "units": "base_currency",
            },
            "evidence_strength": "Low",
            "data_completeness": "Medium",
            "limitations": [
                "Cost impact is a hypothetical calculation and does not reconstruct historical NAV.",
                "Consult a qualified professional",
            ],
            "unavailable_components": ["tax_lot_detail"],
        },
        compiled,
        suppressed,
    )

    summary = _build_summary(compiled, fund_name, benchmark_name)

    return {
        "insights": compiled,
        "suppressed_insights": suppressed,
        "data_quality": quality,
        "summary": summary,
    }


def _build_summary(
    insights: list[dict],
    fund_name: str,
    benchmark_name: str,
) -> dict[str, Any]:
    """Extract key metrics and plain-language interpretation from compiled insights.

    Returns a dict with:
      - metrics: dict of labelled key numbers
      - interpretation: list of factual plain-English sentences
    """
    metrics: dict[str, Any] = {}
    interpretation: list[str] = []

    for insight in insights:
        p = insight.get("payload", {})
        sd = p.get("supporting_data", {})

        # ── Trailing returns ──
        tr = sd.get("trailing_returns")
        if tr:
            for period_key in ("5Y", "3Y", "1Y"):
                if period_key in tr:
                    entry = tr[period_key]
                    metrics["trailing_return_period"] = period_key
                    metrics["fund_cagr_pct"] = entry["fund_cagr_pct"]
                    metrics["benchmark_cagr_pct"] = entry["benchmark_cagr_pct"]
                    metrics["cagr_difference_pct"] = entry["difference_pct_points"]

                    diff = entry["difference_pct_points"]
                    direction = "outperforming" if diff > 0 else "underperforming" if diff < 0 else "matching"
                    interpretation.append(
                        f"Over {period_key}, the fund returned {entry['fund_cagr_pct']:.2f}% CAGR "
                        f"vs {benchmark_name} at {entry['benchmark_cagr_pct']:.2f}% — "
                        f"{direction} by {abs(diff):.2f} percentage points."
                    )
                    break  # Use the longest available period

        # ── Rolling hit ratio ──
        excess = sd.get("excess_return_stats")
        if excess:
            hit = excess["hit_ratio_pct"]
            n = excess["window_count"]
            metrics["rolling_hit_ratio_pct"] = hit
            metrics["rolling_window_count"] = n
            interpretation.append(
                f"The fund beat its benchmark in {hit:.1f}% of {n} rolling return windows."
            )

        # ── Drawdown ──
        fd = sd.get("fund_drawdown")
        bd = sd.get("benchmark_drawdown")
        if fd:
            metrics["max_drawdown_pct"] = fd["max_drawdown_pct"]
            dd_sentence = f"Maximum drawdown was {abs(fd['max_drawdown_pct']):.1f}%"
            if fd.get("trough_date"):
                dd_sentence += f" (trough on {fd['trough_date']})"
            if bd:
                metrics["benchmark_max_drawdown_pct"] = bd["max_drawdown_pct"]
                dd_sentence += f", compared to {abs(bd['max_drawdown_pct']):.1f}% for {benchmark_name}"
            interpretation.append(dd_sentence + ".")

    # ── Self-benchmark detection ──
    is_self = "(self)" in benchmark_name.lower()
    if is_self:
        metrics["benchmark_type"] = "self"
        interpretation.insert(
            0,
            "Note: benchmark is self-referencing — relative metrics compare the fund to itself "
            "and will show zero difference.",
        )
    else:
        metrics["benchmark_type"] = "index"
        metrics["benchmark_name"] = benchmark_name

    return {"metrics": metrics, "interpretation": interpretation}
