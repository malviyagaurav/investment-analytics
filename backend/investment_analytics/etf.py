from __future__ import annotations

from datetime import datetime, timezone
from statistics import mean, median, pstdev
from typing import Any

from .compiler import compile_insight
from .errors import PolicyError
from .lineage import LICENSE_REDIS, make_source
from .mutual_funds import (
    EXPECTED_TRADING_DAYS_PER_YEAR,
    DEFAULT_ROLLING_WINDOW_POINTS,
    DEFAULT_ROLLING_STEP_POINTS,
    DEFAULT_MIN_ROLLING_WINDOWS,
    DEFAULT_EXPECTED_WINDOW_SPAN_DAYS,
    _align_series,
    _annualized_return,
    _compile_or_record,
    _data_quality,
    _distribution,
    _downgrade_evidence,
    _drawdown_profile,
    _evidence_strength,
    _normalize_series,
    _now_iso,
    _pct,
    _period_return,
    _point_on_or_before,
    _rolling_evidence,
    _rolling_windows,
    _series_source,
    _source_lineage,
    _target_date,
    _window_span_summary,
)

ETF_ANALYZER_VERSION = "etf_v1"


def _tracking_difference(aligned: list) -> dict[str, Any]:
    """Cumulative ETF return minus cumulative benchmark return over the full span."""
    if len(aligned) < 2:
        return {"tracking_difference_pct": 0.0, "span_years": 0.0}
    start = aligned[0]
    end = aligned[-1]
    years = max((end.as_of - start.as_of).days / 365.25, 0.0)
    etf_total = _period_return(start.fund_value, end.fund_value)
    bench_total = _period_return(start.benchmark_value, end.benchmark_value)
    diff = etf_total - bench_total
    return {
        "tracking_difference_pct": _pct(diff),
        "etf_total_return_pct": _pct(etf_total),
        "benchmark_total_return_pct": _pct(bench_total),
        "start_date": start.as_of.isoformat(),
        "end_date": end.as_of.isoformat(),
        "span_years": round(years, 2),
    }


def _tracking_error(aligned: list) -> dict[str, Any]:
    """Standard deviation of periodic excess returns (daily or per-observation)."""
    if len(aligned) < 3:
        return {"tracking_error_pct": 0.0, "observation_count": 0}
    excess_returns: list[float] = []
    for prev, curr in zip(aligned, aligned[1:]):
        etf_r = _period_return(prev.fund_value, curr.fund_value)
        bench_r = _period_return(prev.benchmark_value, curr.benchmark_value)
        excess_returns.append(etf_r - bench_r)
    te = pstdev(excess_returns) if len(excess_returns) > 1 else 0.0
    return {
        "tracking_error_pct": _pct(te),
        "observation_count": len(excess_returns),
        "mean_excess_return_pct": _pct(mean(excess_returns)),
        "excess_return_distribution": _distribution(excess_returns),
    }


def analyze_etf(payload: dict[str, Any]) -> dict[str, Any]:
    """Analyze an ETF against its benchmark.

    Input contract:
        etf_name: str
        benchmark_name: str
        category: str
        expense_ratio_pct: float
        price_series: [{"date": ..., "price": float}]
        benchmark_series: [{"date": ..., "value": float}]
        etf_source / benchmark_source: source dicts (optional)
        rolling_window_points, rolling_step_points, rolling_min_windows: int (optional)
        expense_impact: {investment_amount, horizons_years} (optional)

    Returns same shape as analyze_mutual_fund:
        {"insights": [...], "suppressed_insights": [...], "data_quality": {...}}
    """
    etf_name = str(payload.get("etf_name", "ETF"))
    benchmark_name = str(payload.get("benchmark_name", "Benchmark"))
    etf_source = _series_source(payload, "etf_source", "submitted_etf_series")
    benchmark_source = _series_source(payload, "benchmark_source", "submitted_benchmark_series")

    price_series = _normalize_series(
        list(payload.get("price_series", [])), "price", "etf",
    )
    benchmark_series = _normalize_series(
        list(payload.get("benchmark_series", [])), "value", "benchmark",
    )
    aligned = _align_series(price_series.points, benchmark_series.points)
    if len(aligned) < 2:
        raise PolicyError(
            "data_validation_error",
            "ETF and benchmark require at least two common dates.",
            {
                "etf_points": len(price_series.points),
                "benchmark_points": len(benchmark_series.points),
                "intersection_points": len(aligned),
            },
        )

    normalization_events = price_series.events + benchmark_series.events
    outliers = price_series.outliers + benchmark_series.outliers
    quality = _data_quality(
        aligned,
        fund_count=len(price_series.points),
        benchmark_count=len(benchmark_series.points),
        normalization_events=normalization_events,
        outliers=outliers,
    )
    source = _source_lineage(etf_source, benchmark_source)
    compiled: list[dict] = []
    suppressed: list[dict] = []
    unavailable: list[str] = []

    latest = aligned[-1]

    # --- Trailing returns ---
    trailing_returns: dict[str, dict[str, float | str]] = {}
    for years in (1, 3, 5):
        start = _point_on_or_before(aligned, _target_date(latest.as_of, years))
        if start is None or start.as_of == latest.as_of:
            unavailable.append(f"{years}Y trailing return")
            continue
        actual_years = max((latest.as_of - start.as_of).days / 365.25, 0.0)
        etf_return = _annualized_return(start.fund_value, latest.fund_value, actual_years)
        bench_return = _annualized_return(start.benchmark_value, latest.benchmark_value, actual_years)
        trailing_returns[f"{years}Y"] = {
            "start_date": start.as_of.isoformat(),
            "end_date": latest.as_of.isoformat(),
            "actual_years": round(actual_years, 2),
            "etf_cagr_pct": _pct(etf_return),
            "benchmark_cagr_pct": _pct(bench_return),
            "difference_pct_points": _pct(etf_return - bench_return),
        }

    base_limitations = [
        "Aligned to common dates only.",
        "Historical analytics do not describe future outcomes.",
        "Price series may not reflect total return if distributions are not included.",
    ]
    if quality["data_completeness"] != "High":
        base_limitations.append(
            "Submitted series has fewer common observations than a daily trading calendar.",
        )
    if outliers:
        outlier_dates = ", ".join(sorted({item["date"] for item in outliers}))
        base_limitations.append(
            f"Observed extreme value changes on {outlier_dates} may affect rolling and drawdown metrics.",
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
                    "etf_name": etf_name,
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

    # --- Tracking difference ---
    td = _tracking_difference(aligned)
    _compile_or_record(
        {
            "type": "diagnostic",
            "observation": (
                f"Tracking difference versus {benchmark_name} is "
                f"{td['tracking_difference_pct']:.2f} percentage points over "
                f"{td['span_years']} years."
            ),
            "why_it_matters": (
                "Tracking difference measures the total cumulative gap between the ETF "
                "and its benchmark, capturing the combined effect of expenses, sampling, "
                "and replication methodology."
            ),
            "supporting_data": {
                "tracking_difference": td,
                "source": source,
            },
            "evidence_strength": _evidence_strength(quality),
            "data_completeness": quality["data_completeness"],
            "limitations": base_limitations + [
                "Tracking difference is measured over the full overlapping history and "
                "may differ across sub-periods.",
            ],
            "unavailable_components": [],
        },
        compiled,
        suppressed,
    )

    # --- Tracking error ---
    te = _tracking_error(aligned)
    te_evidence = _evidence_strength(quality)
    te_limitations = list(base_limitations) + [
        "Tracking error is per-observation standard deviation, not annualized.",
    ]
    if te["observation_count"] < EXPECTED_TRADING_DAYS_PER_YEAR:
        te_evidence = _downgrade_evidence(te_evidence)
        te_limitations.append(
            f"Tracking error is based on {te['observation_count']} observations, "
            f"below the {EXPECTED_TRADING_DAYS_PER_YEAR}-point annual threshold.",
        )
    _compile_or_record(
        {
            "type": "diagnostic",
            "observation": (
                f"Tracking error versus {benchmark_name} is "
                f"{te['tracking_error_pct']:.2f} percentage points."
            ),
            "why_it_matters": (
                "Tracking error measures the consistency of the ETF's deviation from its "
                "benchmark. A lower value indicates more consistent replication."
            ),
            "supporting_data": {
                "tracking_error": te,
                "source": source,
            },
            "evidence_strength": te_evidence,
            "data_completeness": quality["data_completeness"],
            "limitations": te_limitations,
            "unavailable_components": [],
        },
        compiled,
        suppressed,
    )

    # --- Rolling excess return ---
    window_points = int(
        payload.get("rolling_window_points", DEFAULT_ROLLING_WINDOW_POINTS)
        or DEFAULT_ROLLING_WINDOW_POINTS
    )
    step_points = int(
        payload.get("rolling_step_points", DEFAULT_ROLLING_STEP_POINTS)
        or DEFAULT_ROLLING_STEP_POINTS
    )
    min_windows = int(
        payload.get("rolling_min_windows", DEFAULT_MIN_ROLLING_WINDOWS)
        or DEFAULT_MIN_ROLLING_WINDOWS
    )
    windows = _rolling_windows(aligned, window_points, step_points)
    if windows:
        fund_returns = [item["fund_return"] for item in windows]
        benchmark_returns = [item["benchmark_return"] for item in windows]
        excess_returns = [item["excess_return"] for item in windows]
        wins = [v for v in excess_returns if v > 0]
        losses = [v for v in excess_returns if v < 0]
        hit_ratio = len(wins) / len(excess_returns) if excess_returns else 0.0
        span_summary = _window_span_summary(windows)
        expected_span = int(
            payload.get("expected_window_span_days", DEFAULT_EXPECTED_WINDOW_SPAN_DAYS)
            or DEFAULT_EXPECTED_WINDOW_SPAN_DAYS
        )
        base_evidence = _evidence_strength(quality)
        rolling_evidence, downgrade_reasons = _rolling_evidence(
            base_evidence, quality, windows, min_windows, expected_span,
        )
        excess_stats = {
            "sample_size": len(excess_returns),
            "window_count": len(excess_returns),
            "hit_ratio_pct": _pct(hit_ratio),
            "avg_win_pct": _pct(mean(wins)) if wins else 0.0,
            "avg_loss_pct": _pct(mean(losses)) if losses else 0.0,
            "spread_pct": _pct(pstdev(excess_returns)) if len(excess_returns) > 1 else 0.0,
            "etf_distribution": _distribution(fund_returns),
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
                    "methodology": "Each rolling window uses the same aligned start and end dates for ETF and benchmark.",
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
                + (
                    [
                        "Rolling windows span expected duration but contain sparse observations, "
                        "which may lower reliability of rolling metrics.",
                    ]
                    if float(quality["calendar_density"]) < 0.70
                    and span_summary["median_days"] >= expected_span * 0.80
                    else []
                )
                + ["Rolling windows overlap and are not independent observations."],
                "unavailable_components": [],
            },
            compiled,
            suppressed,
        )
    else:
        unavailable.append("rolling excess return distribution")

    # --- Drawdown ---
    etf_drawdown = _drawdown_profile(aligned, "fund_value")
    bench_drawdown = _drawdown_profile(aligned, "benchmark_value")
    _compile_or_record(
        {
            "type": "benchmark_comparison",
            "observation": (
                f"Maximum drawdown over aligned history is {etf_drawdown['max_drawdown_pct']:.2f}% "
                f"for {etf_name} and {bench_drawdown['max_drawdown_pct']:.2f}% for {benchmark_name}."
            ),
            "benchmark": {
                "name": benchmark_name,
                "methodology": "Running peak-to-current decline measured identically on aligned observations.",
                "source": "submitted_aligned_series",
            },
            "supporting_data": {
                "etf_drawdown": etf_drawdown,
                "benchmark_drawdown": bench_drawdown,
                "source": source,
            },
            "evidence_strength": _evidence_strength(quality),
            "data_completeness": quality["data_completeness"],
            "limitations": base_limitations
            + ["Drawdown duration depends on observation frequency."]
            + (
                ["Drawdown peak and trough identification may be affected by sparse observations."]
                if float(quality["calendar_density"]) < 0.70
                else []
            ),
            "unavailable_components": [],
        },
        compiled,
        suppressed,
    )

    # --- Expense drag ---
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
    expense_source = make_source("submitted_expense_ratio", _now_iso(), LICENSE_REDIS, [etf_source])
    _compile_or_record(
        {
            "type": "cost_tax",
            "scenario_a": {
                "name": "TER 0.00%",
                "terminal_value_proxy": round(investment_amount, 2),
            },
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
                "Cost impact is a hypothetical calculation and does not reconstruct historical price.",
                "Consult a qualified professional",
            ],
            "unavailable_components": ["tax_lot_detail"],
        },
        compiled,
        suppressed,
    )

    return {
        "insights": compiled,
        "suppressed_insights": suppressed,
        "data_quality": quality,
    }
