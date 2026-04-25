"""Cross-fund comparison engine.

Computes side-by-side metrics on a **global date intersection** across
2-5 funds.  Produces deterministic, neutral comparison insights using
only the ``diagnostic`` template.

All metrics are computed on the aligned (common-date) series only.
No ranking, no "better/worse", no recommendations.
"""
from __future__ import annotations

from datetime import date
from statistics import mean, median, pstdev
from typing import Any, Dict, List, Tuple

from .mutual_funds import (
    SeriesPoint,
    _pct,
    _period_return,
    _annualized_return,
    _years_between,
    _business_days_between,
    _target_date,
)


# ── Constants ──────────────────────────────────────────────────
DRAWDOWN_SIGNIFICANT_PCT = 10.0   # threshold for "significant" drawdowns
MAX_FUNDS = 5
MIN_FUNDS = 2


# ── Alignment ─────────────────────────────────────────────────

def align_multiple_series(
    fund_series: Dict[int, List[SeriesPoint]],
) -> Tuple[List[date], Dict[int, List[float]]]:
    """Compute the global date intersection across all funds.

    Returns:
      common_dates — sorted list of dates present in ALL funds
      aligned_values — {scheme_code: [value_at_date_0, value_at_date_1, ...]}
    """
    if not fund_series:
        return [], {}

    date_sets = [
        {p.as_of for p in points}
        for points in fund_series.values()
    ]
    common = date_sets[0]
    for ds in date_sets[1:]:
        common = common & ds

    common_dates = sorted(common)
    if not common_dates:
        return [], {}

    aligned: Dict[int, List[float]] = {}
    for code, points in fund_series.items():
        by_date = {p.as_of: p.value for p in points}
        aligned[code] = [by_date[d] for d in common_dates]

    return common_dates, aligned


def alignment_quality(
    common_dates: List[date],
    fund_series: Dict[int, List[SeriesPoint]],
) -> Dict[str, Any]:
    """Compute alignment quality metadata."""
    if not common_dates or len(common_dates) < 2:
        return {
            "aligned_points": 0,
            "max_series_points": 0,
            "relative_completeness": 0.0,
            "calendar_density": 0.0,
            "history_years": 0.0,
        }

    max_points = max(len(pts) for pts in fund_series.values())
    expected_bdays = _business_days_between(common_dates[0], common_dates[-1])
    calendar_density = len(common_dates) / expected_bdays if expected_bdays else 0.0
    relative_completeness = len(common_dates) / max_points if max_points else 0.0
    history_years = _years_between(common_dates[0], common_dates[-1])

    return {
        "aligned_points": len(common_dates),
        "max_series_points": max_points,
        "relative_completeness": round(relative_completeness, 4),
        "calendar_density": round(calendar_density, 4),
        "history_years": round(history_years, 2),
    }


def evidence_from_alignment(quality: Dict[str, Any]) -> str:
    """Determine evidence strength from alignment quality."""
    aligned = quality["aligned_points"]
    years = quality["history_years"]
    density = quality["calendar_density"]

    if years >= 3.0 and aligned >= 200 and density >= 0.60:
        return "High"
    if years >= 1.0 and aligned >= 50 and density >= 0.40:
        return "Medium"
    return "Low"


_EVIDENCE_RANK = {"High": 2, "Medium": 1, "Low": 0}
_EVIDENCE_FROM_RANK = {v: k for k, v in _EVIDENCE_RANK.items()}


def worst_case_evidence(
    alignment_evidence: str,
    per_fund_evidence: List[str],
) -> str:
    """Return the weakest evidence across alignment and all per-fund qualities.

    Evidence = min(alignment_evidence, worst per-fund evidence).
    """
    levels = [alignment_evidence] + list(per_fund_evidence)
    min_rank = min(_EVIDENCE_RANK.get(e, 0) for e in levels)
    return _EVIDENCE_FROM_RANK.get(min_rank, "Low")


# ── Per-fund metrics on aligned series ─────────────────────────

def trailing_cagr(
    dates: List[date],
    values: List[float],
    horizons_years: Tuple[int, ...] = (1, 3, 5),
) -> Dict[str, Any]:
    """Compute trailing CAGR for given horizons on aligned series."""
    if len(dates) < 2:
        return {}

    latest_date = dates[-1]
    latest_value = values[-1]
    results: Dict[str, Any] = {}

    for years in horizons_years:
        target = _target_date(latest_date, years)
        # Find closest date on or before target
        eligible = [(i, d) for i, d in enumerate(dates) if d <= target]
        if not eligible:
            continue
        idx, start_date = max(eligible, key=lambda x: x[1])
        start_value = values[idx]
        actual_years = _years_between(start_date, latest_date)
        if actual_years <= 0:
            continue
        cagr = _annualized_return(start_value, latest_value, actual_years)
        results[f"{years}Y"] = {
            "start_date": start_date.isoformat(),
            "end_date": latest_date.isoformat(),
            "actual_years": round(actual_years, 2),
            "cagr_pct": _pct(cagr),
        }

    return results


def drawdown_profile(
    dates: List[date],
    values: List[float],
) -> Dict[str, Any]:
    """Compute full drawdown profile on aligned series."""
    if len(dates) < 2:
        return {
            "max_drawdown_pct": 0.0,
            "drawdowns_gt_threshold_pct": 0,
            "avg_recovery_days": None,
            "max_recovery_days": None,
            "threshold_pct": DRAWDOWN_SIGNIFICANT_PCT,
        }

    # Track all drawdown episodes
    peak = values[0]
    peak_date = dates[0]
    episodes: List[Dict[str, Any]] = []
    current_episode_start: date = dates[0]
    current_episode_peak = peak
    in_drawdown = False
    max_dd = 0.0
    trough_date = dates[0]

    for i in range(len(dates)):
        v = values[i]
        d = dates[i]

        if v >= peak:
            # New peak — close any open episode
            if in_drawdown:
                dd_pct = ((current_episode_trough_val / current_episode_peak) - 1.0) * 100.0
                episodes.append({
                    "start_date": current_episode_start.isoformat(),
                    "trough_date": current_episode_trough_date.isoformat(),
                    "recovery_date": d.isoformat(),
                    "drawdown_pct": round(dd_pct, 2),
                    "recovery_days": (d - current_episode_trough_date).days,
                })
                in_drawdown = False

            peak = v
            peak_date = d
        else:
            dd = (v / peak) - 1.0
            if dd < -0.001 and not in_drawdown:
                # Start new episode
                in_drawdown = True
                current_episode_start = peak_date
                current_episode_peak = peak
                current_episode_trough_val = v
                current_episode_trough_date = d

            if in_drawdown and v < current_episode_trough_val:
                current_episode_trough_val = v
                current_episode_trough_date = d

            if dd * 100.0 < max_dd:
                max_dd = dd * 100.0
                trough_date = d

    # Close any still-open episode (unrecovered)
    if in_drawdown:
        dd_pct = ((current_episode_trough_val / current_episode_peak) - 1.0) * 100.0
        episodes.append({
            "start_date": current_episode_start.isoformat(),
            "trough_date": current_episode_trough_date.isoformat(),
            "recovery_date": None,
            "drawdown_pct": round(dd_pct, 2),
            "recovery_days": None,
        })

    # Significant drawdowns (> threshold)
    significant = [
        ep for ep in episodes
        if abs(ep["drawdown_pct"]) >= DRAWDOWN_SIGNIFICANT_PCT
    ]

    recovery_days = [
        ep["recovery_days"] for ep in significant
        if ep["recovery_days"] is not None
    ]

    return {
        "max_drawdown_pct": round(max_dd, 2),
        "drawdowns_gt_threshold_pct": len(significant),
        "threshold_pct": DRAWDOWN_SIGNIFICANT_PCT,
        "avg_recovery_days": round(mean(recovery_days)) if recovery_days else None,
        "max_recovery_days": max(recovery_days) if recovery_days else None,
    }


def volatility_metrics(
    values: List[float],
    dates: List[date],
) -> Dict[str, Any]:
    """Compute periodic return volatility on aligned series."""
    if len(values) < 3:
        return {"periodic_return_std_pct": 0.0, "observation_count": len(values)}

    returns = [
        _period_return(values[i], values[i + 1])
        for i in range(len(values) - 1)
    ]

    return {
        "periodic_return_std_pct": round(pstdev(returns) * 100.0, 4),
        "observation_count": len(returns),
        "note": "Standard deviation of period-to-period returns; not annualized.",
    }


def consistency_vs_peers(
    dates: List[date],
    aligned_values: Dict[int, List[float]],
    window_points: int,
    step_points: int,
) -> Dict[str, Any]:
    """Compute per-fund consistency versus the peer median.

    For each rolling window, compute each fund's return.
    Then compute the peer median return per window.
    For each fund: hit ratio vs median, avg excess/deficit, spread.
    """
    n = len(dates)
    if n < window_points or window_points < 2:
        return {"window_count": 0, "funds": {}}

    codes = sorted(aligned_values.keys())

    # Compute returns per fund per window
    fund_window_returns: Dict[int, List[float]] = {code: [] for code in codes}
    window_indices: List[Tuple[int, int]] = []

    for start in range(0, n - window_points + 1, step_points):
        end = start + window_points - 1
        window_indices.append((start, end))
        for code in codes:
            vals = aligned_values[code]
            ret = _period_return(vals[start], vals[end])
            fund_window_returns[code].append(ret)

    window_count = len(window_indices)
    if window_count == 0:
        return {"window_count": 0, "funds": {}}

    # Per-window peer median
    per_window_medians: List[float] = []
    for wi in range(window_count):
        window_rets = [fund_window_returns[code][wi] for code in codes]
        per_window_medians.append(median(window_rets))

    # Per-fund stats vs median
    fund_stats: Dict[int, Dict[str, Any]] = {}
    for code in codes:
        excess_list = [
            fund_window_returns[code][wi] - per_window_medians[wi]
            for wi in range(window_count)
        ]
        above = [e for e in excess_list if e > 0]
        below = [e for e in excess_list if e < 0]
        hit_ratio = len(above) / window_count if window_count else 0.0

        fund_stats[code] = {
            "hit_ratio_vs_peer_median": round(hit_ratio, 4),
            "avg_excess_when_above": _pct(mean(above)) if above else 0.0,
            "avg_deficit_when_below": _pct(mean(below)) if below else 0.0,
            "excess_spread_pct": _pct(pstdev(excess_list)) if len(excess_list) > 1 else 0.0,
        }

    return {
        "window_count": window_count,
        "window_points": window_points,
        "step_points": step_points,
        "funds": fund_stats,
    }


def cost_impact_comparison(
    fund_expense_ratios: Dict[int, float],
    investment_amount: float = 100000.0,
    horizons_years: Tuple[float, ...] = (3.0, 5.0, 10.0),
) -> Dict[str, Any]:
    """Compute cost drag comparison across funds."""
    funds_data: Dict[int, Dict[str, Any]] = {}
    for code, ter_pct in fund_expense_ratios.items():
        ter = ter_pct / 100.0
        drag_factors = []
        drag_amounts = []
        for years in horizons_years:
            df = (1.0 - ter) ** years
            drag_factors.append(round(df, 6))
            drag_amounts.append(round(investment_amount * (1.0 - df), 2))
        funds_data[code] = {
            "expense_ratio_pct": round(ter_pct, 2),
            "drag_factors": drag_factors,
            "estimated_drag_amounts": drag_amounts,
        }

    return {
        "investment_amount": investment_amount,
        "horizons_years": list(horizons_years),
        "funds": funds_data,
    }


# ── Insight builders (all use 'diagnostic' template) ──────────

def _base_limitations(quality: Dict[str, Any]) -> List[str]:
    lims = [
        "Computed on dates common to all selected funds.",
        "Historical analytics do not describe future outcomes.",
    ]
    if quality["relative_completeness"] < 0.80:
        lims.append(
            "Common date intersection covers "
            f"{quality['relative_completeness']:.0%} of the longest series."
        )
    return lims


def build_comparison_insights(
    dates: List[date],
    aligned_values: Dict[int, List[float]],
    fund_names: Dict[int, str],
    fund_expense_ratios: Dict[int, float],
    quality: Dict[str, Any],
    window_points: int = 60,
    step_points: int = 5,
    per_fund_evidence: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """Build all comparison insight payloads (ready for compile_insight).

    Returns list of raw insight dicts using the 'diagnostic' template.
    Evidence is the worst-case of alignment quality AND per-fund data quality.
    """
    alignment_evidence = evidence_from_alignment(quality)
    evidence = worst_case_evidence(
        alignment_evidence, per_fund_evidence or [],
    )
    base_lims = _base_limitations(quality)
    codes = sorted(aligned_values.keys())
    insights: List[Dict[str, Any]] = []

    if len(dates) < 2:
        return insights

    # 1) Trailing CAGR comparison
    cagr_data = {}
    for code in codes:
        cagr_data[code] = trailing_cagr(dates, aligned_values[code])

    # Find which horizons are available across ALL funds
    all_horizons = set()
    for data in cagr_data.values():
        all_horizons.update(data.keys())
    common_horizons = sorted(all_horizons)

    if common_horizons:
        per_fund_cagr = []
        for code in codes:
            entry = {"scheme_code": code, "name": fund_names.get(code, str(code))}
            for h in common_horizons:
                if h in cagr_data[code]:
                    entry[f"{h}_cagr_pct"] = cagr_data[code][h]["cagr_pct"]
                else:
                    entry[f"{h}_cagr_pct"] = None
            per_fund_cagr.append(entry)

        insights.append({
            "type": "diagnostic",
            "observation": (
                f"Trailing CAGR comparison across {len(codes)} funds "
                f"on {quality['aligned_points']} common dates."
            ),
            "why_it_matters": (
                "Trailing returns show how each fund's NAV changed "
                "over identical calendar windows on common observation dates."
            ),
            "supporting_data": {
                "horizons": common_horizons,
                "funds": per_fund_cagr,
                "alignment": quality,
            },
            "evidence_strength": evidence,
            "data_completeness": "High" if quality["relative_completeness"] >= 0.90 else (
                "Medium" if quality["relative_completeness"] >= 0.70 else "Low"
            ),
            "limitations": base_lims + [
                "CAGR is annualized from the closest available date; actual start dates may differ slightly across horizons.",
            ],
            "unavailable_components": [],
        })

    # 2) Drawdown comparison
    dd_data = []
    for code in codes:
        dd = drawdown_profile(dates, aligned_values[code])
        dd["scheme_code"] = code
        dd["name"] = fund_names.get(code, str(code))
        dd_data.append(dd)

    insights.append({
        "type": "diagnostic",
        "observation": (
            f"Drawdown comparison across {len(codes)} funds "
            f"on {quality['aligned_points']} common dates."
        ),
        "why_it_matters": (
            "Drawdown profiles show the depth and frequency of "
            f"declines exceeding {DRAWDOWN_SIGNIFICANT_PCT}% from a running peak."
        ),
        "supporting_data": {
            "funds": dd_data,
            "alignment": quality,
        },
        "evidence_strength": evidence,
        "data_completeness": "High" if quality["calendar_density"] >= 0.70 else (
            "Medium" if quality["calendar_density"] >= 0.40 else "Low"
        ),
        "limitations": base_lims + [
            "Recovery metrics depend on observation frequency and data completeness.",
            "Drawdown episodes still open at series end have no recovery date or duration.",
        ],
        "unavailable_components": [],
    })

    # 3) Volatility comparison
    vol_data = []
    for code in codes:
        vol = volatility_metrics(aligned_values[code], dates)
        vol["scheme_code"] = code
        vol["name"] = fund_names.get(code, str(code))
        vol_data.append(vol)

    insights.append({
        "type": "diagnostic",
        "observation": (
            f"Volatility comparison across {len(codes)} funds "
            f"({quality['aligned_points']} common observations)."
        ),
        "why_it_matters": (
            "Period-to-period return variability indicates "
            "how much each fund's NAV fluctuates between observations."
        ),
        "supporting_data": {
            "funds": vol_data,
            "alignment": quality,
        },
        "evidence_strength": evidence,
        "data_completeness": "High" if quality["aligned_points"] >= 100 else (
            "Medium" if quality["aligned_points"] >= 20 else "Low"
        ),
        "limitations": base_lims + [
            "Volatility is not annualized; it reflects raw period-to-period standard deviation.",
        ],
        "unavailable_components": [],
    })

    # 4) Consistency vs peer median
    if len(dates) >= window_points:
        cons = consistency_vs_peers(dates, aligned_values, window_points, step_points)
        if cons["window_count"] > 0:
            fund_cons_data = []
            for code in codes:
                entry = {"scheme_code": code, "name": fund_names.get(code, str(code))}
                entry.update(cons["funds"].get(code, {}))
                fund_cons_data.append(entry)

            insights.append({
                "type": "diagnostic",
                "observation": (
                    f"Rolling consistency versus peer median across "
                    f"{cons['window_count']} windows of {window_points} observations."
                ),
                "why_it_matters": (
                    "Hit ratio versus peer median shows how often each fund's "
                    "rolling return was above the group median in that window. "
                    "Peer median is computed per window; no external index is used."
                ),
                "supporting_data": {
                    "window_count": cons["window_count"],
                    "window_points": cons["window_points"],
                    "step_points": cons["step_points"],
                    "funds": fund_cons_data,
                    "alignment": quality,
                },
                "evidence_strength": evidence if cons["window_count"] >= 10 else "Low",
                "data_completeness": "High" if cons["window_count"] >= 50 else (
                    "Medium" if cons["window_count"] >= 10 else "Low"
                ),
                "limitations": base_lims + [
                    "Peer median is computed per window; no external index is used.",
                    "Rolling windows overlap and are not independent observations.",
                ],
                "unavailable_components": [],
            })

    # 5) Cost impact comparison
    if any(ter > 0 for ter in fund_expense_ratios.values()):
        cost = cost_impact_comparison(fund_expense_ratios)
        fund_cost_data = []
        for code in codes:
            entry = {"scheme_code": code, "name": fund_names.get(code, str(code))}
            entry.update(cost["funds"].get(code, {}))
            fund_cost_data.append(entry)

        insights.append({
            "type": "diagnostic",
            "observation": (
                f"Cost drag comparison across {len(codes)} funds "
                f"over {', '.join(str(int(h)) for h in cost['horizons_years'])} year horizons."
            ),
            "why_it_matters": (
                "Expense ratios compound over time. "
                "Drag factor shows the fraction of a hypothetical investment "
                "retained after annual cost deduction."
            ),
            "supporting_data": {
                "investment_amount": cost["investment_amount"],
                "horizons_years": cost["horizons_years"],
                "funds": fund_cost_data,
            },
            "evidence_strength": "Low",
            "data_completeness": "Medium",
            "limitations": base_lims + [
                "Cost impact is a hypothetical calculation using stated expense ratios.",
                "Actual costs may vary; check latest fund documents.",
            ],
            "unavailable_components": [],
        })

    return insights
