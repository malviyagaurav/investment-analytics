"""SIP (Systematic Investment Plan) simulator.

Simulates rolling historical SIP outcomes across 1-5 funds with
user-defined contribution weights.  All metrics are backward-looking
and computed on common-date-aligned NAV series.

No predictions, no ranking, no recommendations.

Language note — the linter bans "allocation", "buy", "sell", "best",
"should", "avoid", "recommend", "increase", "reduce", etc.
Use "weight", "proportion", "contribution" instead.
"""
from __future__ import annotations

from datetime import date
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

from .comparison import (
    align_multiple_series,
    alignment_quality,
    evidence_from_alignment,
    worst_case_evidence,
)
from .mutual_funds import SeriesPoint, _pct


# ── Monthly date extraction ───────────────────────────────────

def monthly_contribution_dates(dates: List[date]) -> List[int]:
    """Return indices of the first available date per calendar month.

    Each index points into the ``dates`` (and aligned_values) arrays.
    """
    seen: set = set()
    indices: List[int] = []
    for i, d in enumerate(dates):
        key = (d.year, d.month)
        if key not in seen:
            seen.add(key)
            indices.append(i)
    return indices


# ── Single-window SIP simulation ──────────────────────────────

def simulate_sip_window(
    monthly_indices: List[int],
    aligned_values: Dict[int, List[float]],
    weights: Dict[int, float],
    monthly_amount: float,
    start_month: int,
    window_months: int,
) -> Optional[Dict[str, Any]]:
    """Simulate a single SIP window.

    Returns outcome dict or None if insufficient months.
    """
    end_month = start_month + window_months
    if end_month > len(monthly_indices):
        return None

    window_nav_indices = monthly_indices[start_month:end_month]
    codes = sorted(weights.keys())

    units: Dict[int, float] = {code: 0.0 for code in codes}
    total_invested = 0.0

    # Track portfolio value at each contribution date
    min_unrealized_pct = 0.0
    negative_months = 0
    current_negative_streak = 0
    longest_negative_streak = 0

    for nav_idx in window_nav_indices:
        # Contribute and acquire units
        for code in codes:
            amount = monthly_amount * weights[code]
            nav = aligned_values[code][nav_idx]
            if nav > 0:
                units[code] += amount / nav
        total_invested += monthly_amount

        # Portfolio value at this point
        current_value = sum(
            units[code] * aligned_values[code][nav_idx]
            for code in codes
        )
        unrealized_pct = (
            ((current_value / total_invested) - 1.0) * 100.0
            if total_invested > 0 else 0.0
        )

        if unrealized_pct < min_unrealized_pct:
            min_unrealized_pct = unrealized_pct

        if unrealized_pct < 0:
            negative_months += 1
            current_negative_streak += 1
            if current_negative_streak > longest_negative_streak:
                longest_negative_streak = current_negative_streak
        else:
            current_negative_streak = 0

    # Final value at last contribution date
    last_nav_idx = window_nav_indices[-1]
    final_value = sum(
        units[code] * aligned_values[code][last_nav_idx]
        for code in codes
    )
    total_return_pct = (
        ((final_value / total_invested) - 1.0) * 100.0
        if total_invested > 0 else 0.0
    )

    return {
        "total_invested": round(total_invested, 2),
        "final_value": round(final_value, 2),
        "total_return_pct": round(total_return_pct, 2),
        "deepest_unrealized_drawdown_pct": round(min_unrealized_pct, 2),
        "negative_months": negative_months,
        "longest_negative_streak_months": longest_negative_streak,
        "window_months": window_months,
    }


# ── Rolling simulation ────────────────────────────────────────

def rolling_sip_simulation(
    dates: List[date],
    aligned_values: Dict[int, List[float]],
    weights: Dict[int, float],
    monthly_amount: float,
    window_months: int = 36,
    step_months: int = 1,
) -> Dict[str, Any]:
    """Run rolling SIP simulation across all available windows."""
    monthly_indices = monthly_contribution_dates(dates)

    outcomes: List[Dict[str, Any]] = []
    start = 0
    while start + window_months <= len(monthly_indices):
        result = simulate_sip_window(
            monthly_indices, aligned_values, weights,
            monthly_amount, start, window_months,
        )
        if result:
            outcomes.append(result)
        start += step_months

    return {
        "window_months": window_months,
        "step_months": step_months,
        "total_monthly_dates": len(monthly_indices),
        "window_count": len(outcomes),
        "outcomes": outcomes,
    }


# ── Outcome statistics ────────────────────────────────────────

def _percentile(sorted_data: List[float], p: float) -> float:
    """Linear interpolation percentile on pre-sorted data."""
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * p / 100.0
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[-1]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def sip_return_distribution(outcomes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute return distribution across SIP windows."""
    if not outcomes:
        return {
            "window_count": 0,
            "median_return_pct": None,
            "p25_return_pct": None,
            "p75_return_pct": None,
            "min_return_pct": None,
            "max_return_pct": None,
        }

    returns = sorted(o["total_return_pct"] for o in outcomes)
    n = len(returns)

    return {
        "window_count": n,
        "median_return_pct": round(_percentile(returns, 50), 2),
        "p25_return_pct": round(_percentile(returns, 25), 2),
        "p75_return_pct": round(_percentile(returns, 75), 2),
        "min_return_pct": round(returns[0], 2),
        "max_return_pct": round(returns[-1], 2),
    }


def sip_drawdown_summary(outcomes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize drawdown behaviour during SIP across windows."""
    if not outcomes:
        return {
            "median_deepest_drawdown_pct": None,
            "extreme_deepest_drawdown_pct": None,
            "median_negative_months": None,
            "maximum_negative_months": None,
            "median_longest_negative_streak": None,
            "maximum_longest_negative_streak": None,
        }

    drawdowns = sorted(o["deepest_unrealized_drawdown_pct"] for o in outcomes)
    neg_months = sorted(o["negative_months"] for o in outcomes)
    streaks = sorted(o["longest_negative_streak_months"] for o in outcomes)

    return {
        "median_deepest_drawdown_pct": round(median(drawdowns), 2),
        "extreme_deepest_drawdown_pct": round(min(drawdowns), 2),
        "median_negative_months": round(median(neg_months), 1),
        "maximum_negative_months": max(neg_months),
        "median_longest_negative_streak": round(median(streaks), 1),
        "maximum_longest_negative_streak": max(streaks),
    }


def sip_consistency(
    outcomes: List[Dict[str, Any]],
    threshold_pct: float = 5.0,
) -> Dict[str, Any]:
    """Compute outcome consistency metrics."""
    if not outcomes:
        return {
            "positive_pct": None,
            "above_threshold_pct": None,
            "threshold_pct": threshold_pct,
            "window_count": 0,
        }

    n = len(outcomes)
    positive = sum(1 for o in outcomes if o["total_return_pct"] > 0)
    above = sum(1 for o in outcomes if o["total_return_pct"] >= threshold_pct)

    return {
        "positive_pct": round(positive / n * 100, 2),
        "above_threshold_pct": round(above / n * 100, 2),
        "threshold_pct": threshold_pct,
        "window_count": n,
    }


def sip_cost_impact(
    fund_expense_ratios: Dict[int, float],
    weights: Dict[int, float],
    monthly_amount: float,
    window_months: int = 36,
) -> Dict[str, Any]:
    """Estimate cost drag on SIP accumulation."""
    total_invested = monthly_amount * window_months

    # Weighted average TER
    weighted_ter = sum(
        fund_expense_ratios.get(code, 0.0) * w
        for code, w in weights.items()
    )

    # Monthly TER deduction
    monthly_ter = weighted_ter / 100.0 / 12.0

    accumulated_with_drag = 0.0
    for _ in range(window_months):
        accumulated_with_drag = (accumulated_with_drag + monthly_amount) * (1 - monthly_ter)

    drag_amount = total_invested - accumulated_with_drag

    return {
        "weighted_expense_ratio_pct": round(weighted_ter, 4),
        "total_invested": round(total_invested, 2),
        "estimated_cost_drag": round(drag_amount, 2),
        "drag_as_pct_of_invested": round(
            drag_amount / total_invested * 100, 4
        ) if total_invested > 0 else 0.0,
        "window_months": window_months,
    }


# ── Insight builders (all use 'diagnostic' template) ──────────

def _base_limitations(quality: Dict[str, Any]) -> List[str]:
    lims = [
        "Computed on dates common to all selected funds.",
        "Historical SIP simulation does not describe future outcomes.",
        "Actual investment timing and amounts may differ from this simulation.",
    ]
    if quality["relative_completeness"] < 0.80:
        lims.append(
            "Common date intersection covers "
            f"{quality['relative_completeness']:.0%} of the longest series."
        )
    return lims


def _data_completeness(window_count: int) -> str:
    if window_count >= 30:
        return "High"
    if window_count >= 10:
        return "Medium"
    return "Low"


def build_sip_insights(
    dates: List[date],
    aligned_values: Dict[int, List[float]],
    fund_names: Dict[int, str],
    fund_expense_ratios: Dict[int, float],
    weights: Dict[int, float],
    monthly_amount: float,
    quality: Dict[str, Any],
    window_months: int = 36,
    step_months: int = 1,
    per_fund_evidence: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build SIP simulation insights.

    Returns (insights_list, simulation_metadata).
    """
    alignment_ev = evidence_from_alignment(quality)
    evidence = worst_case_evidence(alignment_ev, per_fund_evidence or [])

    base_lims = _base_limitations(quality)
    insights: List[Dict[str, Any]] = []

    if len(dates) < 2:
        return insights, {"window_count": 0}

    # Run simulation
    sim = rolling_sip_simulation(
        dates, aligned_values, weights, monthly_amount,
        window_months, step_months,
    )
    outcomes = sim["outcomes"]

    sim_meta = {
        "window_count": sim["window_count"],
        "window_months": window_months,
        "step_months": step_months,
        "total_monthly_dates": sim["total_monthly_dates"],
        "aligned_points": quality["aligned_points"],
    }

    if not outcomes:
        return insights, sim_meta

    # Weight description (safe wording — no "allocation")
    weight_desc = ", ".join(
        f"{fund_names.get(code, str(code))} ({weights[code]:.0%})"
        for code in sorted(weights.keys())
    )

    # 1) Return distribution
    dist = sip_return_distribution(outcomes)
    total_invested = outcomes[0]["total_invested"]  # same for all windows
    final_values = sorted(o["final_value"] for o in outcomes)
    median_final = round(_percentile(final_values, 50), 2)
    min_final = round(final_values[0], 2)
    max_final = round(final_values[-1], 2)

    insights.append({
        "type": "diagnostic",
        "observation": (
            f"SIP return distribution across {dist['window_count']} rolling "
            f"{window_months}-month windows: "
            f"median {dist['median_return_pct']}%, "
            f"range {dist['min_return_pct']}% to {dist['max_return_pct']}%. "
            f"On {total_invested:.0f} invested, median final value was {median_final:.0f}."
        ),
        "why_it_matters": (
            "Rolling SIP outcomes show the range of historical results "
            "for a fixed monthly contribution across different starting points."
        ),
        "supporting_data": {
            "distribution": dist,
            "total_invested": total_invested,
            "median_final_value": median_final,
            "min_final_value": min_final,
            "max_final_value": max_final,
            "monthly_amount": monthly_amount,
            "fund_weights": {
                fund_names.get(c, str(c)): round(w, 4)
                for c, w in weights.items()
            },
            "aligned_points": quality["aligned_points"],
        },
        "evidence_strength": evidence,
        "data_completeness": _data_completeness(sim["window_count"]),
        "limitations": base_lims + [
            f"Derived from {sim['window_count']} overlapping windows; windows are not independent.",
        ],
        "unavailable_components": [],
    })

    # 2) Drawdown during SIP
    dd = sip_drawdown_summary(outcomes)
    insights.append({
        "type": "diagnostic",
        "observation": (
            f"During {window_months}-month SIP periods, "
            f"median peak unrealized drawdown was {dd['median_deepest_drawdown_pct']}%; "
            f"the deepest observed drawdown was {dd['extreme_deepest_drawdown_pct']}%."
        ),
        "why_it_matters": (
            "Unrealized drawdown during an active SIP shows how far "
            "the portfolio value dropped relative to total invested amount."
        ),
        "supporting_data": {
            "drawdown_summary": dd,
            "window_months": window_months,
            "window_count": len(outcomes),
        },
        "evidence_strength": evidence,
        "data_completeness": _data_completeness(sim["window_count"]),
        "limitations": base_lims + [
            "Drawdown is measured at monthly contribution dates only, not intra-month.",
        ],
        "unavailable_components": [],
    })

    # 3) Consistency
    cons = sip_consistency(outcomes)
    insights.append({
        "type": "diagnostic",
        "observation": (
            f"{cons['positive_pct']}% of {window_months}-month SIP windows "
            f"resulted in a positive return; "
            f"{cons['above_threshold_pct']}% exceeded {cons['threshold_pct']}% total return."
        ),
        "why_it_matters": (
            "Outcome consistency indicates how frequently a fixed "
            "monthly contribution produced a positive or threshold-exceeding "
            "result across historical starting points."
        ),
        "supporting_data": {
            "consistency": cons,
            "window_months": window_months,
        },
        "evidence_strength": evidence,
        "data_completeness": _data_completeness(sim["window_count"]),
        "limitations": base_lims,
        "unavailable_components": [],
    })

    # 4) Recovery analysis
    neg_streaks = [o["longest_negative_streak_months"] for o in outcomes]
    median_streak = round(median(neg_streaks), 1) if neg_streaks else 0
    max_streak = max(neg_streaks) if neg_streaks else 0
    pct_with_drawdown = round(
        sum(1 for s in neg_streaks if s > 0) / len(neg_streaks) * 100, 2
    ) if neg_streaks else 0

    insights.append({
        "type": "diagnostic",
        "observation": (
            f"Across {len(outcomes)} SIP windows, "
            f"{pct_with_drawdown}% experienced at least one month below invested amount. "
            f"Median longest negative streak: {median_streak} months; "
            f"longest observed: {max_streak} months."
        ),
        "why_it_matters": (
            "Recovery duration shows how long an active SIP remained below "
            "its total invested amount before returning to positive territory."
        ),
        "supporting_data": {
            "pct_with_negative_months": pct_with_drawdown,
            "median_longest_negative_streak_months": median_streak,
            "longest_observed_negative_streak_months": max_streak,
            "window_count": len(outcomes),
        },
        "evidence_strength": evidence,
        "data_completeness": _data_completeness(sim["window_count"]),
        "limitations": base_lims + [
            "Recovery is measured at monthly intervals; actual recovery may occur between dates.",
        ],
        "unavailable_components": [],
    })

    # 5) Cost impact
    if any(ter > 0 for ter in fund_expense_ratios.values()):
        cost = sip_cost_impact(
            fund_expense_ratios, weights, monthly_amount, window_months,
        )
        insights.append({
            "type": "diagnostic",
            "observation": (
                f"Estimated cost drag on a {window_months}-month SIP: "
                f"{cost['estimated_cost_drag']:.2f} "
                f"({cost['drag_as_pct_of_invested']}% of total invested) "
                f"at a weighted expense ratio of {cost['weighted_expense_ratio_pct']}%."
            ),
            "why_it_matters": (
                "Expense ratios compound over time. "
                "This estimates the hypothetical value impact "
                "due to recurring cost deductions during SIP tenure."
            ),
            "supporting_data": {
                "cost_impact": cost,
                "fund_weights": {
                    fund_names.get(c, str(c)): round(w, 4)
                    for c, w in weights.items()
                },
            },
            "evidence_strength": "Low",
            "data_completeness": "Medium",
            "limitations": base_lims + [
                "Cost impact is hypothetical; actual expense deductions may differ.",
                "Does not account for entry or exit loads or taxes.",
            ],
            "unavailable_components": [],
        })

    return insights, sim_meta
