"""Portfolio Aggregator v2.

Combines 2-5 funds with user-defined weights into a synthetic portfolio
series, then computes combined behavior: return distribution, drawdown,
correlation, concentration, and per-fund contribution.

All metrics are computed on the common-date-aligned series.
No ranking, no advice, no recommendations.

Language note — banned words include: allocation, best, strong, improvement,
optimized, overweight, underweight, increase, reduce, rebalance.
Use: weight, proportion, contribution, observed, higher/lower.
"""
from __future__ import annotations

from datetime import date
from math import sqrt
from statistics import mean, median, pstdev
from typing import Any, Dict, List, Optional, Tuple

from .comparison import (
    align_multiple_series,
    alignment_quality,
    drawdown_profile,
    evidence_from_alignment,
    trailing_cagr,
    volatility_metrics,
    worst_case_evidence,
    DRAWDOWN_SIGNIFICANT_PCT,
)
from .mutual_funds import (
    SeriesPoint,
    _pct,
    _period_return,
    _annualized_return,
    _years_between,
)


# ── Portfolio series construction ──────────────────────────────

def compute_portfolio_series(
    aligned_values: Dict[int, List[float]],
    weights: Dict[int, float],
) -> List[float]:
    """Compute weighted portfolio NAV series, normalized to base 100.

    Each fund's daily return is weighted and combined, then the
    cumulative series is built from base 100.
    """
    codes = sorted(aligned_values.keys())
    n = len(next(iter(aligned_values.values())))
    if n == 0:
        return []

    portfolio = [100.0]
    for i in range(1, n):
        daily_return = 0.0
        for code in codes:
            prev = aligned_values[code][i - 1]
            curr = aligned_values[code][i]
            if prev > 0:
                fund_ret = (curr / prev) - 1.0
            else:
                fund_ret = 0.0
            daily_return += weights[code] * fund_ret
        portfolio.append(portfolio[-1] * (1.0 + daily_return))

    return portfolio


# ── Rolling returns (lump-sum, not SIP) ────────────────────────

def rolling_period_returns(
    dates: List[date],
    values: List[float],
    window_points: int = 252,
    step_points: int = 21,
) -> List[Dict[str, Any]]:
    """Compute rolling period returns over the portfolio series."""
    results = []
    i = 0
    while i + window_points <= len(values):
        start_idx = i
        end_idx = i + window_points - 1
        start_val = values[start_idx]
        end_val = values[end_idx]
        if start_val > 0:
            years = _years_between(dates[start_idx], dates[end_idx])
            if years > 0:
                cagr = _annualized_return(start_val, end_val, years)
                results.append({
                    "start_date": dates[start_idx].isoformat(),
                    "end_date": dates[end_idx].isoformat(),
                    "years": round(years, 2),
                    "cagr_pct": _pct(cagr),
                })
        i += step_points
    return results


def return_distribution(rolling_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize rolling return distribution."""
    if not rolling_results:
        return {
            "window_count": 0,
            "median_cagr_pct": None,
            "p25_cagr_pct": None,
            "p75_cagr_pct": None,
            "min_cagr_pct": None,
            "max_cagr_pct": None,
        }

    cagrs = sorted(r["cagr_pct"] for r in rolling_results)
    n = len(cagrs)

    def _pctile(data, p):
        k = (len(data) - 1) * p / 100.0
        f = int(k)
        c = f + 1
        if c >= len(data):
            return data[-1]
        return data[f] + (k - f) * (data[c] - data[f])

    return {
        "window_count": n,
        "median_cagr_pct": round(_pctile(cagrs, 50), 2),
        "p25_cagr_pct": round(_pctile(cagrs, 25), 2),
        "p75_cagr_pct": round(_pctile(cagrs, 75), 2),
        "min_cagr_pct": round(cagrs[0], 2),
        "max_cagr_pct": round(cagrs[-1], 2),
    }


# ── Pairwise correlation ──────────────────────────────────────

def _daily_returns(values: List[float]) -> List[float]:
    """Compute daily returns from a value series."""
    return [
        (values[i] / values[i - 1]) - 1.0
        if values[i - 1] > 0 else 0.0
        for i in range(1, len(values))
    ]


def _pearson_correlation(xs: List[float], ys: List[float]) -> float:
    """Pearson correlation coefficient between two return series."""
    n = len(xs)
    if n < 3:
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    sx = pstdev(xs)
    sy = pstdev(ys)
    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)


def pairwise_correlations(
    aligned_values: Dict[int, List[float]],
    fund_names: Dict[int, str],
) -> Dict[str, Any]:
    """Compute pairwise return correlation matrix."""
    codes = sorted(aligned_values.keys())
    returns_map = {code: _daily_returns(aligned_values[code]) for code in codes}

    pairs = []
    for i, c1 in enumerate(codes):
        for c2 in codes[i + 1:]:
            corr = _pearson_correlation(returns_map[c1], returns_map[c2])
            pairs.append({
                "fund_a": fund_names.get(c1, str(c1)),
                "fund_b": fund_names.get(c2, str(c2)),
                "code_a": c1,
                "code_b": c2,
                "correlation": round(corr, 4),
            })

    # Average correlation
    avg_corr = round(mean(p["correlation"] for p in pairs), 4) if pairs else 0.0

    return {
        "pairs": pairs,
        "average_correlation": avg_corr,
        "pair_count": len(pairs),
    }


def _correlation_label(corr: float) -> str:
    """Neutral descriptive label for correlation level."""
    ac = abs(corr)
    if ac >= 0.80:
        return "very high"
    if ac >= 0.60:
        return "high"
    if ac >= 0.40:
        return "moderate"
    if ac >= 0.20:
        return "low"
    return "very low"


# ── Concentration metrics ──────────────────────────────────────

def concentration_metrics(weights: Dict[int, float]) -> Dict[str, Any]:
    """Compute portfolio concentration (HHI-based)."""
    w_list = list(weights.values())
    hhi = sum(w ** 2 for w in w_list)
    n = len(w_list)

    # Effective number of funds = 1/HHI
    effective_n = 1.0 / hhi if hhi > 0 else n

    max_weight = max(w_list) if w_list else 0.0
    min_weight = min(w_list) if w_list else 0.0

    return {
        "hhi": round(hhi, 4),
        "effective_fund_count": round(effective_n, 2),
        "actual_fund_count": n,
        "largest_weight": round(max_weight, 4),
        "smallest_weight": round(min_weight, 4),
    }


# ── Contribution analysis ─────────────────────────────────────

def contribution_analysis(
    dates: List[date],
    aligned_values: Dict[int, List[float]],
    weights: Dict[int, float],
    fund_names: Dict[int, str],
) -> Dict[str, Any]:
    """Per-fund contribution to portfolio return and drawdown."""
    codes = sorted(aligned_values.keys())
    n = len(dates)
    if n < 2:
        return {"funds": []}

    # Per-fund weighted return contribution over full period
    fund_contributions = []
    portfolio_return = 0.0

    for code in codes:
        first = aligned_values[code][0]
        last = aligned_values[code][-1]
        if first > 0:
            fund_total_return = (last / first) - 1.0
        else:
            fund_total_return = 0.0
        weighted_return = weights[code] * fund_total_return
        portfolio_return += weighted_return

        # Per-fund drawdown (on that fund's own series)
        dd = drawdown_profile(dates, aligned_values[code])

        fund_contributions.append({
            "scheme_code": code,
            "name": fund_names.get(code, str(code)),
            "weight": round(weights[code], 4),
            "fund_return_pct": _pct(fund_total_return),
            "weighted_return_pct": _pct(weighted_return),
            "fund_max_drawdown_pct": dd["max_drawdown_pct"],
        })

    return {
        "portfolio_return_pct": _pct(portfolio_return),
        "funds": fund_contributions,
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


def _data_completeness_from_quality(quality: Dict[str, Any]) -> str:
    rc = quality["relative_completeness"]
    if rc >= 0.90:
        return "High"
    if rc >= 0.70:
        return "Medium"
    return "Low"


def build_portfolio_insights(
    dates: List[date],
    aligned_values: Dict[int, List[float]],
    fund_names: Dict[int, str],
    weights: Dict[int, float],
    quality: Dict[str, Any],
    rolling_window_points: int = 252,
    rolling_step_points: int = 21,
    per_fund_evidence: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Build portfolio aggregation insights.

    Returns list of raw insight dicts using the 'diagnostic' template.
    Evidence = worst-case of alignment quality and per-fund data quality.
    """
    alignment_ev = evidence_from_alignment(quality)
    evidence = worst_case_evidence(alignment_ev, per_fund_evidence or [])
    base_lims = _base_limitations(quality)
    data_comp = _data_completeness_from_quality(quality)
    codes = sorted(aligned_values.keys())
    insights: List[Dict[str, Any]] = []

    if len(dates) < 2:
        return insights

    # Build combined portfolio series
    portfolio_series = compute_portfolio_series(aligned_values, weights)

    # 1) Portfolio return behavior (trailing CAGR)
    trailing = trailing_cagr(dates, portfolio_series, horizons_years=(1, 3, 5))
    if trailing:
        horizons_text = ", ".join(
            f"{h}: {trailing[h]['cagr_pct']}%"
            for h in sorted(trailing.keys())
        )
        insights.append({
            "type": "diagnostic",
            "observation": (
                f"Combined portfolio trailing CAGR on "
                f"{quality['aligned_points']} common dates: {horizons_text}."
            ),
            "why_it_matters": (
                "Trailing CAGR shows how the weighted combination of "
                "selected funds performed over standard horizons."
            ),
            "supporting_data": {
                "trailing_cagr": trailing,
                "fund_count": len(codes),
                "fund_weights": {
                    fund_names.get(c, str(c)): round(w, 4)
                    for c, w in weights.items()
                },
            },
            "evidence_strength": evidence,
            "data_completeness": data_comp,
            "limitations": base_lims + [
                "Portfolio series is synthetically constructed from weighted daily returns.",
            ],
            "unavailable_components": [],
        })

    # 2) Rolling return distribution
    rolling = rolling_period_returns(
        dates, portfolio_series, rolling_window_points, rolling_step_points,
    )
    dist = return_distribution(rolling)
    if dist["window_count"] > 0:
        insights.append({
            "type": "diagnostic",
            "observation": (
                f"Rolling return distribution across {dist['window_count']} windows: "
                f"median CAGR {dist['median_cagr_pct']}%, "
                f"range {dist['min_cagr_pct']}% to {dist['max_cagr_pct']}%."
            ),
            "why_it_matters": (
                "Rolling returns show the range of annualized outcomes "
                "for a lump-sum held across different historical periods."
            ),
            "supporting_data": {
                "distribution": dist,
                "window_points": rolling_window_points,
                "step_points": rolling_step_points,
            },
            "evidence_strength": evidence,
            "data_completeness": "High" if dist["window_count"] >= 50 else (
                "Medium" if dist["window_count"] >= 10 else "Low"
            ),
            "limitations": base_lims + [
                "Rolling windows overlap; individual windows are not independent.",
            ],
            "unavailable_components": [],
        })

    # 3) Portfolio drawdown profile
    dd = drawdown_profile(dates, portfolio_series)
    insights.append({
        "type": "diagnostic",
        "observation": (
            f"Combined portfolio experienced a maximum drawdown of "
            f"{dd['max_drawdown_pct']}% with "
            f"{dd['drawdowns_gt_threshold_pct']} episode(s) "
            f"exceeding {dd['threshold_pct']}%."
        ),
        "why_it_matters": (
            "Portfolio drawdown shows the depth and frequency of declines "
            "from running peak for the weighted fund combination."
        ),
        "supporting_data": {
            "drawdown": dd,
            "fund_count": len(codes),
        },
        "evidence_strength": evidence,
        "data_completeness": data_comp,
        "limitations": base_lims + [
            "Drawdown is measured on the synthetic portfolio series.",
            "Drawdown episodes still open at series end have no recovery date.",
        ],
        "unavailable_components": [],
    })

    # 4) Volatility
    vol = volatility_metrics(portfolio_series, dates)
    insights.append({
        "type": "diagnostic",
        "observation": (
            f"Combined portfolio return variability: "
            f"period-to-period standard deviation {vol['periodic_return_std_pct']}% "
            f"across {vol['observation_count']} observations."
        ),
        "why_it_matters": (
            "Period-to-period return variability indicates "
            "how much the combined portfolio value fluctuates between observations."
        ),
        "supporting_data": {
            "volatility": vol,
        },
        "evidence_strength": evidence,
        "data_completeness": data_comp,
        "limitations": base_lims + [
            "Volatility reflects raw period-to-period standard deviation, not annualized.",
        ],
        "unavailable_components": [],
    })

    # 5) Correlation analysis
    if len(codes) >= 2:
        corr = pairwise_correlations(aligned_values, fund_names)
        pair_desc = "; ".join(
            f"{p['fund_a']} vs {p['fund_b']}: {p['correlation']}"
            for p in corr["pairs"][:5]  # cap display for readability
        )
        insights.append({
            "type": "diagnostic",
            "observation": (
                f"Pairwise return correlations across {corr['pair_count']} pair(s): "
                f"{pair_desc}. "
                f"Average pairwise correlation: {corr['average_correlation']}."
            ),
            "why_it_matters": (
                "Correlation indicates how similarly two funds move. "
                "High correlation means less diversification effect; "
                "low correlation means the combined series may be less volatile "
                "than individual funds."
            ),
            "supporting_data": {
                "correlation_pairs": corr["pairs"],
                "average_correlation": corr["average_correlation"],
            },
            "evidence_strength": evidence,
            "data_completeness": "High" if quality["aligned_points"] >= 100 else (
                "Medium" if quality["aligned_points"] >= 20 else "Low"
            ),
            "limitations": base_lims + [
                "Correlation is computed on daily returns over the full common period.",
                "Correlation may vary across different sub-periods.",
            ],
            "unavailable_components": [],
        })

    # 6) Concentration
    conc = concentration_metrics(weights)
    insights.append({
        "type": "diagnostic",
        "observation": (
            f"Portfolio concentration: HHI {conc['hhi']}, "
            f"effective fund count {conc['effective_fund_count']} "
            f"out of {conc['actual_fund_count']} funds. "
            f"Largest weight: {conc['largest_weight']:.0%}, "
            f"smallest: {conc['smallest_weight']:.0%}."
        ),
        "why_it_matters": (
            "HHI (Herfindahl-Hirschman Index) measures weight concentration. "
            "Effective fund count shows how many equally weighted funds "
            "would produce the same concentration level."
        ),
        "supporting_data": {
            "concentration": conc,
        },
        "evidence_strength": evidence,
        "data_completeness": "High",
        "limitations": [
            "Concentration is computed from user-specified weights only.",
            "Does not account for underlying holdings overlap between funds.",
        ],
        "unavailable_components": [],
    })

    # 7) Contribution analysis
    contrib = contribution_analysis(dates, aligned_values, weights, fund_names)
    fund_contrib_desc = "; ".join(
        f"{f['name']}: {f['weighted_return_pct']}%"
        for f in contrib["funds"]
    )
    insights.append({
        "type": "diagnostic",
        "observation": (
            f"Per-fund weighted return contribution over the full period: "
            f"{fund_contrib_desc}. "
            f"Combined portfolio return: {contrib['portfolio_return_pct']}%."
        ),
        "why_it_matters": (
            "Contribution analysis shows how each fund's return, scaled by "
            "its weight, contributed to the total portfolio outcome."
        ),
        "supporting_data": {
            "contribution": contrib,
        },
        "evidence_strength": evidence,
        "data_completeness": data_comp,
        "limitations": base_lims + [
            "Contribution is over the full common period; sub-period contributions may differ.",
        ],
        "unavailable_components": [],
    })

    return insights
