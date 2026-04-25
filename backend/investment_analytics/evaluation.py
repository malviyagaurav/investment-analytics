"""Evaluation engine — constraint-based portfolio validation.

Evaluates portfolio insights against user-defined constraints.
Returns PASS/FAIL per constraint + structural red flags.

No recommendations. No rankings. No advice.
Only: does observed behavior meet user-defined tolerances?
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


# ── Default constraints (user can override all) ───────────────

DEFAULT_CONSTRAINTS: Dict[str, Any] = {
    "max_drawdown_pct": -30.0,          # worst acceptable drawdown (negative)
    "max_recovery_days": 730,            # ~2 years
    "min_median_rolling_cagr_pct": 5.0,  # minimum acceptable median rolling return
    "max_volatility_pct": 2.0,           # period-to-period std
    "max_correlation": 0.85,             # pairwise correlation threshold
    "max_concentration_hhi": 0.50,       # HHI threshold
    "max_single_fund_drawdown_pct": -40.0,  # per-fund drawdown limit
}

# Red flag thresholds (not configurable — structural signals)
_HIGH_CORRELATION_THRESHOLD = 0.90
_DOMINANT_CONTRIBUTION_PCT = 70.0
_EXTREME_CONCENTRATION_HHI = 0.60
_MIN_OBSERVATIONS_FOR_CORRELATION_FLAG = 30
_MIN_PORTFOLIO_RETURN_FOR_DOMINANCE = 1.0  # ignore dominance below 1% return


# ── Metric extraction from insight supporting_data ────────────

def _extract_metrics(insights: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract evaluable metrics from portfolio insight payloads.

    Works with both raw insights and compiled insights (which wrap
    the raw payload under 'payload' key).
    """
    metrics: Dict[str, Any] = {}

    for insight in insights:
        # Handle compiled insights: {template, labels, payload: {...}}
        sd = insight.get("supporting_data")
        if sd is None and "payload" in insight:
            sd = insight["payload"].get("supporting_data")
        if sd is None:
            continue

        # Drawdown
        if "drawdown" in sd:
            dd = sd["drawdown"]
            metrics["max_drawdown_pct"] = dd.get("max_drawdown_pct")
            metrics["avg_recovery_days"] = dd.get("avg_recovery_days")
            metrics["max_recovery_days"] = dd.get("max_recovery_days")
            metrics["drawdowns_gt_threshold"] = dd.get("drawdowns_gt_threshold_pct")

        # Rolling distribution
        if "distribution" in sd:
            dist = sd["distribution"]
            metrics["median_rolling_cagr_pct"] = dist.get("median_cagr_pct")
            metrics["min_rolling_cagr_pct"] = dist.get("min_cagr_pct")
            metrics["max_rolling_cagr_pct"] = dist.get("max_cagr_pct")
            metrics["p25_rolling_cagr_pct"] = dist.get("p25_cagr_pct")
            metrics["p75_rolling_cagr_pct"] = dist.get("p75_cagr_pct")
            metrics["rolling_window_count"] = dist.get("window_count")

        # Volatility
        if "volatility" in sd:
            vol = sd["volatility"]
            metrics["volatility_pct"] = vol.get("periodic_return_std_pct")
            metrics["observation_count"] = vol.get("observation_count")

        # Correlation
        if "correlation_pairs" in sd:
            pairs = sd["correlation_pairs"]
            metrics["correlation_pairs"] = pairs
            metrics["average_correlation"] = sd.get("average_correlation")
            if pairs:
                metrics["max_pairwise_correlation"] = max(
                    abs(p["correlation"]) for p in pairs
                )

        # Concentration
        if "concentration" in sd:
            conc = sd["concentration"]
            metrics["hhi"] = conc.get("hhi")
            metrics["effective_fund_count"] = conc.get("effective_fund_count")
            metrics["largest_weight"] = conc.get("largest_weight")

        # Contribution
        if "contribution" in sd:
            contrib = sd["contribution"]
            metrics["portfolio_return_pct"] = contrib.get("portfolio_return_pct")
            metrics["fund_contributions"] = contrib.get("funds", [])

        # Trailing CAGR
        if "trailing_cagr" in sd:
            metrics["trailing_cagr"] = sd["trailing_cagr"]

    return metrics


# ── Constraint evaluation ─────────────────────────────────────

def _check_constraint(
    name: str,
    observed: Any,
    threshold: Any,
    comparator: str = "gte",
) -> Dict[str, Any]:
    """Evaluate a single constraint.

    comparator:
      'gte'  — observed >= threshold → PASS (e.g., min return)
      'lte'  — observed <= threshold → PASS (e.g., max drawdown, max vol)
    """
    if observed is None:
        return {
            "name": name,
            "status": "INSUFFICIENT_DATA",
            "observed": None,
            "threshold": threshold,
            "why": f"No data available for {name}.",
        }

    if comparator == "gte":
        passed = observed >= threshold
    elif comparator == "lte":
        passed = observed <= threshold
    else:
        passed = False

    result = {
        "name": name,
        "status": "PASS" if passed else "FAIL",
        "observed": observed,
        "threshold": threshold,
    }
    if not passed:
        result["why"] = (
            f"{name}: observed {observed}, threshold {threshold}. "
            f"{'Observed must be >= threshold.' if comparator == 'gte' else 'Observed must be <= threshold.'}"
        )
    return result


def evaluate_constraints(
    metrics: Dict[str, Any],
    constraints: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Evaluate all constraints against extracted metrics."""
    checks: List[Dict[str, Any]] = []

    # Drawdown (negative values — observed must be >= threshold, i.e. less severe)
    if "max_drawdown_pct" in constraints:
        checks.append(_check_constraint(
            "max_drawdown",
            metrics.get("max_drawdown_pct"),
            constraints["max_drawdown_pct"],
            comparator="gte",  # -20 >= -30 → PASS
        ))

    # Recovery time
    if "max_recovery_days" in constraints:
        checks.append(_check_constraint(
            "max_recovery_days",
            metrics.get("max_recovery_days"),
            constraints["max_recovery_days"],
            comparator="lte",
        ))

    # Median rolling CAGR
    if "min_median_rolling_cagr_pct" in constraints:
        checks.append(_check_constraint(
            "min_median_rolling_cagr",
            metrics.get("median_rolling_cagr_pct"),
            constraints["min_median_rolling_cagr_pct"],
            comparator="gte",
        ))

    # Volatility
    if "max_volatility_pct" in constraints:
        checks.append(_check_constraint(
            "max_volatility",
            metrics.get("volatility_pct"),
            constraints["max_volatility_pct"],
            comparator="lte",
        ))

    # Max pairwise correlation
    if "max_correlation" in constraints:
        checks.append(_check_constraint(
            "max_pairwise_correlation",
            metrics.get("max_pairwise_correlation"),
            constraints["max_correlation"],
            comparator="lte",
        ))

    # Concentration HHI
    if "max_concentration_hhi" in constraints:
        checks.append(_check_constraint(
            "max_concentration_hhi",
            metrics.get("hhi"),
            constraints["max_concentration_hhi"],
            comparator="lte",
        ))

    # Per-fund drawdown
    if "max_single_fund_drawdown_pct" in constraints:
        fund_contribs = metrics.get("fund_contributions", [])
        for fund in fund_contribs:
            fund_dd = fund.get("fund_max_drawdown_pct")
            checks.append(_check_constraint(
                f"fund_drawdown_{fund.get('scheme_code', '?')}",
                fund_dd,
                constraints["max_single_fund_drawdown_pct"],
                comparator="gte",
            ))

    return checks


# ── Red flag detection ────────────────────────────────────────

def detect_flags(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Detect structural red flags independent of user constraints."""
    flags: List[Dict[str, Any]] = []

    # High pairwise correlation (only flag with sufficient observations)
    obs_count = metrics.get("observation_count")
    pairs = metrics.get("correlation_pairs", [])
    if obs_count is None or obs_count >= _MIN_OBSERVATIONS_FOR_CORRELATION_FLAG:
        for pair in pairs:
            if abs(pair.get("correlation", 0)) >= _HIGH_CORRELATION_THRESHOLD:
                flags.append({
                    "flag": "high_correlation",
                    "detail": (
                        f"{pair['fund_a']} and {pair['fund_b']} have "
                        f"correlation {pair['correlation']}"
                    ),
                    "value": pair["correlation"],
                })

    # Dominant contribution (one fund drives >70% of total return)
    fund_contribs = metrics.get("fund_contributions", [])
    portfolio_return = metrics.get("portfolio_return_pct")
    if (fund_contribs and portfolio_return
            and abs(portfolio_return) >= _MIN_PORTFOLIO_RETURN_FOR_DOMINANCE):
        for fund in fund_contribs:
            wr = fund.get("weighted_return_pct", 0.0)
            share = abs(wr / portfolio_return) * 100.0
            if share >= _DOMINANT_CONTRIBUTION_PCT:
                flags.append({
                    "flag": "dominant_contributor",
                    "detail": (
                        f"{fund['name']} contributes "
                        f"{share:.0f}% of portfolio return"
                    ),
                    "value": round(share, 1),
                })

    # Extreme concentration
    hhi = metrics.get("hhi")
    if hhi is not None and hhi >= _EXTREME_CONCENTRATION_HHI:
        flags.append({
            "flag": "high_concentration",
            "detail": f"HHI {hhi} indicates high weight concentration",
            "value": hhi,
        })

    # No rolling windows (insufficient data for distribution)
    wc = metrics.get("rolling_window_count")
    if wc is not None and wc == 0:
        flags.append({
            "flag": "no_rolling_data",
            "detail": "No rolling return windows computed; series may be too short",
            "value": 0,
        })

    return flags


# ── Main evaluation function ──────────────────────────────────

def evaluate_portfolio(
    insights: List[Dict[str, Any]],
    constraints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate portfolio insights against constraints.

    Args:
        insights: list of portfolio insight dicts (raw or compiled)
        constraints: user-defined thresholds (uses defaults for missing keys)

    Returns:
        Evaluation report with checks, flags, and summary.
    """
    # Merge user constraints over defaults
    effective = dict(DEFAULT_CONSTRAINTS)
    if constraints:
        effective.update(constraints)

    metrics = _extract_metrics(insights)
    checks = evaluate_constraints(metrics, effective)
    flags = detect_flags(metrics)

    # Summary counts
    passed = sum(1 for c in checks if c["status"] == "PASS")
    failed = sum(1 for c in checks if c["status"] == "FAIL")
    insufficient = sum(1 for c in checks if c["status"] == "INSUFFICIENT_DATA")

    return {
        "constraints_applied": effective,
        "checks": checks,
        "flags": flags,
        "summary": {
            "total_checks": len(checks),
            "passed": passed,
            "failed": failed,
            "insufficient_data": insufficient,
            "flag_count": len(flags),
            "verdict": "ALL_PASS" if failed == 0 and insufficient == 0
                       else "FAIL" if failed > 0
                       else "INCOMPLETE",
        },
        "extracted_metrics": metrics,
    }
