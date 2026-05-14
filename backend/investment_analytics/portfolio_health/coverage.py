"""Coverage Integrity Layer — capital-weighted analyzed coverage.

Computes how much of the portfolio's capital received a real peer
rank (versus falling into "Not Ranked" due to ETF/hybrid/sectoral/
insufficient-data paths). Picks a confidence band the UI uses to
suppress, tag, or pass through portfolio-level conclusions.

This is the most load-bearing anti-hallucination guardrail in the
system: when coverage is low, conclusions get suppressed rather
than fabricated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    # Forward reference only — avoids a circular import at module
    # load time. _build_coverage_report only reads .scheme_code and
    # .status off each holding (duck-typed), so no runtime dep.
    from backend.investment_analytics.portfolio_health import FundHealthResult  # noqa: F401


# Coverage-band thresholds. Below these capital-weighted ratios of
# successfully-peer-ranked holdings, portfolio-level conclusions
# (concentration, redundancy, exposure gaps, "Well diversified" label)
# carry uncertainty the user must see. We DO NOT block the conclusions
# at low coverage — blocking is unfriendly and removes the partial
# signal the user paid attention for. Instead we tag them.
COVERAGE_FULL_PCT = 70.0   # >= 70% capital analyzed → full confidence
COVERAGE_PARTIAL_PCT = 50.0  # 50-70% → partial; <50% → low


@dataclass
class CoverageReport:
    """How much of the portfolio's capital we can actually analyze.

    `analyzed_pct` is the capital-weighted share of holdings that
    received a real peer rank (status != "Not Ranked"). The
    `confidence_band` summarizes that for the UI:
      - "full":    >= COVERAGE_FULL_PCT (no banner needed)
      - "partial": 50..<70% (caveat banner)
      - "low":     < 50%  (prominent banner)
    Per-fund decisions are unaffected by coverage; only portfolio-level
    conclusions get the caveat.
    """
    total_holdings: int
    analyzed_holdings: int
    analyzed_pct: float           # 0..100, capital-weighted
    not_ranked_pct: float         # 0..100, capital-weighted
    confidence_band: str          # "full" | "partial" | "low"
    note: str                     # human-readable explanation; "" when full
    affected_metrics: List[str]   # names of portfolio-level outputs whose
                                  # confidence is degraded by low coverage


def _build_coverage_report(
    holdings: List["FundHealthResult"],
    weights: Dict[int, float],
) -> CoverageReport:
    """Compute capital-weighted analyzed coverage and pick a band."""
    total = len(holdings)
    if total == 0:
        return CoverageReport(
            total_holdings=0, analyzed_holdings=0,
            analyzed_pct=0.0, not_ranked_pct=0.0,
            confidence_band="full", note="", affected_metrics=[],
        )

    eq_weight = 1.0 / total
    held_codes = {h.scheme_code for h in holdings}
    held_weight_total = sum(weights.get(c, 0) for c in held_codes) or 0.0
    if held_weight_total > 0:
        resolved = {c: weights.get(c, 0) / held_weight_total for c in held_codes}
    else:
        resolved = {h.scheme_code: eq_weight for h in holdings}

    analyzed_weight = sum(
        resolved.get(h.scheme_code, eq_weight)
        for h in holdings
        if h.status != "Not Ranked"
    )
    analyzed_pct = round(analyzed_weight * 100, 1)
    not_ranked_pct = round((1.0 - analyzed_weight) * 100, 1)
    analyzed_holdings = sum(1 for h in holdings if h.status != "Not Ranked")

    affected = [
        "portfolio_status",
        "concentration",
        "redundancies",
        "exposure_gaps",
        "correlations",
        "no_major_issues",
    ]

    if analyzed_pct >= COVERAGE_FULL_PCT:
        return CoverageReport(
            total_holdings=total, analyzed_holdings=analyzed_holdings,
            analyzed_pct=analyzed_pct, not_ranked_pct=not_ranked_pct,
            confidence_band="full", note="", affected_metrics=[],
        )
    if analyzed_pct >= COVERAGE_PARTIAL_PCT:
        note = (
            f"Only {analyzed_pct}% of portfolio capital received peer analysis. "
            f"The remaining {not_ranked_pct}% (ETFs, hybrids, sectoral funds, "
            f"or holdings with insufficient data) cannot be peer-ranked. "
            f"Treat portfolio-level conclusions as approximate."
        )
        return CoverageReport(
            total_holdings=total, analyzed_holdings=analyzed_holdings,
            analyzed_pct=analyzed_pct, not_ranked_pct=not_ranked_pct,
            confidence_band="partial", note=note, affected_metrics=affected,
        )
    note = (
        f"Only {analyzed_pct}% of portfolio capital received peer analysis. "
        f"Portfolio-level conclusions (diversification, concentration, exposure "
        f"gaps, correlation) are based on a minority of capital and may be "
        f"misleading. Per-fund decisions on the analyzed subset remain valid."
    )
    return CoverageReport(
        total_holdings=total, analyzed_holdings=analyzed_holdings,
        analyzed_pct=analyzed_pct, not_ranked_pct=not_ranked_pct,
        confidence_band="low", note=note, affected_metrics=affected,
    )
