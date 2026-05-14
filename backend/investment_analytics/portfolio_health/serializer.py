"""Serializer — PortfolioHealthResult → API-shaped dict.

The single source of truth for the JSON shape returned to the
frontend and any external consumer. Versioned via
PORTFOLIO_HEALTH_SCHEMA_VERSION so that breaking shape changes are
visible to the UI without sniffing field names.

Discipline:
- Pure function over the already-computed result. No new computation,
  no decision logic, no advisory framing — anything that would
  classify, rank, or recommend lives upstream in `decision.py`.
- The `weights` parameter is the SAME normalized-or-equal map used
  by the orchestrator; passing it lets the serializer carry per-row
  weight_pct so the UI can sort decision_summary by capital impact
  rather than input order.
- Schema bumps are breaking changes. The version string travels in
  every response so callers can guard against drift.

Schema version history:
  v1 — initial release.
  v2 — current. decision_summary entries gain weight_pct;
       alternatives.justification became List[{reason, magnitude,
       metric}] (was List[str]); alternatives must pass the
       _alternative_is_material gate.

Dependency direction (acyclic):
  serializer.py imports from:
    - stdlib (typing)
    - portfolio_health._util         (_short_category)
    - portfolio_health.coverage      (CoverageReport — type only)
    - portfolio_health.correlation   (_enrich_correlations)
    - portfolio_health.models        (PortfolioHealthResult)
    - ranking                        (DEBT_RISK_TAGS)
  serializer.py does NOT import from __init__.py or .decision.

No test seam needed: no name in this module is patched by any
existing test (grep confirmed pre-extraction).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.investment_analytics.ranking import DEBT_RISK_TAGS

from backend.investment_analytics.portfolio_health._util import _short_category
from backend.investment_analytics.portfolio_health.coverage import CoverageReport
from backend.investment_analytics.portfolio_health.correlation import _enrich_correlations
from backend.investment_analytics.portfolio_health.models import PortfolioHealthResult


# Response schema version. Bumped from v1 to v2 for Slice 1:
# - decision_summary entries gain weight_pct
# - alternatives.justification is now List[{reason, magnitude, metric}]
#   (was List[str])
# - alternatives must pass _alternative_is_material gate
PORTFOLIO_HEALTH_SCHEMA_VERSION = "v2"


def _filter_top_ranked_by_coverage(
    rows: List[Dict[str, Any]],
    coverage: Optional[CoverageReport],
) -> List[Dict[str, Any]]:
    """Suppress top-ranked visibility entirely when coverage is low —
    declaring 'top-ranked in your categories' on the back of a minority
    of capital is the false-confidence pattern Item 1 was meant to stop.
    Tag with a partial-coverage note when band is partial."""
    if not rows:
        return []
    if coverage and coverage.confidence_band == "low":
        return []
    if coverage and coverage.confidence_band == "partial":
        return [
            {**r, "coverage_note": "based on partial portfolio coverage"}
            for r in rows
        ]
    return rows


def portfolio_health_to_dict(
    result: PortfolioHealthResult,
    weights: Optional[Dict[int, float]] = None,
) -> dict:
    """Serialize for API response.

    `weights` (scheme_code -> 0..1) lets the decision_summary carry
    weight_pct per entry so the frontend can prioritize Review items by
    actual capital impact instead of input order. If omitted, all
    weights default to equal share.
    """
    # Resolve weights — default to equal share if not provided.
    n_holdings = len(result.holdings) or 1
    eq_weight = 1.0 / n_holdings
    if weights:
        # Normalize to sum 1 across the holdings actually evaluated.
        held_codes = {h.scheme_code for h in result.holdings}
        held_weight_total = sum(weights.get(c, 0) for c in held_codes) or 0
        if held_weight_total > 0:
            resolved = {c: weights.get(c, 0) / held_weight_total for c in held_codes}
        else:
            resolved = {c: eq_weight for c in held_codes}
    else:
        resolved = {h.scheme_code: eq_weight for h in result.holdings}

    # Build decision summary (grouped by action) — entries carry
    # weight_pct so the UI can sort by capital impact.
    decision_summary: Dict[str, List[Dict[str, Any]]] = {
        "Continue": [],
        "Monitor": [],
        "Review": [],
    }
    decision_weight_pct: Dict[str, float] = {"Continue": 0.0, "Monitor": 0.0, "Review": 0.0}
    for h in result.holdings:
        w_pct = round(resolved.get(h.scheme_code, eq_weight) * 100, 1)
        entry = {
            "scheme_code": h.scheme_code,
            "fund_name": h.fund_name,
            "category_short": _short_category(h.category),
            "action_note": h.action_note,
            "weight_pct": w_pct,
        }
        bucket = h.action if h.action in decision_summary else "Monitor"
        decision_summary[bucket].append(entry)
        decision_weight_pct[bucket] = round(decision_weight_pct[bucket] + w_pct, 1)

    # Sort each bucket by weight_pct descending so highest-capital-
    # impact action shows first.
    for bucket in decision_summary:
        decision_summary[bucket].sort(key=lambda e: -e["weight_pct"])

    return {
        "computed_at": result.computed_at,
        "schema_version": PORTFOLIO_HEALTH_SCHEMA_VERSION,
        "total_holdings": len(result.holdings),
        "not_found_count": len(result.not_found),
        "decision_summary": decision_summary,
        "decision_summary_weight_pct": decision_weight_pct,
        "holdings": [
            {
                "scheme_code": h.scheme_code,
                "fund_name": h.fund_name,
                "fund_house": h.fund_house,
                "category": h.category,
                "category_short": _short_category(h.category),
                "asset_class": h.asset_class,
                "rank": h.rank,
                "total_in_category": h.total_in_category,
                "status": h.status,
                "action": h.action,
                "action_note": h.action_note,
                "confidence_level": h.confidence_level,
                "history_years": h.history_years,
                "benchmark_name": h.benchmark_name,
                "strengths": h.strengths,
                "weaknesses": h.weaknesses,
                "metrics": h.metrics,
                "alternatives": h.alternatives,
                "horizon": h.horizon,
                "risk_tag": DEBT_RISK_TAGS.get(h.category, None) if h.asset_class == "debt" else None,
                "data_quality_flags": h.data_quality_flags,
                "outlier_flags": h.outlier_flags,
                "your_fund_gaps": h.your_fund_gaps,
            }
            for h in result.holdings
        ],
        "not_found": result.not_found,
        "concentration": [
            {
                "category": c.category,
                "category_short": _short_category(c.category),
                "count": c.count,
                "weight_pct": c.weight_pct,
                "message": c.message,
            }
            for c in result.concentration
        ],
        "mistakes": result.mistakes,
        "redundancies": result.redundancies,
        "exposure_gaps": result.exposure_gaps,
        "correlations": _enrich_correlations(
            result.correlations, result.holdings, weights,
        ),
        "correlation_threshold": result.correlation_threshold,
        "coverage": (
            None if result.coverage is None else {
                "total_holdings": result.coverage.total_holdings,
                "analyzed_holdings": result.coverage.analyzed_holdings,
                "analyzed_pct": result.coverage.analyzed_pct,
                "not_ranked_pct": result.coverage.not_ranked_pct,
                "confidence_band": result.coverage.confidence_band,
                "note": result.coverage.note,
                "affected_metrics": result.coverage.affected_metrics,
            }
        ),
        "action_priority": result.action_priority,
        "structural_priority": result.structural_priority,
        "top_ranked_by_category": _filter_top_ranked_by_coverage(
            result.top_ranked_by_category, result.coverage,
        ),
        "plan_efficiency_flags": result.plan_efficiency_flags,
        "risk_summary": result.risk_summary,
        "portfolio_status": result.portfolio_status,
        "no_major_issues": (
            len(result.mistakes) == 0
            and len(result.redundancies) == 0
            and len(result.exposure_gaps) == 0
            and len(result.concentration) == 0
            and len(result.correlations) == 0
        ),
        "data_as_of": result.computed_at[:10],
        "limitations": [
            "Health check is based on historical peer ranking — not predictive.",
            "Only currently active funds are ranked (survivorship bias).",
            "Status reflects peer position, not absolute quality.",
            "Strong requires both top-25% rank and sufficient data history.",
            "Actions (Continue/Monitor/Review) reflect data signals, not financial advice.",
            "Alternatives shown for context — not recommendations.",
            "Potential overlap between holdings is not measured — actual diversification may differ.",
            "Redundancy detection uses metric similarity, not portfolio correlation.",
            "Exposure gaps are observations, not allocation guidance.",
            "Debt metrics are NAV-based. Credit quality and duration not captured.",
            "ETFs are not evaluated — NAV-based ranking does not apply to exchange-traded instruments.",
            "Horizon tags are estimated from historical volatility/drawdown — not a holding-period recommendation.",
        ],
    }
