"""Portfolio Health Check — evaluate user holdings against category peers.

For each fund the user holds, this module:
1. Identifies the fund's AMFI category
2. Runs or reuses the category ranking (equity or debt)
3. Finds where the fund places among its peers
4. Tags: Strong / Neutral / Weak
5. Assigns action: Continue / Monitor / Review
6. Lists higher-ranked alternatives with justification
7. Detects portfolio-level mistakes, redundancy, exposure gaps

No advisory language. No BUY/SELL. Factual peer comparison only.

Package layout (after Phase -1 refactor):
- __init__.py       — public surface; imports + re-exports + serializer
- models.py         — public dataclasses (FundHealthResult etc.)
- _util.py          — shared leaf helpers (_short_category, _resolve_weights)
- coverage.py       — Coverage Integrity Layer
- alternatives.py   — alternative-selection gate + justification
- correlation.py    — hidden-overlap detection
- structural.py     — Regular vs Direct plan flag
- decision.py       — per-fund + portfolio-level decision engine
                      (check_portfolio_health orchestrator)

Backward compatibility: every name defined or re-exported here was
present in the original portfolio_health.py monolith. External
callers and tests that do `from ...portfolio_health import X` or
`patch.object(ph, "X", ...)` continue to work unchanged.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

# Response schema bumped from v1 to v2 for Slice 1:
# - decision_summary entries gain weight_pct
# - alternatives.justification is now List[{reason, magnitude, metric}]
#   (was List[str])
# - alternatives must pass _alternative_is_material gate
PORTFOLIO_HEALTH_SCHEMA_VERSION = "v2"

# Eager imports kept at package level so tests can patch the names via
# `patch.object(ph, "<name>", ...)` AND so submodules that use the
# lazy-import seam can resolve them via `_ph.<name>` at call time. Do
# NOT remove — see feedback_refactor_lazy_import_seam.md.
from backend.data_discovery.fetch import fetch_scheme_nav, _convert_nav_to_records
from backend.data_discovery.registry import SchemeEntry, load_registry
from backend.investment_analytics.ranking import (
    ALL_RANKABLE_CATEGORIES,
    DEBT_CATEGORIES,
    DEBT_RISK_TAGS,
    DEFAULT_DEBT_CATEGORIES,
    EXCLUDED_CATEGORIES,
    RANKABLE_CATEGORIES,
    CategoryRanking,
    DebtCategoryRanking,
    RankedFund,
    _confidence_level,
    rank_category,
    rank_debt_category,
    ranking_to_dict,
    debt_ranking_to_dict,
)

logger = logging.getLogger("investment_analytics.portfolio_health")


# Shared leaf helpers — extracted to ._util so submodules can depend
# on them without forming a cycle through __init__.py.
from backend.investment_analytics.portfolio_health._util import (
    _resolve_weights,
    _short_category,
)

# Coverage Integrity Layer — extracted to .coverage submodule.
from backend.investment_analytics.portfolio_health.coverage import (
    COVERAGE_FULL_PCT,
    COVERAGE_PARTIAL_PCT,
    CoverageReport,
    _build_coverage_report,
)

# Alternative-selection gate, justification, metric-gap helpers
# — extracted to .alternatives submodule.
from backend.investment_analytics.portfolio_health.alternatives import (
    _ALT_THRESHOLDS,
    _DEBT_BULLETS,
    _EQUITY_BULLETS,
    _alternative_is_material,
    _build_alternatives,
    _build_justification,
    _build_your_fund_gaps,
    _improvement_magnitude,
    _metrics_for_display,
    _primary_metric_gap,
    _signed_delta,
)

# Hidden-overlap correlation — extracted to .correlation submodule.
from backend.investment_analytics.portfolio_health.correlation import (
    HIGH_CORRELATION_THRESHOLD,
    MIN_CORRELATION_DAYS,
    _compute_held_correlations,
    _correlation_pairs_from_nav,
    _enrich_correlations,
    _pearson,
)

# Structural priority (Regular vs Direct plan) — extracted to .structural.
from backend.investment_analytics.portfolio_health.structural import (
    _VARIANT_SUFFIXES_RE,
    _build_base_name_index,
    _build_structural_priority,
    _detect_regular_plan_holdings,
    _scheme_base_name,
)

# Public dataclass models — extracted to .models leaf so submodules
# can depend on them without forming a cycle through __init__.py.
from backend.investment_analytics.portfolio_health.models import (
    ConcentrationWarning,
    FundHealthResult,
    PortfolioHealthResult,
)

# Decision engine — orchestrator + per-fund/portfolio-level helpers.
# Extracted to .decision submodule. These names MUST remain reachable
# at `portfolio_health.<name>` so:
#   - external callers' imports continue to work
#   - tests' `patch.object(ph, "_get_or_rank_equity", ...)` and
#     `patch.object(ph, "load_registry", ...)` find the binding
#   - check_portfolio_health's lazy-seam re-resolution
#     (`_ph._get_or_rank_equity`) returns the patched value.
from backend.investment_analytics.portfolio_health.decision import (
    _MAJOR_DEBT_CATEGORIES,
    _MAJOR_EQUITY_CATEGORIES,
    _assign_action,
    _build_action_priority,
    _build_risk_summary,
    _collect_top_ranked_per_category,
    _data_quality_flags,
    _dedup_overlap_signals,
    _detect_exposure_gaps,
    _detect_mistakes,
    _detect_redundancy,
    _find_fund_in_ranking,
    _fund_status,
    _get_or_rank_debt,
    _get_or_rank_equity,
    _horizon_tag,
    _outlier_flags,
    _portfolio_status_label,
    _ranking_cache,
    check_portfolio_health,
)


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


# ── Serialization ──────────────────────────────────────────────

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
