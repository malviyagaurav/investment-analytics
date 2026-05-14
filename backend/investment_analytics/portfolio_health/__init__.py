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
- __init__.py       — public surface; imports + re-exports only
- models.py         — public dataclasses (FundHealthResult etc.)
- _util.py          — shared leaf helpers (_short_category, _resolve_weights)
- coverage.py       — Coverage Integrity Layer
- alternatives.py   — alternative-selection gate + justification
- correlation.py    — hidden-overlap detection
- structural.py     — Regular vs Direct plan flag
- decision.py       — per-fund + portfolio-level decision engine
                      (check_portfolio_health orchestrator)
- serializer.py     — PortfolioHealthResult → API dict

Backward compatibility: every name defined or re-exported here was
present in the original portfolio_health.py monolith. External
callers and tests that do `from ...portfolio_health import X` or
`patch.object(ph, "X", ...)` continue to work unchanged.

The eager imports of `load_registry`, `fetch_scheme_nav`,
`_convert_nav_to_records`, `_get_or_rank_equity`, `_get_or_rank_debt`
are LOAD-BEARING for the test-mocking pattern used in
test_portfolio_health.py, test_correlation.py and
validate_decision_quality.py. Inner submodules resolve these names
via `from backend.investment_analytics import portfolio_health as
_ph; _ph.<name>(...)` at call time so that `patch.object(ph, "<name>",
...)` re-bindings flow through. Removing any of these imports would
silently bypass test patches. See feedback_refactor_lazy_import_seam.md.
"""
from __future__ import annotations

# Eager imports kept at package level so tests can patch the names via
# `patch.object(ph, "<name>", ...)` AND so submodules using the
# lazy-import seam can resolve them via `_ph.<name>` at call time.
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

# API-shape serializer + schema-version constant — extracted to
# .serializer. portfolio_health_to_dict is part of the public surface
# (api/main.py imports it directly).
from backend.investment_analytics.portfolio_health.serializer import (
    PORTFOLIO_HEALTH_SCHEMA_VERSION,
    _filter_top_ranked_by_coverage,
    portfolio_health_to_dict,
)
