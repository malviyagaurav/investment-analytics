"""Category-level fund ranking engine — public package surface.

Fetches all Direct Growth funds in an AMFI category, computes a 5-
metric signature, and ranks funds by pairwise dominance (a fund wins
the head-to-head if it beats the other on ≥ 3 of 5 metrics). No
composite score, no arbitrary weights, no advisory language.

Package layout (after Phase -1 refactor):
- __init__.py     — public surface; eager imports + submodule re-exports
- _util.py        — shared leaf helpers + tunable constants
- equity.py       — FundMetrics, CategoryRanking, rank_category, dominance
- debt.py         — DebtFundMetrics, DebtCategoryRanking, rank_debt_category
- gold.py         — GOLD_FOF_CATEGORY, rank_gold_funds (debt-style metrics)
- multi.py        — MultiCategoryRanking, rank_all_categories, category lists
- all_assets.py   — AllAssetsRanking, rank_all_assets (cross-asset overview)

Backward compatibility: every name re-exported here was present in
the original ranking.py monolith. External callers (api/main.py,
portfolio_health.decision, jobs.watchlist, tests) keep working
unchanged through `from backend.investment_analytics.ranking import X`
and `patch.object(rk, "X", ...)`.

The eager imports of fetch_scheme_nav and _convert_nav_to_records
are LOAD-BEARING: tests in tests/test_ranking_orchestration.py mock
the NAV fetch via `patch.object(rk, "fetch_scheme_nav", ...)`. The
entry-point orchestrators (rank_category, rank_debt_category,
rank_gold_funds) resolve fetch_scheme_nav via the parent package at
call time so those patches reach their call sites. Removing the
eager import here would silently bypass every test patch. See
feedback_refactor_lazy_import_seam.md.
"""
from __future__ import annotations

import logging

# Eager imports preserved at the package level for two reasons:
#   1. Tests reach attributes (CATEGORY_BENCHMARK_MAP, DEFAULT_BENCHMARK)
#      via `rk.<name>` for fixture construction.
#   2. Submodules using the lazy-import seam (equity.rank_category,
#      debt.rank_debt_category, gold.rank_gold_funds) resolve
#      fetch_scheme_nav via `_rk.fetch_scheme_nav` at call time so
#      `patch.object(rk, "fetch_scheme_nav", ...)` flows through.
from backend.data_discovery.fetch import (
    CATEGORY_BENCHMARK_MAP,
    DEFAULT_BENCHMARK,
    fetch_scheme_nav,
    _convert_nav_to_records,
    _resolve_benchmark,
)
from backend.data_discovery.registry import SchemeEntry, load_registry


logger = logging.getLogger("investment_analytics.ranking")


# Shared leaf helpers + tunable constants — .util submodule.
from backend.investment_analytics.ranking._util import (
    BENCHMARK_FALLBACK_CATEGORIES,
    EXCLUDED_CATEGORIES,
    MIN_ALIGNED_POINTS,
    ROLLING_STEP_DAYS,
    ROLLING_WINDOW_DAYS,
    _align_to_common_dates,
    _annualized_return,
    _confidence_level,
    _deduplicate_variants,
    _scheme_base_name,
    _VARIANT_SUFFIXES,
    _years_between,
)

# Equity ranking — 5-metric peer dominance vs benchmark.
from backend.investment_analytics.ranking.equity import (
    CategoryRanking,
    ExcludedFund,
    FundMetrics,
    RankedFund,
    _build_limitations,
    _compute_dominance,
    _compute_metrics,
    _dominates,
    _label_strengths_weaknesses,
    rank_category,
    ranking_to_dict,
)

# Debt ranking — absolute 5-metric peer dominance (no benchmark).
from backend.investment_analytics.ranking.debt import (
    DEBT_CATEGORIES,
    DEBT_RISK_TAGS,
    DEFAULT_DEBT_CATEGORIES,
    DebtCategoryRanking,
    DebtFundMetrics,
    _build_debt_limitations,
    _compute_debt_dominance,
    _compute_debt_metrics,
    _debt_dominates,
    _debt_strengths_weaknesses,
    debt_ranking_to_dict,
    rank_debt_category,
)

# Gold-fund ranking — FoF Direct Growth, debt-style metrics.
from backend.investment_analytics.ranking.gold import (
    GOLD_FOF_CATEGORY,
    gold_ranking_to_dict,
    rank_gold_funds,
)

# Multi-category equity overview — top N per category, no cross-category.
from backend.investment_analytics.ranking.multi import (
    ALL_RANKABLE_CATEGORIES,
    FALLBACK_CATEGORIES,
    MultiCategoryRanking,
    RANKABLE_CATEGORIES,
    multi_ranking_to_dict,
    rank_all_categories,
)

# Cross-asset overview — equity + debt + gold independently.
from backend.investment_analytics.ranking.all_assets import (
    AllAssetsRanking,
    _build_all_assets_summary,
    all_assets_to_dict,
    rank_all_assets,
)
