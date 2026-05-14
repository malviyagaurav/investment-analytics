"""Category-level fund ranking engine using pairwise dominance.

Fetches all Direct Growth equity funds in a given AMFI category,
computes 5 metrics against the category benchmark, then ranks funds
by counting how many peers each fund dominates (wins ≥ 3 of 5 metrics).

No composite scores, no arbitrary weights, no advisory language.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from statistics import mean, median, pstdev
from typing import Any, Dict, List, Optional, Tuple

from backend.data_discovery.fetch import (
    CATEGORY_BENCHMARK_MAP,
    DEFAULT_BENCHMARK,
    fetch_scheme_nav,
    _convert_nav_to_records,
    _resolve_benchmark,
)
from backend.data_discovery.registry import SchemeEntry, load_registry

logger = logging.getLogger("investment_analytics.ranking")

# Leaf helpers + tunable constants — extracted to ._util submodule.
# Re-imported here so the existing module-internal references and any
# external `from backend.investment_analytics.ranking import X` keep
# working unchanged.
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


# Equity ranking — dataclasses, metric computation, pairwise dominance,
# rank_category orchestrator, ranking_to_dict serializer — extracted to
# .equity submodule. Re-imported here so internal references and any
# external `from backend.investment_analytics.ranking import X` keep
# working unchanged.
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


# ── Multi-category ranking ──────────────────────────────────────

# Categories that can be ranked with their dedicated benchmarks
RANKABLE_CATEGORIES = [
    "Equity Scheme - Large Cap Fund",
    "Equity Scheme - Mid Cap Fund",
    "Equity Scheme - Small Cap Fund",
    "Equity Scheme - Large & Mid Cap Fund",
    "Equity Scheme - Flexi Cap Fund",
    "Equity Scheme - Multi Cap Fund",
]

# Categories that work but use Nifty 50 fallback (lower confidence)
FALLBACK_CATEGORIES = [
    "Equity Scheme - Flexi Cap Fund",
    "Equity Scheme - Multi Cap Fund",
    "Equity Scheme - ELSS",
    "Equity Scheme - Value Fund",
    "Equity Scheme - Focused Fund",
    "Equity Scheme - Dividend Yield Fund",
    "Equity Scheme - Contra",
]

ALL_RANKABLE_CATEGORIES = RANKABLE_CATEGORIES + FALLBACK_CATEGORIES


@dataclass
class MultiCategoryRanking:
    """Result of ranking multiple categories."""
    categories: Dict[str, CategoryRanking]
    errors: Dict[str, str]  # category → error message
    computed_at: str
    top_n: int


def rank_all_categories(
    registry_path: str,
    top_n: int = 5,
    categories: Optional[List[str]] = None,
) -> MultiCategoryRanking:
    """Rank funds across multiple categories independently.

    Returns top N funds per category, each ranked within their own category.
    No cross-category comparison.
    """
    target_categories = categories or RANKABLE_CATEGORIES
    results: Dict[str, CategoryRanking] = {}
    errors: Dict[str, str] = {}

    for cat in target_categories:
        if cat in EXCLUDED_CATEGORIES:
            errors[cat] = "Category excluded (too heterogeneous)"
            continue
        try:
            result = rank_category(cat, registry_path)
            results[cat] = result
        except (ValueError, Exception) as exc:
            errors[cat] = str(exc)
            logger.warning("Failed to rank category %s: %s", cat, exc)

    return MultiCategoryRanking(
        categories=results,
        errors=errors,
        computed_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        top_n=top_n,
    )


def multi_ranking_to_dict(result: MultiCategoryRanking) -> dict:
    """Serialize MultiCategoryRanking for API response."""
    category_results = {}
    for cat, ranking in result.categories.items():
        full = ranking_to_dict(ranking)
        # Include only top N in the multi-category view
        full["ranked"] = full["ranked"][:result.top_n]
        full["showing_top_n"] = min(result.top_n, ranking_to_dict(ranking)["ranked_count"])
        category_results[cat] = full

    return {
        "computed_at": result.computed_at,
        "top_n": result.top_n,
        "categories_ranked": len(result.categories),
        "categories_failed": len(result.errors),
        "categories": category_results,
        "errors": result.errors,
    }


# Debt ranking — categories, risk tags, dataclasses, dominance,
# rank_debt_category orchestrator, debt_ranking_to_dict serializer —
# extracted to .debt submodule. Re-imported here so internal references
# and external `from backend.investment_analytics.ranking import X`
# keep working unchanged.
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


# Gold-fund ranking — GOLD_FOF_CATEGORY, rank_gold_funds,
# gold_ranking_to_dict — extracted to .gold submodule. Re-imported
# here so external imports continue to work unchanged.
from backend.investment_analytics.ranking.gold import (
    GOLD_FOF_CATEGORY,
    gold_ranking_to_dict,
    rank_gold_funds,
)


# ═══════════════════════════════════════════════════════════════
# FULL INVESTMENT VIEW — all assets ranked independently
# ═══════════════════════════════════════════════════════════════

# DEFAULT_DEBT_CATEGORIES moved to .debt; re-imported above.


@dataclass
class AllAssetsRanking:
    """Complete investment view across all asset classes."""
    equity: Dict[str, CategoryRanking]
    debt: Dict[str, DebtCategoryRanking]
    gold: Optional[DebtCategoryRanking]
    equity_errors: Dict[str, str]
    debt_errors: Dict[str, str]
    gold_error: Optional[str]
    computed_at: str
    top_n: int


def rank_all_assets(
    registry_path: str,
    top_n: int = 5,
    equity_categories: Optional[List[str]] = None,
    debt_categories: Optional[List[str]] = None,
) -> AllAssetsRanking:
    """Rank all asset classes independently. No cross-asset comparison."""
    eq_cats = equity_categories or RANKABLE_CATEGORIES
    dt_cats = debt_categories or DEFAULT_DEBT_CATEGORIES

    equity_results: Dict[str, CategoryRanking] = {}
    equity_errors: Dict[str, str] = {}
    for cat in eq_cats:
        if cat in EXCLUDED_CATEGORIES:
            equity_errors[cat] = "Excluded"
            continue
        try:
            equity_results[cat] = rank_category(cat, registry_path)
        except Exception as exc:
            equity_errors[cat] = str(exc)
            logger.warning("Equity rank failed for %s: %s", cat, exc)

    debt_results: Dict[str, DebtCategoryRanking] = {}
    debt_errors: Dict[str, str] = {}
    for cat in dt_cats:
        try:
            debt_results[cat] = rank_debt_category(cat, registry_path)
        except Exception as exc:
            debt_errors[cat] = str(exc)
            logger.warning("Debt rank failed for %s: %s", cat, exc)

    gold_result: Optional[DebtCategoryRanking] = None
    gold_error: Optional[str] = None
    try:
        gold_result = rank_gold_funds(registry_path)
    except Exception as exc:
        gold_error = str(exc)
        logger.warning("Gold rank failed: %s", exc)

    return AllAssetsRanking(
        equity=equity_results,
        debt=debt_results,
        gold=gold_result,
        equity_errors=equity_errors,
        debt_errors=debt_errors,
        gold_error=gold_error,
        computed_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        top_n=top_n,
    )


def all_assets_to_dict(result: AllAssetsRanking) -> dict:
    """Serialize full investment view."""
    eq = {}
    for cat, ranking in result.equity.items():
        full = ranking_to_dict(ranking)
        full["ranked"] = full["ranked"][:result.top_n]
        full["showing_top_n"] = min(result.top_n, len(ranking.ranked))
        eq[cat] = full

    dt = {}
    for cat, ranking in result.debt.items():
        full = debt_ranking_to_dict(ranking)
        full["ranked"] = full["ranked"][:result.top_n]
        full["showing_top_n"] = min(result.top_n, len(ranking.ranked))
        dt[cat] = full

    gold = None
    if result.gold:
        gold = gold_ranking_to_dict(result.gold)
        # Gold: show only #1 — all gold funds track the same underlying
        gold["ranked"] = gold["ranked"][:1]
        gold["showing_top_n"] = min(1, len(result.gold.ranked))

    return {
        "computed_at": result.computed_at,
        "top_n": result.top_n,
        "summary": _build_all_assets_summary(eq, dt, gold),
        "equity": {
            "categories": eq,
            "categories_ranked": len(result.equity),
            "errors": result.equity_errors,
        },
        "debt": {
            "categories": dt,
            "categories_ranked": len(result.debt),
            "errors": result.debt_errors,
        },
        "gold": gold,
        "gold_error": result.gold_error,
    }


def _build_all_assets_summary(
    equity: dict, debt: dict, gold: Optional[dict],
) -> dict:
    """Extract #1 pick from each ranked category for the summary block."""
    picks: dict = {"equity": {}, "debt": {}, "gold": None}

    # Short name mappings
    eq_names = {
        "Equity Scheme - Large Cap Fund": "Large Cap",
        "Equity Scheme - Mid Cap Fund": "Mid Cap",
        "Equity Scheme - Small Cap Fund": "Small Cap",
        "Equity Scheme - Large & Mid Cap Fund": "Large & Mid Cap",
        "Equity Scheme - Flexi Cap Fund": "Flexi Cap",
        "Equity Scheme - Multi Cap Fund": "Multi Cap",
    }
    dt_names = {
        "Debt Scheme - Short Duration Fund": "Short Duration",
        "Debt Scheme - Corporate Bond Fund": "Corporate Bond",
        "Debt Scheme - Banking and PSU Fund": "Banking & PSU",
        "Debt Scheme - Gilt Fund": "Gilt",
        "Debt Scheme - Liquid Fund": "Liquid",
    }

    for cat, catdata in equity.items():
        ranked = catdata.get("ranked", [])
        if ranked:
            top = ranked[0]
            # Exclude Low confidence from summary — unreliable for decisions
            if top["confidence_level"] == "Low":
                continue
            short = eq_names.get(cat, cat.replace("Equity Scheme - ", ""))
            picks["equity"][short] = {
                "fund_name": top["fund_name"],
                "fund_house": top["fund_house"],
                "dominance": top["dominance"],
                "confidence_level": top["confidence_level"],
            }

    for cat, catdata in debt.items():
        ranked = catdata.get("ranked", [])
        if ranked:
            top = ranked[0]
            if top["confidence_level"] == "Low":
                continue
            short = dt_names.get(cat, cat.replace("Debt Scheme - ", ""))
            picks["debt"][short] = {
                "fund_name": top["fund_name"],
                "fund_house": top["fund_house"],
                "dominance": top["dominance"],
                "confidence_level": top["confidence_level"],
            }

    if gold and gold.get("ranked"):
        top = gold["ranked"][0]
        if top["confidence_level"] != "Low":
            picks["gold"] = {
                "fund_name": top["fund_name"],
                "fund_house": top["fund_house"],
                "dominance": top["dominance"],
                "confidence_level": top["confidence_level"],
            }

    return picks
