"""Multi-category equity ranking — top N per category, no cross-asset.

Wraps rank_category over a list of equity categories so the
front-end can render a "rank-1 in each style box" view in one
request. Categories are ranked INDEPENDENTLY — no composite score
across categories, no cross-asset comparison.

Two category lists are exposed:
- RANKABLE_CATEGORIES: categories with a dedicated tradable benchmark
  index that has long enough history to support full-fidelity peer
  ranking (Large Cap → Nifty 100 TRI, Mid Cap → Nifty Midcap 150,
  Small Cap → Nifty Smallcap 250, etc.).
- FALLBACK_CATEGORIES: categories whose nominal benchmark has
  insufficient tradable history; rank_category falls back to Nifty 50
  as a proxy and surfaces `benchmark_fallback=True`. These are still
  included in the union ALL_RANKABLE_CATEGORIES so callers can
  optionally rank them with explicit awareness of the proxy.

Discipline:
- A category with no benchmark proxy at all is in EXCLUDED_CATEGORIES
  (e.g., Sectoral/Thematic) and absent from both lists above.
- Errors per category are captured in `errors: Dict[category, str]`
  rather than raised, so one bad category never kills the multi run.

Dependency direction (acyclic):
  multi.py imports from:
    - stdlib (datetime, logging, dataclasses, typing)
    - ranking._util            (EXCLUDED_CATEGORIES)
    - ranking.equity           (CategoryRanking, rank_category,
                                ranking_to_dict)
  multi.py does NOT import from ranking __init__ or sibling submodules
  other than equity (a leaf relative to multi).

No test seam needed: tests do not mock rank_category via the parent
ranking namespace. The fetch-level lazy seam already lives inside
equity.rank_category, so any fetch_scheme_nav patch flows through
when rank_all_categories drives equity.rank_category.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

from backend.investment_analytics.ranking._util import EXCLUDED_CATEGORIES
from backend.investment_analytics.ranking.equity import (
    CategoryRanking,
    rank_category,
    ranking_to_dict,
)


logger = logging.getLogger("investment_analytics.ranking.multi")


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
