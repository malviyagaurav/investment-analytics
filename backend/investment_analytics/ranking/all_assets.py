"""Cross-asset overview — equity + debt + gold ranked independently.

The "full investment view" the front-end uses to render the homepage
summary. Asset classes are ranked SEPARATELY using their respective
modules — equity vs benchmark, debt absolute, gold absolute — and
the output is a packed view of "rank-1 in each held style box" plus
the per-category top-N tables.

No composite cross-asset score. Asking "is Large Cap rank-1 better
than Corporate Bond rank-1?" is not a meaningful question without a
risk-tolerance input, and we do not have or want a risk-tolerance
input here (the platform is non-advisory).

Discipline:
- Errors per category captured in `equity_errors` / `debt_errors` /
  `gold_error` strings rather than raised — one failing category does
  not poison the whole view.
- Summary picks (`_build_all_assets_summary`) deliberately SKIP any
  rank-1 fund whose `confidence_level == "Low"`: showing "this is the
  top large-cap" when the underlying ranking is data-thin is exactly
  the false-confidence pattern the Strong-tag downgrade was designed
  to prevent.
- Gold output is trimmed to ranked[:1]. All gold funds track the same
  underlying spot price, so the rank-2 / rank-3 gold pick adds no
  information.

Dependency direction (acyclic):
  all_assets.py imports from:
    - stdlib (datetime, logging, dataclasses, typing)
    - ranking._util            (EXCLUDED_CATEGORIES)
    - ranking.equity           (CategoryRanking, rank_category,
                                ranking_to_dict)
    - ranking.debt             (DebtCategoryRanking,
                                DEFAULT_DEBT_CATEGORIES,
                                rank_debt_category, debt_ranking_to_dict)
    - ranking.gold             (rank_gold_funds, gold_ranking_to_dict)
    - ranking.multi            (RANKABLE_CATEGORIES — equity default
                                category list)
  all_assets.py does NOT import from ranking __init__.

No test seam needed: tests do not mock the entry-point ranking
functions via the parent rk namespace. fetch_scheme_nav patches flow
through the lazy seam already in each leaf orchestrator.
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
from backend.investment_analytics.ranking.debt import (
    DEFAULT_DEBT_CATEGORIES,
    DebtCategoryRanking,
    debt_ranking_to_dict,
    rank_debt_category,
)
from backend.investment_analytics.ranking.gold import (
    gold_ranking_to_dict,
    rank_gold_funds,
)
from backend.investment_analytics.ranking.multi import RANKABLE_CATEGORIES


logger = logging.getLogger("investment_analytics.ranking.all_assets")


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
