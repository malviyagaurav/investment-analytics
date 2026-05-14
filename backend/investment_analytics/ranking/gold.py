"""Gold-fund ranking — FoF Direct Growth, debt-metric framework reused.

Gold ETFs have no Direct/Growth variants on the Indian market and most
listed gold ETFs are too new to satisfy MIN_ALIGNED_POINTS. Gold FoFs
(funds-of-funds that invest in gold ETFs) DO carry full NAV history
through their AMCs, so we rank gold FoFs and treat them as the
investable proxy for gold exposure.

Mechanically identical to debt ranking:
- absolute (no benchmark) NAV-based metrics
- the same DebtFundMetrics shape, _compute_debt_metrics computation,
  _compute_debt_dominance pairwise rule and DebtCategoryRanking output
- gold_ranking_to_dict piggybacks debt_ranking_to_dict and overrides
  asset_class + limitations text

Why piggyback debt instead of duplicating equity-style code: all gold
FoFs in India track the same underlying spot price, so a benchmark-
relative metric set (excess return vs Nifty 50) would be meaningless.
Absolute CAGR, vol, drawdown, consistency and risk-adjusted return are
the appropriate axes — the same five the debt module uses.

Limitations surfaced to users:
- All gold funds track the same underlying; differences are tracking
  efficiency + expense ratio.
- Expense ratio is not directly available — return differences proxy.
- Gold ETFs are excluded for the reason above.

Dependency direction (acyclic):
  gold.py imports from:
    - stdlib (datetime, pathlib, typing)
    - data_discovery.fetch     (_convert_nav_to_records — eager;
                                fetch_scheme_nav goes through the
                                parent-package seam)
    - data_discovery.registry  (load_registry)
    - ranking._util            (MIN_ALIGNED_POINTS, _deduplicate_variants,
                                _confidence_level)
    - ranking.equity           (FundMetrics, RankedFund, ExcludedFund)
    - ranking.debt             (DebtFundMetrics, DebtCategoryRanking,
                                _compute_debt_metrics,
                                _compute_debt_dominance,
                                _debt_strengths_weaknesses,
                                debt_ranking_to_dict)
  gold.py does NOT import from ranking __init__ or any sibling submodule
  other than equity + debt (both leaf-relative to gold).

Lazy-import seam (refactor-stability):
  Inside rank_gold_funds, fetch_scheme_nav is resolved via
  `from backend.investment_analytics import ranking as _rk` at call
  time. Same reasoning as equity/debt: tests mock the function via
  `patch.object(rk, "fetch_scheme_nav", ...)` and an eager import here
  would create a separate binding the patch does not reach.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from backend.data_discovery.fetch import _convert_nav_to_records
from backend.data_discovery.registry import load_registry

from backend.investment_analytics.ranking._util import (
    MIN_ALIGNED_POINTS,
    _confidence_level,
    _deduplicate_variants,
)
from backend.investment_analytics.ranking.equity import (
    ExcludedFund,
    FundMetrics,
    RankedFund,
)
from backend.investment_analytics.ranking.debt import (
    DebtCategoryRanking,
    DebtFundMetrics,
    _compute_debt_dominance,
    _compute_debt_metrics,
    _debt_strengths_weaknesses,
    debt_ranking_to_dict,
)


logger = logging.getLogger("investment_analytics.ranking.gold")


GOLD_FOF_CATEGORY = "Other Scheme - FoF Domestic"


def rank_gold_funds(registry_path: str) -> DebtCategoryRanking:
    """Rank gold FoF funds using absolute metrics.

    Gold ETFs have no Direct/Growth variants and most are too new.
    Gold FoFs (investing in gold ETFs) have full NAV history.
    """
    registry = load_registry(Path(registry_path))
    gold_funds_raw = [
        s for s in registry
        if s.scheme_category == GOLD_FOF_CATEGORY
        and "Direct" in s.scheme_name
        and "Growth" in s.scheme_name
        and "gold" in s.scheme_name.lower()
    ]
    gold_funds = _deduplicate_variants(gold_funds_raw)
    total = len(gold_funds)

    if total < 2:
        raise ValueError(f"Only {total} gold FoF Direct Growth funds found (need >= 2)")

    # SEAM: route fetch_scheme_nav through the parent package so tests'
    # patch.object(rk, "fetch_scheme_nav", ...) reach this call site.
    from backend.investment_analytics import ranking as _rk

    computed: List[DebtFundMetrics] = []
    excluded: List[ExcludedFund] = []

    for scheme in gold_funds:
        try:
            raw = _rk.fetch_scheme_nav(scheme.scheme_code)
            nav_data = raw.get("data", [])
            if not nav_data:
                excluded.append(ExcludedFund(scheme.scheme_code, scheme.scheme_name, "No NAV data"))
                continue
            fund_records = _convert_nav_to_records(nav_data)
            if len(fund_records) < MIN_ALIGNED_POINTS:
                excluded.append(ExcludedFund(
                    scheme.scheme_code, scheme.scheme_name,
                    f"Insufficient data ({len(fund_records)} points, need {MIN_ALIGNED_POINTS})",
                ))
                continue
            metrics = _compute_debt_metrics(scheme.scheme_code, scheme.scheme_name, scheme.fund_house, fund_records)
            if metrics is None:
                excluded.append(ExcludedFund(scheme.scheme_code, scheme.scheme_name, "Insufficient data"))
                continue
            computed.append(metrics)
        except Exception as exc:
            excluded.append(ExcludedFund(scheme.scheme_code, scheme.scheme_name, f"Fetch error: {exc}"))

    if len(computed) < 2:
        raise ValueError(f"Only {len(computed)} gold funds had sufficient data (need >= 2)")

    ranked_pairs = _compute_debt_dominance(computed)

    ranked_funds: List[RankedFund] = []
    for rank_pos, (fund, dom_count) in enumerate(ranked_pairs, start=1):
        strengths, weaknesses = _debt_strengths_weaknesses(fund, computed)
        adapter = FundMetrics(
            scheme_code=fund.scheme_code,
            fund_name=fund.fund_name,
            fund_house=fund.fund_house,
            excess_return_pct=fund.cagr_pct,
            max_drawdown_pct=fund.max_drawdown_pct,
            consistency_pct=fund.consistency_pct,
            volatility_pct=fund.volatility_pct,
            downside_capture_ratio=fund.risk_adj_return,
            fund_cagr_pct=fund.cagr_pct,
            benchmark_cagr_pct=0.0,
            aligned_points=fund.aligned_points,
            history_years=fund.history_years,
            drawdown_trough_date=fund.drawdown_trough_date,
        )
        ranked_funds.append(RankedFund(
            rank=rank_pos,
            fund=adapter,
            dominance_count=dom_count,
            total_peers=len(computed),
            confidence_level=_confidence_level(fund.history_years),
            strengths=strengths,
            weaknesses=weaknesses,
        ))

    return DebtCategoryRanking(
        category="Gold Fund (FoF)",
        ranked=ranked_funds,
        excluded=excluded,
        computed_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        total_funds_in_category=total,
    )


def gold_ranking_to_dict(result: DebtCategoryRanking) -> dict:
    """Serialize gold ranking for API response."""
    d = debt_ranking_to_dict(result)
    d["asset_class"] = "gold"
    d["limitations"] = [
        "All gold funds track the same underlying (gold price). Differences are mainly tracking efficiency and expense ratio.",
        "Expense ratio is not directly available — return differences serve as proxy.",
        "Gold ETFs excluded (no Direct Growth variants; most too new for meaningful ranking).",
    ]
    if result.excluded:
        d["limitations"].append(f"{len(result.excluded)} fund(s) excluded due to insufficient data.")
    return d
