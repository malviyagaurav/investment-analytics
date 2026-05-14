"""Debt ranking — absolute 5-metric pairwise dominance (no benchmark).

Indian debt fund benchmarks (CRISIL composite bond, IBI series) often
lack a tradable proxy with the same history as the funds, and credit
risk varies enough between funds that a single benchmark would mis-
represent dominance. So debt ranking uses ABSOLUTE metrics computed
directly from NAV — no benchmark alignment step.

Five debt metrics (signed so "higher is better" maps to wins):
  1. cagr_pct: annualized return over aligned period
  2. volatility_pct: annualized std-dev of daily returns (lower = better)
  3. max_drawdown_pct: peak-to-trough decline (less negative = better,
     captured as `a.max_drawdown_pct > b.max_drawdown_pct`)
  4. consistency_pct: % of rolling 1y windows with positive return
  5. risk_adj_return: cagr_pct / volatility_pct (higher = better; a
     debt-style Sharpe proxy, no risk-free subtraction since the
     comparison is intra-category and the rate cancels)

Notes:
- DEBT_RISK_TAGS is a STRUCTURAL label per AMFI category (credit +
  duration), not a per-fund computation. A Gilt Fund is "Medium" risk
  because long-duration gilts carry interest-rate risk regardless of
  the issuer's credit quality. Used downstream by the serializer.
- The RankedFund adapter (debt → equity FundMetrics shape) re-uses
  the same response schema as equity rankings so the API surface
  stays uniform. Field mapping: cagr_pct→excess_return_pct (and
  →fund_cagr_pct), risk_adj_return→downside_capture_ratio. The
  adapter is structural, not semantic — downstream code that asks for
  "excess_return_pct" on a debt fund gets the debt CAGR.

Dependency direction (acyclic):
  debt.py imports from:
    - stdlib (math, statistics, datetime, dataclasses, typing, pathlib)
    - data_discovery.fetch     (_convert_nav_to_records — eager; runtime
                                fetch_scheme_nav resolution goes through
                                the parent-package seam, see below)
    - data_discovery.registry  (load_registry)
    - ranking._util            (MIN_ALIGNED_POINTS, ROLLING_*,
                                _annualized_return, _years_between,
                                _deduplicate_variants, _confidence_level)
    - ranking.equity           (FundMetrics, RankedFund, ExcludedFund
                                — re-using the equity shapes for the
                                debt-side adapter so callers see one
                                schema)
  debt.py does NOT import from ranking __init__ or sibling submodules
  other than equity (which is itself a leaf relative to debt).

Lazy-import seam (refactor-stability):
  Inside rank_debt_category, fetch_scheme_nav is resolved via
  `from backend.investment_analytics import ranking as _rk` at call
  time. Same reasoning as equity.rank_category: tests that mock the
  function via `patch.object(rk, "fetch_scheme_nav", ...)` need the
  call site to re-resolve through the parent package on each call.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import pstdev
from typing import Dict, List, Optional, Tuple

from backend.data_discovery.fetch import _convert_nav_to_records
from backend.data_discovery.registry import load_registry

from backend.investment_analytics.ranking._util import (
    MIN_ALIGNED_POINTS,
    ROLLING_STEP_DAYS,
    ROLLING_WINDOW_DAYS,
    _annualized_return,
    _confidence_level,
    _deduplicate_variants,
    _years_between,
)
from backend.investment_analytics.ranking.equity import (
    ExcludedFund,
    FundMetrics,
    RankedFund,
)


logger = logging.getLogger("investment_analytics.ranking.debt")


# ── Category constants ──────────────────────────────────────────

DEBT_CATEGORIES = [
    "Debt Scheme - Liquid Fund",
    "Debt Scheme - Ultra Short Duration Fund",
    "Debt Scheme - Short Duration Fund",
    "Debt Scheme - Corporate Bond Fund",
    "Debt Scheme - Banking and PSU Fund",
    "Debt Scheme - Gilt Fund",
    "Debt Scheme - Dynamic Bond",
    "Debt Scheme - Money Market Fund",
    "Debt Scheme - Low Duration Fund",
    "Debt Scheme - Medium Duration Fund",
    "Debt Scheme - Overnight Fund",
    "Debt Scheme - Floater Fund",
    "Debt Scheme - Credit Risk Fund",
]

# Risk labels per debt category — structural, not computed from NAV
# These reflect inherent category-level risk (credit + duration), not individual fund risk.
DEBT_RISK_TAGS: Dict[str, str] = {
    "Debt Scheme - Liquid Fund": "Low",
    "Debt Scheme - Overnight Fund": "Low",
    "Debt Scheme - Money Market Fund": "Low",
    "Debt Scheme - Ultra Short Duration Fund": "Low",
    "Debt Scheme - Low Duration Fund": "Low",
    "Debt Scheme - Floater Fund": "Low",
    "Debt Scheme - Short Duration Fund": "Low-Medium",
    "Debt Scheme - Banking and PSU Fund": "Low-Medium",
    "Debt Scheme - Corporate Bond Fund": "Medium",
    "Debt Scheme - Gilt Fund": "Medium",           # Low credit risk, but high duration/interest-rate risk
    "Debt Scheme - Medium Duration Fund": "Medium",
    "Debt Scheme - Dynamic Bond": "Medium-High",
    "Debt Scheme - Credit Risk Fund": "High",
}

# Default set used by the cross-asset orchestrator when the caller
# doesn't specify a debt-category list explicitly.
DEFAULT_DEBT_CATEGORIES = [
    "Debt Scheme - Short Duration Fund",
    "Debt Scheme - Corporate Bond Fund",
    "Debt Scheme - Banking and PSU Fund",
    "Debt Scheme - Gilt Fund",
    "Debt Scheme - Liquid Fund",
]


# ── Data structures ─────────────────────────────────────────────


@dataclass
class DebtFundMetrics:
    """Metrics for a single debt fund — absolute, no benchmark needed."""
    scheme_code: int
    fund_name: str
    fund_house: str
    cagr_pct: float
    volatility_pct: float
    max_drawdown_pct: float
    consistency_pct: float
    risk_adj_return: float
    aligned_points: int
    history_years: float
    drawdown_trough_date: Optional[str] = None


@dataclass
class DebtCategoryRanking:
    """Full ranking for a debt category."""
    category: str
    ranked: List[RankedFund]
    excluded: List[ExcludedFund]
    computed_at: str
    total_funds_in_category: int


# ── Metric computation ──────────────────────────────────────────


def _compute_debt_metrics(
    scheme_code: int,
    fund_name: str,
    fund_house: str,
    fund_records: List[dict],
) -> Optional[DebtFundMetrics]:
    """Compute 5 debt-specific metrics (absolute, no benchmark)."""
    dated = []
    for r in fund_records:
        try:
            d = date.fromisoformat(r["date"])
            dated.append((d, r["nav"]))
        except (ValueError, KeyError):
            continue
    dated.sort(key=lambda x: x[0])

    if len(dated) < MIN_ALIGNED_POINTS:
        return None

    first_date, first_nav = dated[0]
    last_date, last_nav = dated[-1]
    years = _years_between(first_date, last_date)
    if years < 0.5 or first_nav <= 0:
        return None
    cagr = _annualized_return(first_nav, last_nav, years)

    peak = dated[0][1]
    max_dd = 0.0
    trough_date = None
    for dt, nav in dated:
        if nav > peak:
            peak = nav
        dd = (nav - peak) / peak
        if dd < max_dd:
            max_dd = dd
            trough_date = dt.isoformat()

    wins = 0
    total_windows = 0
    for i in range(0, len(dated) - ROLLING_WINDOW_DAYS, ROLLING_STEP_DAYS):
        j = i + ROLLING_WINDOW_DAYS
        if j >= len(dated):
            break
        ret = (dated[j][1] / dated[i][1]) - 1.0
        total_windows += 1
        if ret > 0:
            wins += 1
    consistency = (wins / total_windows * 100.0) if total_windows > 0 else 0.0

    daily_returns = []
    for i in range(1, len(dated)):
        prev = dated[i - 1][1]
        if prev > 0:
            daily_returns.append((dated[i][1] / prev) - 1.0)
    if len(daily_returns) < 20:
        return None
    vol = pstdev(daily_returns) * math.sqrt(252) * 100.0
    risk_adj = (cagr * 100.0) / vol if vol > 0 else 0.0

    return DebtFundMetrics(
        scheme_code=scheme_code,
        fund_name=fund_name,
        fund_house=fund_house,
        cagr_pct=round(cagr * 100, 2),
        volatility_pct=round(vol, 4),
        max_drawdown_pct=round(max_dd * 100, 2),
        consistency_pct=round(consistency, 2),
        risk_adj_return=round(risk_adj, 4),
        aligned_points=len(dated),
        history_years=round(years, 1),
        drawdown_trough_date=trough_date,
    )


# ── Pairwise dominance ──────────────────────────────────────────


def _debt_dominates(a: DebtFundMetrics, b: DebtFundMetrics) -> bool:
    """Fund A dominates Fund B if A wins >= 3 of 5 debt metrics."""
    wins = 0
    if a.cagr_pct > b.cagr_pct:
        wins += 1
    if a.volatility_pct < b.volatility_pct:
        wins += 1
    if a.max_drawdown_pct > b.max_drawdown_pct:
        wins += 1
    if a.consistency_pct > b.consistency_pct:
        wins += 1
    if a.risk_adj_return > b.risk_adj_return:
        wins += 1
    return wins >= 3


def _compute_debt_dominance(
    funds: List[DebtFundMetrics],
) -> List[Tuple[DebtFundMetrics, int]]:
    """Pairwise dominance for debt funds."""
    results = []
    for i, fund in enumerate(funds):
        count = sum(
            1 for j, other in enumerate(funds)
            if i != j and _debt_dominates(fund, other)
        )
        results.append((fund, count))
    results.sort(key=lambda x: (
        -x[1],
        -x[0].consistency_pct,
        x[0].volatility_pct,
        -x[0].max_drawdown_pct,
    ))
    return results


# ── Strength / weakness labeling ────────────────────────────────


def _debt_strengths_weaknesses(
    fund: DebtFundMetrics,
    all_funds: List[DebtFundMetrics],
) -> Tuple[List[str], List[str]]:
    """Factual strength/weakness bullets for debt fund."""
    strengths, weaknesses = [], []
    n = len(all_funds)
    if n < 2:
        return strengths, weaknesses

    ret_rank = sum(1 for f in all_funds if fund.cagr_pct > f.cagr_pct)
    vol_rank = sum(1 for f in all_funds if fund.volatility_pct < f.volatility_pct)
    dd_rank = sum(1 for f in all_funds if fund.max_drawdown_pct > f.max_drawdown_pct)
    cons_rank = sum(1 for f in all_funds if fund.consistency_pct > f.consistency_pct)
    ra_rank = sum(1 for f in all_funds if fund.risk_adj_return > f.risk_adj_return)

    top_third = n * 2 / 3
    bottom_third = n / 3

    if ret_rank >= top_third:
        strengths.append(f"Higher returns than {ret_rank}/{n} peers")
    elif ret_rank <= bottom_third:
        weaknesses.append(f"Lower returns than {n - ret_rank}/{n} peers")
    if vol_rank >= top_third:
        strengths.append(f"Lower volatility than {vol_rank}/{n} peers")
    elif vol_rank <= bottom_third:
        weaknesses.append(f"Higher volatility than {n - vol_rank}/{n} peers")
    if dd_rank >= top_third:
        strengths.append(f"Shallower drawdowns than {dd_rank}/{n} peers")
    elif dd_rank <= bottom_third:
        weaknesses.append(f"Deeper drawdowns than {n - dd_rank}/{n} peers")
    if cons_rank >= top_third:
        strengths.append(f"More consistently positive than {cons_rank}/{n} peers")
    elif cons_rank <= bottom_third:
        weaknesses.append(f"Less consistently positive than {n - cons_rank}/{n} peers")
    if ra_rank >= top_third:
        strengths.append(f"Better risk-adjusted return than {ra_rank}/{n} peers")
    elif ra_rank <= bottom_third:
        weaknesses.append(f"Weaker risk-adjusted return than {n - ra_rank}/{n} peers")

    return strengths, weaknesses


# ── Main entry point ────────────────────────────────────────────


def rank_debt_category(
    category: str,
    registry_path: str,
) -> DebtCategoryRanking:
    """Rank all Direct Growth debt funds in a category using pairwise dominance."""
    registry = load_registry(Path(registry_path))
    category_funds_raw = [
        s for s in registry
        if s.scheme_category == category
        and "Direct" in s.scheme_name
        and "Growth" in s.scheme_name
    ]
    category_funds = _deduplicate_variants(category_funds_raw)
    total_in_category = len(category_funds)

    if total_in_category < 2:
        raise ValueError(f"Debt category '{category}' has fewer than 2 Direct Growth funds")

    # SEAM: route fetch_scheme_nav through the parent package so tests'
    # patch.object(rk, "fetch_scheme_nav", ...) reach this call site.
    # Eager import would create a separate binding the patch misses.
    from backend.investment_analytics import ranking as _rk

    computed: List[DebtFundMetrics] = []
    excluded: List[ExcludedFund] = []

    for scheme in category_funds:
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
                excluded.append(ExcludedFund(scheme.scheme_code, scheme.scheme_name, "Insufficient data after computation"))
                continue
            computed.append(metrics)
        except Exception as exc:
            excluded.append(ExcludedFund(scheme.scheme_code, scheme.scheme_name, f"Fetch error: {exc}"))

    if len(computed) < 2:
        raise ValueError(f"Only {len(computed)} debt funds had sufficient data (need >= 2)")

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
        category=category,
        ranked=ranked_funds,
        excluded=excluded,
        computed_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        total_funds_in_category=total_in_category,
    )


# ── Serialization ───────────────────────────────────────────────


def debt_ranking_to_dict(result: DebtCategoryRanking) -> dict:
    """Serialize debt ranking for API response."""
    risk_tag = DEBT_RISK_TAGS.get(result.category, "Unknown")
    return {
        "category": result.category,
        "asset_class": "debt",
        "risk_tag": risk_tag,
        "benchmark": {"name": "None (absolute metrics)", "fallback_used": False},
        "computed_at": result.computed_at,
        "total_funds_in_category": result.total_funds_in_category,
        "ranked_count": len(result.ranked),
        "excluded_count": len(result.excluded),
        "ranked": [
            {
                "rank": rf.rank,
                "scheme_code": rf.fund.scheme_code,
                "fund_name": rf.fund.fund_name,
                "fund_house": rf.fund.fund_house,
                "dominance": {"beats": rf.dominance_count, "of": rf.total_peers},
                "metrics": {
                    "cagr_pct": rf.fund.fund_cagr_pct,
                    "volatility_pct": rf.fund.volatility_pct,
                    "max_drawdown_pct": rf.fund.max_drawdown_pct,
                    "consistency_pct": rf.fund.consistency_pct,
                    "risk_adj_return": rf.fund.downside_capture_ratio,
                },
                "history_years": rf.fund.history_years,
                "aligned_points": rf.fund.aligned_points,
                "confidence_level": rf.confidence_level,
                "top_in_category": rf.rank == 1,
                "strengths": rf.strengths,
                "weaknesses": rf.weaknesses,
            }
            for rf in result.ranked
        ],
        "excluded": [
            {"scheme_code": ef.scheme_code, "fund_name": ef.fund_name, "reason": ef.reason}
            for ef in result.excluded
        ],
        "limitations": _build_debt_limitations(result),
    }


def _build_debt_limitations(result: DebtCategoryRanking) -> List[str]:
    lims = [
        "Debt fund rankings use absolute metrics (no benchmark). NAV-based only.",
        "Volatility and drawdown serve as proxies for yield stability and credit risk.",
        "Actual yield-to-maturity, credit ratings, and duration are not available from NAV data.",
        "Results only include currently active funds. Funds closed or merged due to poor performance are excluded (survivorship bias).",
    ]
    if result.excluded:
        lims.append(f"{len(result.excluded)} fund(s) excluded due to insufficient data.")
    low_count = sum(1 for rf in result.ranked if rf.confidence_level == "Low")
    if low_count > len(result.ranked) * 0.5:
        lims.append(
            f"{low_count}/{len(result.ranked)} funds have limited history (<5 years). "
            f"Interpret cautiously."
        )
    return lims
