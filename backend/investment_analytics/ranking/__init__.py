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


# ── Data structures ─────────────────────────────────────────────


@dataclass
class FundMetrics:
    """Computed metrics for a single fund."""
    scheme_code: int
    fund_name: str
    fund_house: str

    # 1. Excess return: CAGR(fund) - CAGR(benchmark) over full aligned period
    excess_return_pct: float
    # 2. Max drawdown (negative number)
    max_drawdown_pct: float
    # 3. Consistency: % of rolling windows where fund beat benchmark
    consistency_pct: float
    # 4. Volatility: annualized std dev of daily returns
    volatility_pct: float
    # 5. Downside capture: avg fund return on days benchmark was negative
    downside_capture_ratio: float

    # Metadata for display
    fund_cagr_pct: float
    benchmark_cagr_pct: float
    aligned_points: int
    history_years: float
    drawdown_trough_date: Optional[str] = None


@dataclass
class RankedFund:
    """A fund with its ranking result."""
    rank: int
    fund: FundMetrics
    dominance_count: int  # how many peers this fund dominates
    total_peers: int
    confidence_level: str = "Low"  # High (10y+), Medium (5-10y), Low (3-5y)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)


@dataclass
class ExcludedFund:
    """A fund excluded from ranking with reason."""
    scheme_code: int
    fund_name: str
    reason: str


@dataclass
class CategoryRanking:
    """Full ranking result for a category."""
    category: str
    benchmark_name: str
    benchmark_code: int
    benchmark_fallback: bool  # True if benchmark proxy was used
    ranked: List[RankedFund]
    excluded: List[ExcludedFund]
    computed_at: str
    total_funds_in_category: int


# ── Metric computation ──────────────────────────────────────────

# _align_to_common_dates, _annualized_return, _years_between moved
# to ._util; re-imported above.


def _compute_metrics(
    scheme_code: int,
    fund_name: str,
    fund_house: str,
    fund_records: List[dict],
    bench_records: List[dict],
) -> Optional[FundMetrics]:
    """Compute all 5 ranking metrics for a single fund.

    Returns None if insufficient data.
    """
    aligned = _align_to_common_dates(fund_records, bench_records)
    if len(aligned) < MIN_ALIGNED_POINTS:
        return None

    # ── 1. Excess return (full-period CAGR difference) ──
    first = aligned[0]
    last = aligned[-1]
    years = _years_between(first[0], last[0])
    if years < 0.5:
        return None

    fund_cagr = _annualized_return(first[1], last[1], years)
    bench_cagr = _annualized_return(first[2], last[2], years)
    excess_return = fund_cagr - bench_cagr

    # ── 2. Max drawdown ──
    peak = aligned[0][1]
    max_dd = 0.0
    trough_date = None
    for dt, fv, _ in aligned:
        if fv > peak:
            peak = fv
        dd = (fv - peak) / peak
        if dd < max_dd:
            max_dd = dd
            trough_date = dt.isoformat()

    # ── 3. Consistency (rolling hit ratio) ──
    wins = 0
    total_windows = 0
    for i in range(0, len(aligned) - ROLLING_WINDOW_DAYS, ROLLING_STEP_DAYS):
        j = i + ROLLING_WINDOW_DAYS
        if j >= len(aligned):
            break
        f_ret = (aligned[j][1] / aligned[i][1]) - 1.0
        b_ret = (aligned[j][2] / aligned[i][2]) - 1.0
        total_windows += 1
        if f_ret > b_ret:
            wins += 1
    consistency = (wins / total_windows * 100.0) if total_windows > 0 else 0.0

    # ── 4. Volatility (annualized std dev of daily returns) ──
    daily_returns = []
    for i in range(1, len(aligned)):
        prev_val = aligned[i - 1][1]
        if prev_val > 0:
            daily_returns.append((aligned[i][1] / prev_val) - 1.0)
    if len(daily_returns) < 20:
        return None
    vol = pstdev(daily_returns) * math.sqrt(252) * 100.0  # annualized %

    # ── 5. Downside capture ratio ──
    # On days the benchmark was down, what fraction of the decline did the fund capture?
    fund_down_returns = []
    bench_down_returns = []
    for i in range(1, len(aligned)):
        b_prev = aligned[i - 1][2]
        if b_prev <= 0:
            continue
        b_ret = (aligned[i][2] / b_prev) - 1.0
        if b_ret < 0:
            f_prev = aligned[i - 1][1]
            if f_prev > 0:
                f_ret = (aligned[i][1] / f_prev) - 1.0
                fund_down_returns.append(f_ret)
                bench_down_returns.append(b_ret)

    if bench_down_returns:
        avg_fund_down = mean(fund_down_returns)
        avg_bench_down = mean(bench_down_returns)
        # Ratio < 1.0 means fund loses less than benchmark on down days (good)
        downside_capture = avg_fund_down / avg_bench_down if avg_bench_down != 0 else 1.0
    else:
        downside_capture = 1.0

    return FundMetrics(
        scheme_code=scheme_code,
        fund_name=fund_name,
        fund_house=fund_house,
        excess_return_pct=round(fund_cagr * 100 - bench_cagr * 100, 2),
        max_drawdown_pct=round(max_dd * 100, 2),
        consistency_pct=round(consistency, 2),
        volatility_pct=round(vol, 2),
        downside_capture_ratio=round(downside_capture, 4),
        fund_cagr_pct=round(fund_cagr * 100, 2),
        benchmark_cagr_pct=round(bench_cagr * 100, 2),
        aligned_points=len(aligned),
        history_years=round(years, 1),
        drawdown_trough_date=trough_date,
    )


# ── Pairwise dominance ─────────────────────────────────────────


def _dominates(a: FundMetrics, b: FundMetrics) -> bool:
    """Fund A dominates Fund B if A wins ≥ 3 of 5 metrics.

    Win conditions (higher is better / lower is better):
    1. excess_return_pct: higher wins
    2. max_drawdown_pct: less negative wins (closer to 0)
    3. consistency_pct: higher wins
    4. volatility_pct: lower wins
    5. downside_capture_ratio: lower wins (captures less downside)
    """
    wins = 0
    if a.excess_return_pct > b.excess_return_pct:
        wins += 1
    if a.max_drawdown_pct > b.max_drawdown_pct:  # less negative = better
        wins += 1
    if a.consistency_pct > b.consistency_pct:
        wins += 1
    if a.volatility_pct < b.volatility_pct:
        wins += 1
    if a.downside_capture_ratio < b.downside_capture_ratio:
        wins += 1
    return wins >= 3


def _compute_dominance(funds: List[FundMetrics]) -> List[Tuple[FundMetrics, int]]:
    """For each fund, count how many peers it dominates.

    Returns list of (fund, dominance_count) sorted by dominance descending.
    Ties broken by: consistency (desc) → drawdown (less negative) → volatility (asc).
    """
    results: List[Tuple[FundMetrics, int]] = []
    for i, fund in enumerate(funds):
        count = sum(
            1 for j, other in enumerate(funds)
            if i != j and _dominates(fund, other)
        )
        results.append((fund, count))

    # Sort: dominance desc, then tie-breakers
    results.sort(key=lambda x: (
        -x[1],                      # dominance count descending
        -x[0].consistency_pct,      # tie-break 1: higher consistency
        -x[0].max_drawdown_pct,     # tie-break 2: less negative drawdown (higher value)
        x[0].volatility_pct,        # tie-break 3: lower volatility
    ))
    return results


# ── Strength / weakness labeling ────────────────────────────────


def _label_strengths_weaknesses(
    fund: FundMetrics,
    all_funds: List[FundMetrics],
) -> Tuple[List[str], List[str]]:
    """Generate factual strength/weakness bullets based on peer percentile."""
    strengths: List[str] = []
    weaknesses: List[str] = []
    n = len(all_funds)
    if n < 2:
        return strengths, weaknesses

    # Count how many peers this fund beats per metric
    excess_rank = sum(1 for f in all_funds if fund.excess_return_pct > f.excess_return_pct)
    dd_rank = sum(1 for f in all_funds if fund.max_drawdown_pct > f.max_drawdown_pct)
    cons_rank = sum(1 for f in all_funds if fund.consistency_pct > f.consistency_pct)
    vol_rank = sum(1 for f in all_funds if fund.volatility_pct < f.volatility_pct)
    down_rank = sum(1 for f in all_funds if fund.downside_capture_ratio < f.downside_capture_ratio)

    top_third = n * 2 / 3
    bottom_third = n / 3

    # Excess return
    if excess_rank >= top_third:
        strengths.append(f"Higher returns than {excess_rank}/{n} peers")
    elif excess_rank <= bottom_third:
        weaknesses.append(f"Lower returns than {n - excess_rank}/{n} peers")

    # Drawdown
    if dd_rank >= top_third:
        strengths.append(f"Shallower drawdowns than {dd_rank}/{n} peers")
    elif dd_rank <= bottom_third:
        weaknesses.append(f"Deeper drawdowns than {n - dd_rank}/{n} peers")

    # Consistency
    if cons_rank >= top_third:
        strengths.append(f"Beats benchmark more consistently than {cons_rank}/{n} peers")
    elif cons_rank <= bottom_third:
        weaknesses.append(f"Less consistent than {n - cons_rank}/{n} peers")

    # Volatility
    if vol_rank >= top_third:
        strengths.append(f"Lower volatility than {vol_rank}/{n} peers")
    elif vol_rank <= bottom_third:
        weaknesses.append(f"Higher volatility than {n - vol_rank}/{n} peers")

    # Downside capture
    if down_rank >= top_third:
        strengths.append(f"Better downside protection than {down_rank}/{n} peers")
    elif down_rank <= bottom_third:
        weaknesses.append(f"Weaker downside protection than {n - down_rank}/{n} peers")

    return strengths, weaknesses


# ── Deduplication + confidence level ────────────────────────────
# _VARIANT_SUFFIXES, _scheme_base_name, _deduplicate_variants,
# _confidence_level moved to ._util; re-imported above.


# ── Main entry point ────────────────────────────────────────────


def rank_category(
    category: str,
    registry_path: str,
) -> CategoryRanking:
    """Rank all Direct Growth funds in a given AMFI category.

    Steps:
    1. Load registry, filter to category + Direct + Growth
    2. Fetch benchmark NAV
    3. Fetch each fund NAV (cached)
    4. Compute 5 metrics per fund
    5. Pairwise dominance ranking
    6. Label strengths/weaknesses
    """
    if category in EXCLUDED_CATEGORIES:
        raise ValueError(
            f"Category '{category}' is excluded from ranking "
            "(too heterogeneous for meaningful peer comparison)"
        )

    # ── Load funds from registry ──
    from pathlib import Path
    registry = load_registry(Path(registry_path))
    category_funds_raw = [
        s for s in registry
        if s.scheme_category == category
        and "Direct" in s.scheme_name
        and "Growth" in s.scheme_name
    ]

    # Deduplicate: group by base name, prefer "Direct Plan - Growth" variant
    category_funds = _deduplicate_variants(category_funds_raw)
    total_in_category = len(category_funds)

    if total_in_category < 2:
        raise ValueError(
            f"Category '{category}' has fewer than 2 Direct Growth funds ({total_in_category})"
        )

    # ── Resolve benchmark ──
    benchmark_fallback = False
    bench_code, bench_name = _resolve_benchmark(category)

    # Check if primary benchmark has enough history; fallback to Nifty 50 if needed
    if category in BENCHMARK_FALLBACK_CATEGORIES:
        try:
            bench_raw = fetch_scheme_nav(bench_code)
            bench_nav_data = bench_raw.get("data", [])
            bench_records = _convert_nav_to_records(bench_nav_data)
            if len(bench_records) < MIN_ALIGNED_POINTS:
                raise ValueError("Insufficient benchmark history")
        except Exception:
            # Fallback to Nifty 50
            bench_code, bench_name = DEFAULT_BENCHMARK
            benchmark_fallback = True
            bench_raw = fetch_scheme_nav(bench_code)
            bench_nav_data = bench_raw.get("data", [])
            bench_records = _convert_nav_to_records(bench_nav_data)
    else:
        bench_raw = fetch_scheme_nav(bench_code)
        bench_nav_data = bench_raw.get("data", [])
        bench_records = _convert_nav_to_records(bench_nav_data)

    if len(bench_records) < MIN_ALIGNED_POINTS:
        raise ValueError(
            f"Benchmark {bench_name} (code {bench_code}) has insufficient data "
            f"({len(bench_records)} points, need {MIN_ALIGNED_POINTS})"
        )

    # ── Fetch + compute metrics for each fund ──
    computed: List[FundMetrics] = []
    excluded: List[ExcludedFund] = []

    for scheme in category_funds:
        # Skip if the fund IS the benchmark index fund
        if scheme.scheme_code == bench_code:
            excluded.append(ExcludedFund(
                scheme_code=scheme.scheme_code,
                fund_name=scheme.scheme_name,
                reason="Fund is the benchmark index fund itself",
            ))
            continue

        try:
            raw = fetch_scheme_nav(scheme.scheme_code)
            nav_data = raw.get("data", [])
            if not nav_data:
                excluded.append(ExcludedFund(
                    scheme_code=scheme.scheme_code,
                    fund_name=scheme.scheme_name,
                    reason="No NAV data available",
                ))
                continue

            fund_records = _convert_nav_to_records(nav_data)
            if len(fund_records) < MIN_ALIGNED_POINTS:
                excluded.append(ExcludedFund(
                    scheme_code=scheme.scheme_code,
                    fund_name=scheme.scheme_name,
                    reason=f"Insufficient data ({len(fund_records)} points, need {MIN_ALIGNED_POINTS})",
                ))
                continue

            metrics = _compute_metrics(
                scheme.scheme_code,
                scheme.scheme_name,
                scheme.fund_house,
                fund_records,
                bench_records,
            )
            if metrics is None:
                excluded.append(ExcludedFund(
                    scheme_code=scheme.scheme_code,
                    fund_name=scheme.scheme_name,
                    reason="Insufficient overlap with benchmark after alignment",
                ))
                continue

            computed.append(metrics)

        except Exception as exc:
            excluded.append(ExcludedFund(
                scheme_code=scheme.scheme_code,
                fund_name=scheme.scheme_name,
                reason=f"Fetch/compute error: {exc}",
            ))

    if len(computed) < 2:
        raise ValueError(
            f"Only {len(computed)} funds had sufficient data for ranking (need ≥ 2)"
        )

    # ── Pairwise dominance ──
    ranked_pairs = _compute_dominance(computed)

    # ── Build ranked results with strengths/weaknesses ──
    ranked_funds: List[RankedFund] = []
    for rank_pos, (fund, dom_count) in enumerate(ranked_pairs, start=1):
        strengths, weaknesses = _label_strengths_weaknesses(fund, computed)
        ranked_funds.append(RankedFund(
            rank=rank_pos,
            fund=fund,
            dominance_count=dom_count,
            total_peers=len(computed),
            confidence_level=_confidence_level(fund.history_years),
            strengths=strengths,
            weaknesses=weaknesses,
        ))

    return CategoryRanking(
        category=category,
        benchmark_name=bench_name,
        benchmark_code=bench_code,
        benchmark_fallback=benchmark_fallback,
        ranked=ranked_funds,
        excluded=excluded,
        computed_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        total_funds_in_category=total_in_category,
    )


def ranking_to_dict(result: CategoryRanking) -> dict:
    """Serialize CategoryRanking to JSON-safe dict for API response."""
    return {
        "category": result.category,
        "benchmark": {
            "name": result.benchmark_name,
            "scheme_code": result.benchmark_code,
            "fallback_used": result.benchmark_fallback,
        },
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
                "dominance": {
                    "beats": rf.dominance_count,
                    "of": rf.total_peers,
                },
                "metrics": {
                    "excess_return_pct": rf.fund.excess_return_pct,
                    "fund_cagr_pct": rf.fund.fund_cagr_pct,
                    "benchmark_cagr_pct": rf.fund.benchmark_cagr_pct,
                    "max_drawdown_pct": rf.fund.max_drawdown_pct,
                    "drawdown_trough_date": rf.fund.drawdown_trough_date,
                    "consistency_pct": rf.fund.consistency_pct,
                    "volatility_pct": rf.fund.volatility_pct,
                    "downside_capture_ratio": rf.fund.downside_capture_ratio,
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
            {
                "scheme_code": ef.scheme_code,
                "fund_name": ef.fund_name,
                "reason": ef.reason,
            }
            for ef in result.excluded
        ],
        "limitations": _build_limitations(result),
    }


def _build_limitations(result: CategoryRanking) -> List[str]:
    """Build factual limitation notes for the ranking."""
    lims = [
        "Rankings are based on historical data and do not predict future performance.",
        "Dominance counts how many peers a fund beats across 5 metrics — not a composite score.",
        "Results only include currently active funds. Funds closed or merged due to poor performance are excluded (survivorship bias).",
    ]
    if result.benchmark_fallback:
        lims.append(
            f"Category benchmark had insufficient history. "
            f"Nifty 50 index fund used as proxy benchmark. "
            f"Relative metrics may not reflect true category-specific performance."
        )
    if result.excluded:
        lims.append(
            f"{len(result.excluded)} fund(s) excluded due to insufficient data or fetch errors."
        )
    # Confidence warning: flag when most ranked funds have Low confidence
    low_count = sum(1 for rf in result.ranked if _confidence_level(rf.fund.history_years) == "Low")
    if low_count > len(result.ranked) * 0.5:
        lims.append(
            f"{low_count}/{len(result.ranked)} funds have limited benchmark overlap (<5 years). "
            f"Interpret rankings cautiously — short history may not capture full market cycles."
        )
    return lims


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


# ═══════════════════════════════════════════════════════════════
# DEBT FUND RANKING — absolute metrics, no benchmark
# ═══════════════════════════════════════════════════════════════

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


@dataclass
class DebtCategoryRanking:
    """Full ranking for a debt category."""
    category: str
    ranked: List[RankedFund]
    excluded: List[ExcludedFund]
    computed_at: str
    total_funds_in_category: int


def rank_debt_category(
    category: str,
    registry_path: str,
) -> DebtCategoryRanking:
    """Rank all Direct Growth debt funds in a category using pairwise dominance."""
    from pathlib import Path
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

    computed: List[DebtFundMetrics] = []
    excluded: List[ExcludedFund] = []

    for scheme in category_funds:
        try:
            raw = fetch_scheme_nav(scheme.scheme_code)
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


# ═══════════════════════════════════════════════════════════════
# GOLD FUND RANKING — FoF Direct Growth funds
# ═══════════════════════════════════════════════════════════════

GOLD_FOF_CATEGORY = "Other Scheme - FoF Domestic"


def rank_gold_funds(registry_path: str) -> DebtCategoryRanking:
    """Rank gold FoF funds using absolute metrics.

    Gold ETFs have no Direct/Growth variants and most are too new.
    Gold FoFs (investing in gold ETFs) have full NAV history.
    """
    from pathlib import Path
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

    computed: List[DebtFundMetrics] = []
    excluded: List[ExcludedFund] = []

    for scheme in gold_funds:
        try:
            raw = fetch_scheme_nav(scheme.scheme_code)
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


# ═══════════════════════════════════════════════════════════════
# FULL INVESTMENT VIEW — all assets ranked independently
# ═══════════════════════════════════════════════════════════════

DEFAULT_DEBT_CATEGORIES = [
    "Debt Scheme - Short Duration Fund",
    "Debt Scheme - Corporate Bond Fund",
    "Debt Scheme - Banking and PSU Fund",
    "Debt Scheme - Gilt Fund",
    "Debt Scheme - Liquid Fund",
]


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
