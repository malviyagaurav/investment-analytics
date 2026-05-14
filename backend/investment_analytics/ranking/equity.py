"""Equity ranking — 5-metric pairwise-dominance peer rank vs a benchmark.

For a given AMFI equity category, fetches all Direct Growth schemes,
aligns each to the category benchmark NAV, computes 5 metrics, and
ranks funds by counting how many peers each one dominates (wins ≥ 3
of 5 metrics).

Five metrics (signed so "higher is better" maps to wins where noted):
  1. excess_return_pct: fund CAGR − benchmark CAGR over aligned period
  2. max_drawdown_pct: peak-to-trough decline (negative number; less
     negative = better, captured as `a.max_drawdown_pct > b.max_drawdown_pct`)
  3. consistency_pct: % of rolling 1y windows where fund beat benchmark
  4. volatility_pct: annualized std-dev of daily returns (lower = better)
  5. downside_capture_ratio: avg fund return / avg benchmark return on
     days the benchmark was negative (lower = better)

Discipline:
- No composite score, no arbitrary weights. The 3-of-5 win rule is the
  whole opinion — every other number is observable.
- Survivorship bias acknowledged in _build_limitations: only currently-
  active funds appear in the registry, so closed/merged underperformers
  are absent.
- Benchmark fallback (Nifty 50 proxy for newer category indices) is
  surfaced as `benchmark_fallback=True` so callers can disclose it.

Dependency direction (acyclic):
  equity.py imports from:
    - stdlib (math, statistics, datetime, dataclasses, typing, pathlib)
    - data_discovery.fetch     (CATEGORY_BENCHMARK_MAP, DEFAULT_BENCHMARK,
                                fetch_scheme_nav — eager for the local
                                module binding; runtime resolution
                                inside rank_category goes through the
                                lazy parent-package seam — see below;
                                _convert_nav_to_records, _resolve_benchmark)
    - data_discovery.registry  (SchemeEntry, load_registry)
    - ranking._util            (MIN_ALIGNED_POINTS, ROLLING_*,
                                EXCLUDED_CATEGORIES,
                                BENCHMARK_FALLBACK_CATEGORIES,
                                _align_to_common_dates,
                                _annualized_return, _years_between,
                                _deduplicate_variants, _confidence_level)
  equity.py does NOT import from ranking __init__ or any sibling submodule.

Lazy-import seam (refactor-stability):
  Inside rank_category, fetch_scheme_nav is resolved via
  `from backend.investment_analytics import ranking as _rk` at call
  time, NOT bound at module load. Reason:
  test_ranking_orchestration.py mocks the function via
  `patch.object(rk, "fetch_scheme_nav", ...)`. An eager
  `from ... import fetch_scheme_nav` here would create a separate
  binding in equity.py that the patch does not reach. See
  feedback_refactor_lazy_import_seam.md.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import List, Optional, Tuple

from backend.data_discovery.fetch import (
    DEFAULT_BENCHMARK,
    _convert_nav_to_records,
    _resolve_benchmark,
)
from backend.data_discovery.registry import load_registry

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
    _years_between,
)


logger = logging.getLogger("investment_analytics.ranking.equity")


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

    # SEAM: resolve fetch_scheme_nav through the parent package so that
    # tests' patch.object(rk, "fetch_scheme_nav", ...) reach us. An
    # eager `from backend.data_discovery.fetch import fetch_scheme_nav`
    # at the top of this module would create a separate binding that
    # the patch does not touch. Refactor-stability test seam.
    from backend.investment_analytics import ranking as _rk

    # ── Resolve benchmark ──
    benchmark_fallback = False
    bench_code, bench_name = _resolve_benchmark(category)

    # Check if primary benchmark has enough history; fallback to Nifty 50 if needed
    if category in BENCHMARK_FALLBACK_CATEGORIES:
        try:
            bench_raw = _rk.fetch_scheme_nav(bench_code)
            bench_nav_data = bench_raw.get("data", [])
            bench_records = _convert_nav_to_records(bench_nav_data)
            if len(bench_records) < MIN_ALIGNED_POINTS:
                raise ValueError("Insufficient benchmark history")
        except Exception:
            # Fallback to Nifty 50
            bench_code, bench_name = DEFAULT_BENCHMARK
            benchmark_fallback = True
            bench_raw = _rk.fetch_scheme_nav(bench_code)
            bench_nav_data = bench_raw.get("data", [])
            bench_records = _convert_nav_to_records(bench_nav_data)
    else:
        bench_raw = _rk.fetch_scheme_nav(bench_code)
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
            raw = _rk.fetch_scheme_nav(scheme.scheme_code)
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
