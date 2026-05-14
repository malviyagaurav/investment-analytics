"""Leaf helpers + tunable constants shared across ranking submodules.

Extracted so the equity/debt/gold/multi/all_assets submodules can
import these from one place without re-implementing or forming
circular dependencies through __init__.py. Pure functions only — any
dependency on registry, fetch, or fund-shaped dataclasses lives in
the caller, not here.

Dependency direction (acyclic):
  _util.py imports from:
    - stdlib (re, math is not needed here; date, typing)
    - data_discovery.registry  (SchemeEntry — used by _deduplicate_variants)
  _util.py does NOT import from ranking __init__ or any sibling submodule.

Stability:
- MIN_ALIGNED_POINTS, ROLLING_WINDOW_DAYS, ROLLING_STEP_DAYS are
  calibration constants. Bumping them changes which funds qualify
  and which windows count — treat as a calibration release, not a
  patch.
- _confidence_level thresholds (5y / 10y) are intentional: the
  Strong tag in portfolio_health.decision requires confidence ≠ Low,
  so the 5-year floor for Medium is load-bearing for the whole
  Strong/Neutral/Weak axis.
- BENCHMARK_FALLBACK_CATEGORIES + EXCLUDED_CATEGORIES are policy
  knobs consumed by rank_category and external callers
  (portfolio_health.decision imports EXCLUDED_CATEGORIES).

No test seam needed: no symbol defined here is patched by any
existing test.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Dict, List, Tuple

from backend.data_discovery.registry import SchemeEntry


# Minimum aligned data points to include a fund (roughly 3 years of trading)
MIN_ALIGNED_POINTS = 700

# Rolling window for consistency calculation (trading days ≈ 1 year)
ROLLING_WINDOW_DAYS = 252
ROLLING_STEP_DAYS = 5

# Categories that are too heterogeneous for meaningful peer comparison
EXCLUDED_CATEGORIES = frozenset({"Equity Scheme - Sectoral/ Thematic"})

# Categories where the primary benchmark has insufficient history
# and we fall back to Nifty 50 as proxy
BENCHMARK_FALLBACK_CATEGORIES = frozenset({
    "Equity Scheme - Large & Mid Cap Fund",
    "Equity Scheme - Multi Cap Fund",
    "Equity Scheme - Flexi Cap Fund",
    "Equity Scheme - ELSS",
    "Equity Scheme - Value Fund",
    "Equity Scheme - Focused Fund",
    "Equity Scheme - Contra",
})


def _align_to_common_dates(
    fund_records: List[dict],
    bench_records: List[dict],
) -> List[Tuple[date, float, float]]:
    """Align fund and benchmark records to common dates.

    Returns list of (date, fund_nav, bench_nav) tuples, sorted by date.
    """
    bench_map: Dict[str, float] = {r["date"]: r["nav"] for r in bench_records}
    aligned = []
    for r in fund_records:
        d = r["date"]
        if d in bench_map:
            aligned.append((
                date.fromisoformat(d),
                r["nav"],
                bench_map[d],
            ))
    aligned.sort(key=lambda x: x[0])
    return aligned


def _annualized_return(start_val: float, end_val: float, years: float) -> float:
    if years <= 0 or start_val <= 0:
        return 0.0
    return (end_val / start_val) ** (1.0 / years) - 1.0


def _years_between(start: date, end: date) -> float:
    return (end - start).days / 365.25


_VARIANT_SUFFIXES = re.compile(
    r"\s*-?\s*(?:Direct\s+Plan\s*-?\s*)?(?:Growth|Dividend\s+Reinvestment|Payout|IDCW\s+Reinvestment|IDCW\s+Payout).*$",
    re.IGNORECASE,
)


def _scheme_base_name(name: str) -> str:
    """Extract the base fund name by stripping plan/option suffixes."""
    return _VARIANT_SUFFIXES.sub("", name).strip()


def _deduplicate_variants(funds: List[SchemeEntry]) -> List[SchemeEntry]:
    """Group funds by base name, keep one canonical entry per fund.

    Priority: 'Direct Plan - Growth' > 'Direct Plan' > first seen.
    """
    groups: Dict[str, List[SchemeEntry]] = {}
    for f in funds:
        key = _scheme_base_name(f.scheme_name).lower()
        groups.setdefault(key, []).append(f)

    deduped: List[SchemeEntry] = []
    for _key, variants in groups.items():
        if len(variants) == 1:
            deduped.append(variants[0])
            continue
        # Prefer the "Direct Plan - Growth" variant (no Dividend/IDCW/Payout)
        preferred = [
            v for v in variants
            if "growth" in v.scheme_name.lower()
            and "dividend" not in v.scheme_name.lower()
            and "idcw" not in v.scheme_name.lower()
            and "payout" not in v.scheme_name.lower()
        ]
        deduped.append(preferred[0] if preferred else variants[0])
    return deduped


def _confidence_level(history_years: float) -> str:
    """Data confidence based on aligned history length.

    Load-bearing: portfolio_health.decision._fund_status requires
    confidence != Low to award the Strong tag, so the 5-year Medium
    floor is what separates Strong-eligible holdings from neutral
    ones regardless of peer-rank position.
    """
    if history_years >= 10.0:
        return "High"
    elif history_years >= 5.0:
        return "Medium"
    else:
        return "Low"
