"""Hidden-overlap detection across held funds.

Pairwise Pearson correlation of daily returns over the common-date
intersection. Surfaces pairs that move together even when their
category labels suggest they don't — closing the "I have 3 large-cap
funds, I'm diversified" blind spot.

Discipline:
- 0.85 threshold is calibrated for Indian equity active funds where
  same-category pairs typically run 0.93-0.97 and obviously-different
  holdings (equity vs gilt) run <0.3. 0.85 catches the cross-category
  cases where users mis-read category labels as diversification.
- MIN_CORRELATION_DAYS = 252 (~1 trading year) floor below which the
  correlation estimate is too noisy to publish.
- Pair list sorted by combined capital weight first, correlation
  second — a 50%/40% pair at ρ=0.86 is more decision-relevant than
  a 5%/3% pair at ρ=0.99.

Dependency direction (acyclic):
  correlation.py imports from:
    - stdlib (typing, logging)
    - portfolio_health._util  (_short_category, _resolve_weights)
    - data_discovery.fetch    (fetch_scheme_nav, _convert_nav_to_records)
    - ranking (forward-reference only — duck-typed on .scheme_code etc.)
  correlation.py does NOT import from __init__.py or any sibling
  submodule.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from backend.investment_analytics.portfolio_health._util import (
    _resolve_weights,
    _short_category,
)

if TYPE_CHECKING:  # pragma: no cover
    from backend.investment_analytics.portfolio_health import FundHealthResult  # noqa: F401


logger = logging.getLogger("investment_analytics.portfolio_health.correlation")


# Threshold for flagging a pair as "high overlap". Indian large-cap
# active funds typically run 0.93-0.97 against each other and against
# Nifty 100; cross-category large+flexi+multi often runs 0.90+.
# 0.85 picks up the cross-category cases where the "different category"
# label gives a false sense of diversification, without over-firing on
# obviously-different holdings (e.g., equity vs gilt fund typically <0.3).
HIGH_CORRELATION_THRESHOLD = 0.85

# Minimum aligned days needed for a correlation estimate to be meaningful.
# 252 = ~1 trading year. Below that, correlation is noisy.
MIN_CORRELATION_DAYS = 252


def _pearson(xs: List[float], ys: List[float]) -> float:
    """Pearson correlation. Returns 0.0 on degenerate inputs."""
    n = len(xs)
    if n != len(ys) or n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sxx * syy) ** 0.5
    if den <= 0:
        return 0.0
    return sxy / den


def _correlation_pairs_from_nav(
    nav_by_code: Dict[int, List[Dict[str, Any]]],
    threshold: float = HIGH_CORRELATION_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Pure function: given NAV records per scheme code, return high-
    correlation pairs.

    Aligns on common dates across ALL inputs (intersection — strict),
    computes per-fund daily returns, then pairwise Pearson correlation.
    Returns pairs whose correlation >= threshold, with the actual corr
    value and number of common days so the UI can disclose both.
    """
    if len(nav_by_code) < 2:
        return []
    nav_maps: Dict[int, Dict[str, float]] = {}
    for code, records in nav_by_code.items():
        if not records:
            continue
        nav_maps[code] = {r["date"]: r["nav"] for r in records if r.get("nav", 0) > 0}
    if len(nav_maps) < 2:
        return []
    common_dates_set = set.intersection(*(set(m.keys()) for m in nav_maps.values()))
    if len(common_dates_set) < MIN_CORRELATION_DAYS:
        return []
    sorted_dates = sorted(common_dates_set)
    returns_by_code: Dict[int, List[float]] = {}
    for code, m in nav_maps.items():
        rets: List[float] = []
        prev = m[sorted_dates[0]]
        for d in sorted_dates[1:]:
            curr = m[d]
            if prev > 0:
                rets.append((curr / prev) - 1.0)
            else:
                rets.append(0.0)
            prev = curr
        returns_by_code[code] = rets
    pairs: List[Dict[str, Any]] = []
    codes = list(returns_by_code.keys())
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            a, b = codes[i], codes[j]
            corr = _pearson(returns_by_code[a], returns_by_code[b])
            if corr >= threshold:
                pairs.append({
                    "fund_a_code": a,
                    "fund_b_code": b,
                    "correlation": round(corr, 3),
                    "common_days": len(sorted_dates),
                })
    pairs.sort(key=lambda p: -p["correlation"])
    return pairs


def _compute_held_correlations(
    scheme_codes: List[int],
    threshold: float = HIGH_CORRELATION_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Fetch cached NAV histories for the held funds and return the
    pairs whose return-correlation exceeds the threshold.

    Side-effect free except for cache warming via fetch_scheme_nav.
    Errors per fund are swallowed — partial results are still useful.

    fetch_scheme_nav and _convert_nav_to_records are resolved at
    CALL TIME via the parent package, not bound at module load. This
    keeps existing tests that patch
    `portfolio_health.fetch_scheme_nav` working unchanged across the
    refactor: a name re-bound in the package namespace is the value
    we end up calling.
    """
    if len(scheme_codes) < 2:
        return []
    # Lazy import — see docstring. Refactor-stability test seam.
    from backend.investment_analytics import portfolio_health as _ph
    nav_by_code: Dict[int, List[Dict[str, Any]]] = {}
    for code in scheme_codes:
        try:
            raw = _ph.fetch_scheme_nav(code)
            records = _ph._convert_nav_to_records(raw.get("data", []))
            if records:
                nav_by_code[code] = records
        except Exception as exc:
            logger.warning("Correlation fetch failed for scheme %s: %s", code, exc)
    return _correlation_pairs_from_nav(nav_by_code, threshold=threshold)


def _enrich_correlations(
    pairs: List[Dict[str, Any]],
    holdings: List["FundHealthResult"],
    weights: Optional[Dict[int, float]] = None,
) -> List[Dict[str, Any]]:
    """Attach fund names + categories AND per-fund + combined capital
    weight to each correlation pair so the UI can rank pairs by
    capital impact (a 50%/40% pair matters more than a 5%/3% pair
    even at identical ρ). Drops pairs whose codes are not present in
    `holdings` (defensive)."""
    by_code = {h.scheme_code: h for h in holdings}
    resolved = _resolve_weights(holdings, weights or {})
    eq_w = 1.0 / (len(holdings) or 1)

    def _w(code: int) -> float:
        return resolved.get(code, eq_w)

    enriched: List[Dict[str, Any]] = []
    for p in pairs:
        a = by_code.get(p["fund_a_code"])
        b = by_code.get(p["fund_b_code"])
        if not a or not b:
            continue
        a_w = round(_w(a.scheme_code) * 100, 1)
        b_w = round(_w(b.scheme_code) * 100, 1)
        combined_w = round(a_w + b_w, 1)
        enriched.append({
            "fund_a": {
                "scheme_code": a.scheme_code,
                "fund_name": a.fund_name,
                "category_short": _short_category(a.category),
                "weight_pct": a_w,
            },
            "fund_b": {
                "scheme_code": b.scheme_code,
                "fund_name": b.fund_name,
                "category_short": _short_category(b.category),
                "weight_pct": b_w,
            },
            "correlation": p["correlation"],
            "common_days": p["common_days"],
            "cross_category": a.category != b.category,
            "combined_weight_pct": combined_w,
            "message": (
                "Funds move together (correlation "
                + str(p["correlation"])
                + (" across different categories" if a.category != b.category
                   else " in the same category")
                + f"; combined {combined_w}% of portfolio"
                + ") — actual diversification is lower than category labels suggest"
            ),
        })
    # Sort by combined capital first, correlation second — a heavy
    # pair at 0.86 is more decision-relevant than a tiny pair at 0.99.
    enriched.sort(key=lambda x: (-x["combined_weight_pct"], -x["correlation"]))
    return enriched
