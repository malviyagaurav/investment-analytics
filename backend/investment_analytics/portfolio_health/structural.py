"""Structural priority — Regular vs Direct plan detection.

The discipline this module enforces: plan choice (Regular vs Direct)
is a STRUCTURAL fact about a holding, not a peer-rank verdict. A
rank-1 Strong fund on the Regular plan is still leaking ~1pp/yr in
expense-ratio drag. We surface this as a SEPARATE signal alongside
the Continue/Monitor/Review axis — never as a "verdict" that
collapses the two axes together.

Dependency direction (acyclic):
  structural.py imports from:
    - stdlib (typing, re)
    - portfolio_health._util  (_short_category, _resolve_weights)
    - data_discovery.registry (SchemeEntry — TYPE_CHECKING only)
  structural.py does NOT import from __init__.py or any sibling
  submodule.

No test seam needed: none of the names defined here are mocked by
existing tests. Pre-extraction grep confirmed.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from backend.investment_analytics.portfolio_health._util import (
    _resolve_weights,
    _short_category,
)

if TYPE_CHECKING:  # pragma: no cover
    from backend.data_discovery.registry import SchemeEntry  # noqa: F401
    from backend.investment_analytics.portfolio_health import FundHealthResult  # noqa: F401


_VARIANT_SUFFIXES_RE = re.compile(
    r"\s*-?\s*(?:Direct|Regular)\s*Plan\s*-?\s*(?:Growth|Dividend|IDCW|Payout|Reinvestment)?.*$",
    re.IGNORECASE,
)


def _scheme_base_name(name: str) -> str:
    """Strip "Direct/Regular Plan - Growth/Dividend/..." suffixes so
    two variants of the same scheme map to the same key. Used to
    find a held Regular plan's Direct sibling via base-name match."""
    return _VARIANT_SUFFIXES_RE.sub("", name).strip()


def _build_base_name_index(
    registry: List["SchemeEntry"],
) -> Dict[str, List["SchemeEntry"]]:
    """Group registry entries by base name (plan/option suffixes
    stripped) so Regular-plan-sibling lookup is O(1) per holding."""
    index: Dict[str, List["SchemeEntry"]] = {}
    for entry in registry:
        base = _scheme_base_name(entry.scheme_name).lower()
        index.setdefault(base, []).append(entry)
    return index


def _detect_regular_plan_holdings(
    holdings: List["FundHealthResult"],
    registry: List["SchemeEntry"],
) -> List[Dict[str, Any]]:
    """Detect held funds that are NOT Direct Plans and find the
    Direct sibling in the registry.

    Plan choice is structural: same fund, same portfolio, same
    returns before expenses — but Regular pays an intermediary
    commission baked into TER. Direct typically runs 0.5-1.5pp/yr
    lower.

    The flag carries the held scheme + the Direct sibling code if
    found. Phrased as observation, not action: "Plan choice is
    structural, distinct from peer ranking."

    Uses the registry's scheme_name (canonical AMFI label) — not
    FundHealthResult.fund_name, which can be a downstream-derived
    string and may not preserve the "Regular Plan" / "Direct Plan"
    suffix verbatim.
    """
    flags: List[Dict[str, Any]] = []
    by_code = {s.scheme_code: s for s in registry}
    by_base = _build_base_name_index(registry)
    for h in holdings:
        scheme = by_code.get(h.scheme_code)
        if scheme is None:
            continue
        name = scheme.scheme_name or ""
        if not name:
            continue
        if "Direct" in name:
            continue  # already on the efficient plan
        base = _scheme_base_name(name).lower()
        # Find the Direct sibling — same base name with "Direct" present.
        sibling = None
        for entry in by_base.get(base, []):
            if "Direct" in entry.scheme_name and "Growth" in entry.scheme_name:
                sibling = entry
                break
        flag = {
            "scheme_code": h.scheme_code,
            "fund_name": h.fund_name,
            "category_short": _short_category(h.category),
            "is_regular_plan": True,
            "direct_sibling": (
                None if sibling is None else {
                    "scheme_code": sibling.scheme_code,
                    "fund_name": sibling.scheme_name,
                }
            ),
            "message": (
                "This holding is the Regular plan. The Direct plan "
                "variant of the same scheme has a structurally lower "
                "expense ratio (typically 0.5-1.5 percentage points "
                "per year). Plan choice is structural, distinct from "
                "peer ranking."
            ),
        }
        flags.append(flag)
    return flags


def _build_structural_priority(
    plan_efficiency_flags: List[Dict[str, Any]],
    holdings: List["FundHealthResult"],
    weights: Dict[int, float],
) -> Optional[Dict[str, Any]]:
    """Pick the heaviest STRUCTURAL inefficiency (currently: Regular
    plan holdings). Distinct from action_priority because plan
    choice is not a peer-rank verdict — it's a fixed cost leak that
    exists even on a rank-1 fund.

    Returns None when no Regular plan holdings exist. The UI renders
    this alongside action_priority; whichever has more weight gets
    visual primacy. We deliberately do NOT collapse the two axes —
    "your top fund is on the wrong plan" and "your bottom fund is a
    peer-rank Review" are different problems that deserve different
    headlines.
    """
    if not plan_efficiency_flags:
        return None
    resolved = _resolve_weights(holdings, weights)
    eq_weight = 1.0 / (len(holdings) or 1)
    by_code = {h.scheme_code: h for h in holdings}
    candidates = []
    for f in plan_efficiency_flags:
        code = f["scheme_code"]
        h = by_code.get(code)
        if h is None:
            continue
        w = resolved.get(code, eq_weight)
        candidates.append((w, f, h))
    if not candidates:
        return None
    candidates.sort(key=lambda t: -t[0])
    w, flag, h = candidates[0]
    pct = round(w * 100, 1)
    return {
        "scheme_code": h.scheme_code,
        "fund_name": h.fund_name,
        "weight_pct": pct,
        "category_short": _short_category(h.category),
        "type": "regular_plan",
        "direct_sibling_code": (
            None if flag.get("direct_sibling") is None
            else flag["direct_sibling"]["scheme_code"]
        ),
        "headline": (
            f"Plan inefficiency: {h.fund_name} is on the Regular plan "
            f"({pct}% of portfolio). Direct plan variant has a structurally "
            f"lower expense ratio."
        ),
    }
