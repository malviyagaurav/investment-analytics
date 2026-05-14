"""Leaf utilities used by multiple portfolio_health submodules.

Kept dependency-free so it sits at the bottom of the package import
graph. Any future submodule that needs these helpers imports from
here, never from `portfolio_health.__init__`. That rule is what
keeps the package acyclic as the refactor progresses.

Discipline: only add a helper here if it is needed by at least TWO
submodules. Otherwise keep it co-located with its sole caller.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:  # pragma: no cover
    from backend.investment_analytics.portfolio_health import FundHealthResult  # noqa: F401


def _short_category(cat: str) -> str:
    """Shorten an AMFI category name for display.

    Strips "Equity Scheme - " / "Debt Scheme - " prefix and the
    trailing " Fund" suffix so UI labels read "Large Cap" instead of
    "Equity Scheme - Large Cap Fund". Pure string transform, no
    domain logic.
    """
    return (
        cat.replace("Equity Scheme - ", "")
           .replace("Debt Scheme - ", "")
           .replace(" Fund", "")
    )


def _resolve_weights(
    holdings: List["FundHealthResult"],
    weights: Dict[int, float],
) -> Dict[int, float]:
    """Resolve per-holding weights, normalising user-supplied weights
    and falling back to equal share when none are provided.

    Used by the priority builders (action + structural priority), the
    correlation pair enrichment, and the decision-summary serializer.
    Duck-typed on holdings — only `.scheme_code` is read, so the
    TYPE_CHECKING-only forward reference doesn't create a runtime
    dependency on the FundHealthResult dataclass.
    """
    n = len(holdings) or 1
    eq_weight = 1.0 / n
    held_codes = {h.scheme_code for h in holdings}
    held_weight_total = sum(weights.get(c, 0) for c in held_codes) or 0.0
    if held_weight_total > 0:
        return {c: weights.get(c, 0) / held_weight_total for c in held_codes}
    return {c: eq_weight for c in held_codes}
