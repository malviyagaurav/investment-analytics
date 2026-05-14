"""Leaf utilities used by multiple portfolio_health submodules.

Kept dependency-free so it sits at the bottom of the package import
graph. Any future submodule that needs these helpers imports from
here, never from `portfolio_health.__init__`. That rule is what
keeps the package acyclic as the refactor progresses.

Discipline: only add a helper here if it is needed by at least TWO
submodules. Otherwise keep it co-located with its sole caller.
"""
from __future__ import annotations


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
