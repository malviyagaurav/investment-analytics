"""Search layer for AMFI scheme registry.

Provides case-insensitive search over scheme names and fund houses.
No ML, no ranking — just prefix + contains matching.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import List, Optional

from .registry import SchemeEntry

# Hard ceiling to prevent huge responses
MAX_RESULTS = 50


def search_schemes(
    registry: List[SchemeEntry],
    query: str,
    max_results: int = 20,
    category_filter: Optional[str] = None,
    fund_house_filter: Optional[str] = None,
) -> List[dict]:
    """Search registry by query string.

    Returns list of dicts (serializable), ordered:
      1. Exact scheme_code match (if query is numeric)
      2. Prefix matches on scheme_name
      3. Contains matches on scheme_name
      4. Contains matches on fund_house

    Filters:
      category_filter — substring match on scheme_category
      fund_house_filter — substring match on fund_house
    """
    query = query.strip()
    if not query:
        return []

    cap = min(max_results, MAX_RESULTS)
    q_lower = query.lower()

    # Pre-filter by category / fund_house if requested
    pool = registry
    if category_filter:
        cf = category_filter.lower()
        pool = [e for e in pool if cf in e.scheme_category.lower()]
    if fund_house_filter:
        fhf = fund_house_filter.lower()
        pool = [e for e in pool if fhf in e.fund_house.lower()]

    # 1. Exact scheme_code match
    exact_code: List[SchemeEntry] = []
    if query.isdigit():
        code = int(query)
        exact_code = [e for e in pool if e.scheme_code == code]

    # 2. Prefix match on scheme_name
    prefix_matches = [e for e in pool if e._name_lower.startswith(q_lower)]

    # 3. Contains match on scheme_name (excluding prefix matches)
    prefix_set = set(id(e) for e in prefix_matches)
    contains_matches = [
        e for e in pool
        if q_lower in e._name_lower and id(e) not in prefix_set
    ]

    # 4. Contains on fund_house (excluding already matched)
    already = prefix_set | set(id(e) for e in contains_matches) | set(id(e) for e in exact_code)
    fh_matches = [
        e for e in pool
        if q_lower in e.fund_house.lower() and id(e) not in already
    ]

    # Combine in priority order, cap results
    combined = exact_code + prefix_matches + contains_matches + fh_matches
    results = []
    for e in combined[:cap]:
        d = asdict(e)
        d.pop("_name_lower", None)
        results.append(d)

    return results
