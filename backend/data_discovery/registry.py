"""AMFI mutual fund scheme registry.

Parses NAVAll.txt from AMFI to build a local searchable registry of
Indian mutual fund schemes.  Each entry has:
  - scheme_code (int)
  - isin_growth (str or None)
  - isin_reinvestment (str or None)
  - scheme_name (str)
  - fund_house (str)
  - scheme_category (str)
  - latest_nav (float or None)
  - nav_date (str or None)
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("data_discovery.registry")

NAVALL_URL = "https://portal.amfiindia.com/spages/NAVAll.txt"


@dataclass(frozen=True)
class SchemeEntry:
    scheme_code: int
    scheme_name: str
    fund_house: str
    scheme_category: str
    isin_growth: Optional[str] = None
    isin_reinvestment: Optional[str] = None
    latest_nav: Optional[float] = None
    nav_date: Optional[str] = None

    # Precomputed lowercase name for fast search
    _name_lower: str = field(default="", repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_name_lower", self.scheme_name.lower())


_CATEGORY_RE = re.compile(
    r"^(Open Ended Schemes|Close Ended Schemes|Interval Fund Schemes)\((.+)\)\s*$"
)


def parse_navall_text(raw: str) -> List[SchemeEntry]:
    """Parse raw NAVAll.txt content into a list of SchemeEntry objects."""
    entries: list[SchemeEntry] = []
    current_fund_house = ""
    current_category = ""

    lines = raw.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        # Header line
        if stripped.startswith("Scheme Code;"):
            continue

        # Category line like "Open Ended Schemes(Debt Scheme - Banking and PSU Fund)"
        cat_match = _CATEGORY_RE.match(stripped)
        if cat_match:
            current_category = cat_match.group(2).strip()
            continue

        # Fund house line: non-empty line that doesn't contain ';'
        # and the previous non-empty line was blank or a category
        if ";" not in stripped:
            # This is either a fund house name or an unknown line
            # Fund house names are standalone lines without semicolons
            current_fund_house = stripped
            continue

        # Scheme data line: scheme_code;isin1;isin2;name;nav;date
        parts = stripped.split(";")
        if len(parts) < 6:
            continue

        try:
            scheme_code = int(parts[0].strip())
        except (ValueError, IndexError):
            continue

        isin_growth = parts[1].strip() or None
        if isin_growth == "-":
            isin_growth = None
        isin_reinvestment = parts[2].strip() or None
        if isin_reinvestment == "-":
            isin_reinvestment = None

        scheme_name = parts[3].strip()
        if not scheme_name:
            continue

        nav_str = parts[4].strip()
        try:
            latest_nav = float(nav_str)
        except (ValueError, TypeError):
            latest_nav = None

        nav_date = parts[5].strip() if len(parts) > 5 else None

        entries.append(SchemeEntry(
            scheme_code=scheme_code,
            scheme_name=scheme_name,
            fund_house=current_fund_house,
            scheme_category=current_category,
            isin_growth=isin_growth,
            isin_reinvestment=isin_reinvestment,
            latest_nav=latest_nav,
            nav_date=nav_date,
        ))

    logger.info("Parsed %d scheme entries from NAVAll.txt", len(entries))
    return entries


def save_registry(entries: List[SchemeEntry], path: Path) -> None:
    """Persist registry to JSON."""
    data = []
    for e in entries:
        d = asdict(e)
        d.pop("_name_lower", None)
        data.append(d)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("Saved %d entries to %s", len(data), path)


def load_registry(path: Path) -> List[SchemeEntry]:
    """Load registry from JSON file."""
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = []
    for d in raw:
        d.pop("_name_lower", None)
        entries.append(SchemeEntry(**d))
    return entries
