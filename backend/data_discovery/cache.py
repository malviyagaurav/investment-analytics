"""NAV data cache — avoids redundant MFAPI fetches.

Stores raw MFAPI JSON responses as {scheme_code}.json under data/cache/.
Each file includes a fetched_at timestamp.  Cache is valid for CACHE_TTL_HOURS.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("data_discovery.cache")

CACHE_TTL_HOURS = 24
CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cache"


def _cache_path(scheme_code: int) -> Path:
    return CACHE_DIR / f"{scheme_code}.json"


def _is_fresh(entry: dict) -> bool:
    fetched = entry.get("fetched_at")
    if not fetched:
        return False
    try:
        ts = datetime.fromisoformat(fetched)
    except (ValueError, TypeError):
        return False
    age_hours = (datetime.now(timezone.utc) - ts).total_seconds() / 3600
    return age_hours < CACHE_TTL_HOURS


def get_cached_nav(scheme_code: int) -> Optional[dict]:
    """Return cached MFAPI response if fresh, else None."""
    path = _cache_path(scheme_code)
    if not path.exists():
        return None
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not _is_fresh(entry):
        return None
    return entry.get("response")


def put_cached_nav(scheme_code: int, response: dict) -> None:
    """Write MFAPI response to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "fetched_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "scheme_code": scheme_code,
        "response": response,
    }
    _cache_path(scheme_code).write_text(
        json.dumps(entry), encoding="utf-8",
    )


def clear_cache() -> int:
    """Remove all cached files.  Returns count removed."""
    if not CACHE_DIR.exists():
        return 0
    count = 0
    for f in CACHE_DIR.glob("*.json"):
        f.unlink()
        count += 1
    return count
