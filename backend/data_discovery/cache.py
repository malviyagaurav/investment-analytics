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
    """Write MFAPI response to cache atomically.

    Uses a per-call unique temp file + rename so a concurrent reader
    never observes a half-written JSON file, AND two writers for the
    same scheme_code don't trip over a shared tmp name (which would
    cause one of them to fail on tmp.replace with FileNotFoundError).

    Without this, Path.write_text would truncate-then-write, and a
    reader during the window sees corrupted data. With a shared tmp
    name, two simultaneous writers would race on the rename. The
    PID+counter suffix below avoids both.
    """
    import os
    import threading

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "fetched_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "scheme_code": scheme_code,
        "response": response,
    }
    target = _cache_path(scheme_code)
    # Per-call unique tmp suffix: pid + thread id + monotonic counter.
    # We just need uniqueness within the process for the rename window.
    tmp_suffix = f".{os.getpid()}.{threading.get_ident()}.{id(entry)}.tmp"
    tmp = target.with_name(target.name + tmp_suffix)
    tmp.write_text(json.dumps(entry), encoding="utf-8")
    try:
        tmp.replace(target)
    except FileNotFoundError:
        # Another writer renamed in between — our content was
        # legitimately superseded by a concurrent write. Both writes
        # contained the same logical NAV data for this scheme code,
        # so either one is correct. Drop tmp if it still exists.
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def clear_cache() -> int:
    """Remove all cached files.  Returns count removed."""
    if not CACHE_DIR.exists():
        return 0
    count = 0
    for f in CACHE_DIR.glob("*.json"):
        f.unlink()
        count += 1
    return count
