"""Per-call NAV cache read tracking for evidence provenance.

Records which cache files were consumed during a single API call so
the resulting evidence record can include a stable
``cache_fingerprint`` — a sha256 over the sorted
``(scheme_code, mtime_ns, size_bytes)`` tuples of every cache file
that contributed.

## Wiring

  - ``cache.get_cached_nav`` calls ``record_cache_read`` after each
    successful cache hit (a no-op when no tracking session is
    active).
  - ``provenance.capture_provenance_inputs`` calls
    ``cache_fingerprint()`` to populate ``inputs.cache_fingerprint``.
  - The migrated API endpoints (ranking + portfolio-health) wrap
    their computation in ``start_tracking()`` / ``stop_tracking()``.

## Thread-locality

Two concurrent FastAPI handlers do NOT bleed reads into each other.
Uses ``threading.local`` so each request thread has its own session.

## Inactive semantics

When tracking is NOT active (no surrounding ``start_tracking``),
reads are no-ops and ``cache_fingerprint()`` returns ``None``. This
preserves CLI / ad-hoc / test usage where the caller doesn't care
about cache fingerprint capture. Explicit None lets callers
distinguish "no fingerprint" from "empty fingerprint" (the latter
is the sha256 of an empty list, a real value).

## Why mtime+size, not content hash

mtime_ns + size_bytes is effectively content-stable on the
single-machine deployment: cache files are written atomically and
never edited in place. A content hash would multiply the per-read
cost by file size. Escalate to content hash only if false-positive
"same fingerprint" cases ever appear.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Optional


_local = threading.local()


def start_tracking() -> None:
    """Begin a tracking session for the current thread."""
    _local.active = True
    _local.reads = {}


def stop_tracking() -> None:
    """End the tracking session for the current thread.

    Idempotent: calling without a prior start_tracking is safe (used
    in finally blocks). Clears all session state.
    """
    _local.active = False
    if hasattr(_local, "reads"):
        del _local.reads


def is_active() -> bool:
    return getattr(_local, "active", False)


def record_cache_read(path: Path, scheme_code: int) -> None:
    """Record a cache hit. No-op outside an active session.

    Reads mtime+size at record time so the fingerprint reflects what
    was actually consumed during this call, not what's on disk later.
    """
    if not is_active():
        return
    try:
        stat = path.stat()
        _local.reads[scheme_code] = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        # Cache file vanished between read and stat — skip; do not
        # poison the fingerprint with a sentinel.
        pass


def cache_fingerprint() -> Optional[str]:
    """Sha-256 over the canonical-JSON list of
    ``[{"scheme_code", "mtime_ns", "size_bytes"}, ...]`` sorted by
    scheme_code. Returns ``None`` when no tracking session is active.

    Order-independent: callers reading cache entries in any order
    produce the same fingerprint for the same set of (code, mtime,
    size) tuples.
    """
    if not is_active():
        return None
    reads = getattr(_local, "reads", {})
    canonical = json.dumps(
        [
            {"scheme_code": code, "mtime_ns": mt, "size_bytes": sz}
            for code, (mt, sz) in sorted(reads.items())
        ],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
