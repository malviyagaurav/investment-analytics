"""Provenance inputs capture — code SHA, Python version, registry
hash, etc. — embedded in every audit event's envelope.

Why this lives outside ``audit.py``:
  - ``audit.py`` is the chain primitive (storage, hashing, locking).
    Anything that does process introspection (subprocess, git,
    sys.version) lives here so audit.py stays a tight leaf.
  - Future replay/drift/experiment tooling consumes this without
    importing the chain primitive.

Discipline:
  - Failures NEVER raise into audit.append_audit_record. Provenance
    capture is best-effort: if git is missing or the working dir
    isn't a repo, ``code_sha`` becomes ``"unknown"``. An audit
    record without provenance is preferable to an audit append
    that crashes.
  - ``_git_head_sha`` is cached for the process lifetime per
    architecture decision (production deploy = process restart;
    no SIGHUP or background refresh). Cache invalidation is NOT a
    bug — it's the design.
  - ``cache_fingerprint`` is intentionally ``None`` at this layer.
    Per-call cache instrumentation lands in step 4 alongside the
    by-reference evidence migration; capturing it earlier would
    require touching every fetch path right now.
"""
from __future__ import annotations

import functools
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Optional


_GIT_TIMEOUT_SEC = 5.0


@functools.lru_cache(maxsize=1)
def _git_head_sha() -> str:
    """Return the current git HEAD SHA (or ``<sha>-dirty`` when the
    working tree has uncommitted changes), or ``"unknown"`` if the
    command fails for any reason (not a git repo, git missing,
    timeout, permissions).

    Cached for the process lifetime via ``lru_cache``. The cache
    survives the entire process run; restarts re-read fresh. Do NOT
    add SIGHUP/invalidation logic — production deploy IS the
    invalidation event.
    """
    try:
        sha_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SEC,
            check=False,
        )
        sha = sha_result.stdout.strip()
        if not sha or sha_result.returncode != 0:
            return "unknown"
        dirty_result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SEC,
            check=False,
        )
        is_dirty = bool(dirty_result.stdout.strip())
        return f"{sha}-dirty" if is_dirty else sha
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _hash_file(path: Optional[Path]) -> Optional[str]:
    """SHA-256 of file bytes, or ``None`` for missing/None path.

    Chunked read so large registries don't load fully into memory.
    Returns None (not "unknown") for missing paths so callers can
    distinguish "no path supplied" from "path supplied but unreadable".
    Unreadable files raise — provenance capture treats that as an
    error worth surfacing.
    """
    if path is None or not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def capture_provenance_inputs(
    registry_path: Optional[Path] = None,
) -> dict[str, Optional[str]]:
    """Snapshot of the inputs that determined an evidence emission.

    Always returns the same key set so audit records have a stable
    shape regardless of caller context:

      - ``code_sha``         — git HEAD SHA (cached); ``"unknown"`` if not in a repo.
      - ``python_version``   — major.minor.micro (e.g. "3.9.6").
      - ``analyzer_version`` — top-level analyzer identifier ("mf_v2" today).
      - ``registry_hash``    — sha256 of the registry file consumed, or null if no path supplied.
      - ``registry_path``    — absolute path string, or null if no path supplied.
      - ``cache_fingerprint``— null at this layer; populated in step 4.

    Callers that have a registry path pass it; callers that don't
    (most non-ranking events) get null for the registry fields.
    """
    return {
        "code_sha":          _git_head_sha(),
        "python_version":    ".".join(str(p) for p in sys.version_info[:3]),
        "analyzer_version":  "mf_v2",
        "registry_hash":     _hash_file(registry_path) if registry_path else None,
        "registry_path":     str(registry_path) if registry_path else None,
        "cache_fingerprint": None,
    }
