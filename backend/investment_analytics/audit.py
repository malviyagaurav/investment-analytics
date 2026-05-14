from __future__ import annotations

import fcntl
import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Two-layer locking on the read-prev-hash → write-record critical
# section so concurrent appenders cannot interleave and corrupt the
# hash chain:
#   1. _APPEND_LOCK (threading.Lock) — fast path for same-process
#      contention (e.g. concurrent FastAPI handlers on a single
#      uvicorn worker).
#   2. fcntl.flock on the open audit file — cross-process
#      serialization (FastAPI + cron-driven watchlist + ad-hoc CLI
#      all appending to the same log).
#
# fsync runs INSIDE the flock so the record is durable before any
# other writer is allowed to observe its hash. Releasing the lock
# pre-fsync would create a visibility/durability split: process B
# could read a hash whose backing record has not yet hit the disk.
#
# IMPORTANT — scope of the guarantee:
#   fcntl.flock serializes ONLY between cooperating local processes
#   on the same machine, on a local filesystem. It is NOT a
#   distributed lock. It is NOT safe across NFS / SMB / network
#   filesystems (the semantics there are unspecified or broken on
#   most kernels). Single-machine single-user is the supported
#   deployment per the architecture memo; do NOT generalize this
#   guarantee to multi-host setups.
_APPEND_LOCK = threading.Lock()

PII_KEYS = {
    "aadhaar",
    "account_number",
    "address",
    "client_id",
    "email",
    "full_name",
    "mobile",
    "name",
    "pan",
    "phone",
    "ssn",
    "user_id",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _last_record_hash(path: Path) -> str:
    if not path.exists():
        return "0" * 64
    last = ""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last = line
    if not last:
        return "0" * 64
    try:
        record = json.loads(last)
    except json.JSONDecodeError:
        return "invalid"
    return str(record.get("current_hash") or record.get("record_hash") or "invalid")


def sanitize_audit_event(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, child in value.items():
            if str(key).lower() in PII_KEYS:
                sanitized[key] = "[redacted]"
            else:
                sanitized[key] = sanitize_audit_event(child)
        return sanitized
    if isinstance(value, list):
        return [sanitize_audit_event(item) for item in value]
    return value


def append_audit_record(path: Path, event: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    sanitized_event = sanitize_audit_event(event)
    payload_hash = _sha256(_canonical_json(sanitized_event))
    # Critical section: thread-lock first (fast same-process path),
    # then fcntl.flock on the open handle for cross-process safety.
    # fsync is inside the flock — see module docstring.
    with _APPEND_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                previous_hash = _last_record_hash(path)
                record = {
                    "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                    "schema_version": "v1",
                    "analyzer_version": "mf_v2",
                    "prev_hash": previous_hash,
                    "payload_hash": payload_hash,
                    "event": sanitized_event,
                }
                record["current_hash"] = _sha256(_canonical_json(record))
                handle.write(_canonical_json(record) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return record


def hash_payload(value: Any) -> str:
    return _sha256(_canonical_json(sanitize_audit_event(value)))


def verify_audit_chain(path: Path) -> bool:
    return verify_audit_chain_diag(path)["valid"]


def verify_audit_chain_diag(path: Path) -> dict[str, Any]:
    """Verify chain integrity and return a diagnostic record.

    Returns a dict: {valid, lines_scanned, first_bad_line, reason, ...}.
    On a clean chain `first_bad_line` is None. On corruption, the first
    failing line is reported with the expected/observed hashes so the
    operator can pinpoint the breakage.
    """
    diag: dict[str, Any] = {
        "valid": True,
        "lines_scanned": 0,
        "first_bad_line": None,
        "reason": None,
    }
    previous_hash = "0" * 64
    if not path.exists():
        return diag
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            diag["lines_scanned"] = lineno
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                diag.update(valid=False, first_bad_line=lineno,
                            reason=f"json_decode_error: {exc}")
                return diag
            record_hash = record.get("current_hash") or record.get("record_hash")
            check = dict(record)
            check.pop("current_hash", None)
            check.pop("record_hash", None)
            got_prev = record.get("prev_hash") or record.get("previous_hash")
            if got_prev != previous_hash:
                diag.update(
                    valid=False, first_bad_line=lineno,
                    reason="prev_hash_mismatch",
                    expected_prev=previous_hash,
                    observed_prev=got_prev,
                    record_timestamp=record.get("timestamp"),
                )
                return diag
            recomputed = _sha256(_canonical_json(check))
            if recomputed != record_hash:
                diag.update(
                    valid=False, first_bad_line=lineno,
                    reason="body_hash_mismatch",
                    expected_hash=recomputed,
                    observed_hash=record_hash,
                    record_timestamp=record.get("timestamp"),
                )
                return diag
            previous_hash = record_hash
    return diag
