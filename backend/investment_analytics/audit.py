from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Serializes the read-prev-hash → write-record critical section so
# concurrent FastAPI handlers in the same process cannot interleave
# writes and break the hash chain. Process-local: if you ever scale
# to multiple uvicorn workers you must replace this with an
# OS-level file lock or a dedicated audit writer service.
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
    # Read-prev-hash through to write must be atomic; otherwise two
    # callers can both observe the same prev_hash and corrupt the chain.
    with _APPEND_LOCK:
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
        with path.open("a", encoding="utf-8") as handle:
            handle.write(_canonical_json(record) + "\n")
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
