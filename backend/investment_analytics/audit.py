from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    previous_hash = _last_record_hash(path)
    payload_hash = _sha256(_canonical_json(sanitized_event))
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
    previous_hash = "0" * 64
    if not path.exists():
        return True
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            record_hash = record.get("current_hash") or record.get("record_hash")
            check = dict(record)
            check.pop("current_hash", None)
            check.pop("record_hash", None)
            if (record.get("prev_hash") or record.get("previous_hash")) != previous_hash:
                return False
            if _sha256(_canonical_json(check)) != record_hash:
                return False
            previous_hash = record_hash
    return True
