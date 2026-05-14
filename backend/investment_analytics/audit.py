from __future__ import annotations

import fcntl
import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from backend.investment_analytics.evidence_envelope import build_event_envelope

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


def _active_epoch(audit_dir: Path) -> int:
    """Return the currently-open epoch number from epochs.json.

    Reads OUTSIDE the audit-file flock — epoch changes are rare,
    appends are frequent, and rotations themselves serialize against
    appends via the audit-file lock. Cost: one small JSON read per
    append; negligible against the existing fsync. NOT cached: a
    concurrent rotation must be observable on the next append.

    Returns 1 when epochs.json is absent (pre-migration behaviour),
    unreadable, or contains no open epoch — keeping the
    chain_epoch field meaningful even before formal migration.
    """
    epochs_path = audit_dir / "epochs.json"
    if not epochs_path.exists():
        return 1
    try:
        index = json.loads(epochs_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 1
    for epoch in reversed(index.get("epochs", [])):
        if epoch.get("status") == "open":
            return int(epoch["epoch"])
    return 1


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


def append_audit_record(
    path: Path,
    event: dict[str, Any],
    evidence_kind: Optional[str] = None,
    run_id: Optional[str] = None,
    parent_run_id: Optional[str] = None,
    inputs: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Append a hash-chained audit record.

    Backward compatible: callers that pass only ``path`` + ``event``
    continue to work; the envelope is auto-added with
    ``evidence_kind=None`` (intentionally unclassified). New call
    sites opt in to ``evidence_kind`` and lineage via the optional
    kwargs.

    The envelope is built BEFORE sanitization so PII in provenance
    fields (e.g. an inputs dict containing user_id / email) is still
    redacted. The hash covers the sanitized envelope, so what's on
    disk and what's hashed agree byte-for-byte.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Envelope first, sanitize after — provenance fields may themselves
    # contain sensitive data and must pass through the same redaction
    # path as caller-supplied event fields.
    enveloped_event = build_event_envelope(
        event,
        evidence_kind=evidence_kind,
        run_id=run_id,
        parent_run_id=parent_run_id,
        inputs=inputs,
    )
    sanitized_event = sanitize_audit_event(enveloped_event)
    payload_hash = _sha256(_canonical_json(sanitized_event))
    # Resolve the active epoch BEFORE the lock — epoch changes are
    # rare and rotations serialize via the same lock we are about to
    # take, so reading epochs.json here is safe.
    epoch = _active_epoch(path.parent)
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
                    "chain_epoch": epoch,
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


def verify_audit_chain_multi(audit_dir: Path) -> dict[str, Any]:
    """Aggregate verification across all epochs in epochs.json plus
    documented orphan chains. Returns a TYPED overall_status so
    monitoring systems can distinguish full health from partial
    corruption without inferring it from buried warnings.

    overall_status values:
      - "valid"           — every registered epoch verified clean.
      - "partial_failure" — at least one epoch valid AND at least one failed.
      - "invalid"         — every registered epoch failed verification.
      - "empty"           — no epochs.json and no live audit.jsonl on disk.
      - "unverifiable"    — epochs.json present but malformed.

    Pre-migration behaviour (no epochs.json): falls back to single-
    file verification of audit_dir/audit.jsonl and reports a one-
    element per_epoch array. Backward-compatible with the existing
    /health surface that calls verify_audit_chain on a single file.

    Orphan chains are documented but NOT re-verified — their
    classification is recorded at migration time (when the chain
    structure was inspected) and is treated as immutable. Echoing
    the recorded classification here keeps the aggregate result
    informative without rescanning every orphan on every health
    poll.
    """
    epochs_path = audit_dir / "epochs.json"
    if not epochs_path.exists():
        live = audit_dir / "audit.jsonl"
        if not live.exists():
            return {
                "overall_status": "empty",
                "epochs_total": 0,
                "epochs_valid": 0,
                "epochs_failed": 0,
                "orphan_chains_total": 0,
                "per_epoch": [],
                "per_orphan": [],
            }
        single = verify_audit_chain_diag(live)
        status = "valid" if single["valid"] else "invalid"
        return {
            "overall_status": status,
            "epochs_total": 1,
            "epochs_valid": 1 if single["valid"] else 0,
            "epochs_failed": 0 if single["valid"] else 1,
            "orphan_chains_total": 0,
            "per_epoch": [{
                "epoch": 1,
                "file": "audit.jsonl",
                "status": status,
                "lines_scanned": single["lines_scanned"],
                "first_bad_line": single["first_bad_line"],
                "reason": single["reason"],
            }],
            "per_orphan": [],
        }

    try:
        index = json.loads(epochs_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return {
            "overall_status": "unverifiable",
            "epochs_total": 0,
            "epochs_valid": 0,
            "epochs_failed": 0,
            "orphan_chains_total": 0,
            "per_epoch": [],
            "per_orphan": [],
            "reason": f"epochs.json could not be parsed: {exc}",
        }

    per_epoch = []
    for epoch_entry in index.get("epochs", []):
        epoch_file = audit_dir / epoch_entry["file"]
        if not epoch_file.exists():
            per_epoch.append({
                "epoch": epoch_entry["epoch"],
                "file": epoch_entry["file"],
                "status": "missing",
                "lines_scanned": 0,
                "first_bad_line": None,
                "reason": "file not found",
            })
            continue
        diag = verify_audit_chain_diag(epoch_file)
        per_epoch.append({
            "epoch": epoch_entry["epoch"],
            "file": epoch_entry["file"],
            "status": "valid" if diag["valid"] else "invalid",
            "lines_scanned": diag["lines_scanned"],
            "first_bad_line": diag["first_bad_line"],
            "reason": diag["reason"],
        })

    per_orphan = []
    for orphan in index.get("orphan_chains", []):
        per_orphan.append({
            "file": orphan.get("file"),
            "classification": orphan.get("classification", "unknown"),
            "valid_through_line": orphan.get("valid_through_line"),
            "total_lines": orphan.get("total_lines"),
            "chain_root_type": orphan.get("chain_root_type"),
        })

    epochs_valid = sum(1 for e in per_epoch if e["status"] == "valid")
    epochs_failed = sum(1 for e in per_epoch if e["status"] in ("invalid", "missing"))

    if not per_epoch:
        overall = "empty"
    elif epochs_failed == 0:
        overall = "valid"
    elif epochs_valid == 0:
        overall = "invalid"
    else:
        overall = "partial_failure"

    return {
        "overall_status": overall,
        "epochs_total": len(per_epoch),
        "epochs_valid": epochs_valid,
        "epochs_failed": epochs_failed,
        "orphan_chains_total": len(per_orphan),
        "per_epoch": per_epoch,
        "per_orphan": per_orphan,
    }


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
