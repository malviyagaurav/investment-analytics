"""One-time idempotent migration creating ``data/audit/epochs.json``.

Step 2 of the evidence-layer roadmap. After this runs:

  - ``epochs.json`` sidecar registers the live ``audit.jsonl`` as
    epoch 1 with ``chain_root_type: genesis`` and ``status: open``.
  - Any sibling ``*.bak`` files are classified (chained predecessor
    vs independent legacy vs partially corrupt vs orphan snapshot)
    and recorded under ``orphan_chains`` so historical artefacts
    are documented but never silently chained into the live log.
  - New audit records carry ``chain_epoch`` (already wired in
    ``append_audit_record``); the migration itself does NOT mutate
    the 7,210+ existing records — they remain byte-immutable.
  - No ``_chain_close`` / ``_chain_open`` markers are synthesised;
    those land only at a real future rotation.

The schema definition for ``epochs.json`` is hashed at module load
to produce ``EPOCHS_SCHEMA_FINGERPRINT`` (sha256 over the canonical-
JSON schema definition). Changing the schema definition changes the
fingerprint, giving explicit visibility into structural drift.

Idempotency: if ``epochs.json`` already exists, the migration is a
no-op that returns the existing index. Operators can re-run safely.

CLI usage:
  python -m backend.investment_analytics.audit_migrate
  python -m backend.investment_analytics.audit_migrate --dry-run
  python -m backend.investment_analytics.audit_migrate --audit-dir <path>
"""
from __future__ import annotations

import argparse
from backend.investment_analytics._locking import (
    acquire_exclusive_blocking,
    release as release_exclusive,
)
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from backend.investment_analytics.audit import (
    _canonical_json,
    _sha256,
    verify_audit_chain_diag,
)


# Canonical schema definition — hashed to produce schema_fingerprint.
# Any change to the structural template here CHANGES the fingerprint
# (intentional: explicit drift visibility for replay/diagnostics).
EPOCHS_SCHEMA_DEFINITION: Dict[str, Any] = {
    "schema_version": "v1",
    "top_level_fields": {
        "schema_version":       "str",
        "schema_fingerprint":   "str (sha256 hex)",
        "epochs":               "list[epoch_entry]",
        "orphan_chains":        "list[orphan_entry]",
        "registered_at":        "iso8601 str",
    },
    "epoch_entry": {
        "epoch":                "int (>= 1, monotonic)",
        "file":                 "str (filename relative to audit_dir)",
        "started_at":           "iso8601 str",
        "closed_at":            "iso8601 str | null",
        "first_record_hash":    "str (sha256 hex)",
        "last_record_hash":     "str (sha256 hex) | null",
        "record_count_at_close":"int | null",
        "status":               "open | closed | sealed | quarantined",
        "chain_root_type":      "genesis | rotated_handoff | imported_legacy | repaired | replay_reconstructed",
    },
    "orphan_entry": {
        "file":                    "str",
        "classification":          "str",
        "chain_root_type":         "genesis | rotated_handoff | imported_legacy | repaired | replay_reconstructed",
        "first_record_timestamp":  "iso8601 str | null",
        "last_record_timestamp":   "iso8601 str | null",
        "first_prev_hash":         "str (sha256 hex) | null",
        "last_record_hash":        "str (sha256 hex) | null",
        "valid_through_line":      "int | null",
        "total_lines":             "int",
        "first_failure_reason":    "str | null",
        "note":                    "str",
    },
}
EPOCHS_SCHEMA_FINGERPRINT = _sha256(_canonical_json(EPOCHS_SCHEMA_DEFINITION))


def _read_first_record(path: Path) -> Optional[Dict[str, Any]]:
    """First non-blank JSON record of ``path``, or None if file is
    missing / empty / unparseable."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="\n") as handle:
        for line in handle:
            if line.strip():
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    return None
    return None


def _read_last_record(path: Path) -> Optional[Dict[str, Any]]:
    """Last non-blank JSON record of ``path``, or None if file is
    missing / empty / unparseable."""
    if not path.exists():
        return None
    last: Optional[str] = None
    with path.open("r", encoding="utf-8", newline="\n") as handle:
        for line in handle:
            if line.strip():
                last = line
    if last is None:
        return None
    try:
        return json.loads(last)
    except json.JSONDecodeError:
        return None


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8", newline="\n") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def classify_orphan(orphan_path: Path, live_first_prev_hash: Optional[str]) -> Dict[str, Any]:
    """Classify a candidate orphan file.

    Cross-reference rules:
      - If ``orphan.last_record_hash`` matches ``live_first_prev_hash``,
        the orphan IS a chained predecessor of the live chain.
        Classification = ``chained_predecessor``.
      - Otherwise the orphan is independent of the live chain.
        Internal validity determines fine-grained classification:
          - clean chain end-to-end → ``valid_independent_legacy``
          - chain breaks mid-file  → ``partially_corrupt_independent_legacy``
          - file unparseable        → ``unparseable_orphan``

    chain_root_type is ``genesis`` when the orphan's first record
    has ``prev_hash == 0*64``; otherwise it's ``imported_legacy``
    (the orphan came from somewhere else with an unknown root).
    """
    total_lines = _count_lines(orphan_path)
    first = _read_first_record(orphan_path)
    last = _read_last_record(orphan_path)

    if first is None or last is None:
        return {
            "file": orphan_path.name,
            "classification": "unparseable_orphan",
            "chain_root_type": "imported_legacy",
            "first_record_timestamp": None,
            "last_record_timestamp": None,
            "first_prev_hash": None,
            "last_record_hash": None,
            "valid_through_line": None,
            "total_lines": total_lines,
            "first_failure_reason": "file empty or first/last record unparseable",
            "note": "File could not be parsed as JSONL; preserved as-is, never walked.",
        }

    first_prev = first.get("prev_hash") or first.get("previous_hash")
    last_hash = last.get("current_hash") or last.get("record_hash")
    genesis = "0" * 64

    diag = verify_audit_chain_diag(orphan_path)
    if diag["valid"]:
        internal_status = "valid_independent_legacy"
        valid_through = diag["lines_scanned"]
        failure_reason: Optional[str] = None
    else:
        internal_status = "partially_corrupt_independent_legacy"
        valid_through = (diag["first_bad_line"] - 1) if diag.get("first_bad_line") else None
        failure_reason = (
            f"{diag.get('reason', 'unknown')} at line {diag.get('first_bad_line')}"
        )

    # Chained-predecessor override: if the orphan's last hash matches
    # the live chain's first prev_hash AND that link is non-genesis,
    # we are looking at a true predecessor (not actually an orphan).
    if (
        live_first_prev_hash is not None
        and live_first_prev_hash != genesis
        and last_hash == live_first_prev_hash
    ):
        classification = "chained_predecessor"
    else:
        classification = internal_status

    chain_root_type = "genesis" if first_prev == genesis else "imported_legacy"

    if classification == "chained_predecessor":
        note = (
            "Orphan's last hash matches the live chain's first prev_hash. "
            "This file IS a chained predecessor; preserved with that lineage recorded."
        )
    elif classification == "valid_independent_legacy":
        note = (
            "Independent chain (does NOT chain into the live audit). "
            "Internally valid end-to-end. Preserved as historical artifact; "
            "NOT walked by verify_audit_chain_multi."
        )
    elif classification == "partially_corrupt_independent_legacy":
        note = (
            "Independent chain (does NOT chain into the live audit) with a "
            "chain break mid-file. Preserved as historical artifact; "
            "NOT walked by verify_audit_chain_multi."
        )
    else:
        note = "Unclassified orphan; preserved as-is for operator review."

    return {
        "file": orphan_path.name,
        "classification": classification,
        "chain_root_type": chain_root_type,
        "first_record_timestamp": first.get("timestamp"),
        "last_record_timestamp": last.get("timestamp"),
        "first_prev_hash": first_prev,
        "last_record_hash": last_hash,
        "valid_through_line": valid_through,
        "total_lines": total_lines,
        "first_failure_reason": failure_reason,
        "note": note,
    }


def _discover_orphan_candidates(audit_dir: Path) -> list[Path]:
    """Files in ``audit_dir`` that look like audit logs but are NOT
    the live ``audit.jsonl``. Currently: any ``*.jsonl`` or
    ``audit.jsonl.*`` sibling."""
    candidates: list[Path] = []
    live = audit_dir / "audit.jsonl"
    for entry in audit_dir.iterdir():
        if not entry.is_file():
            continue
        if entry == live:
            continue
        name = entry.name
        if name.endswith(".jsonl") or name.startswith("audit.jsonl."):
            candidates.append(entry)
    return sorted(candidates)


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    """tmp + rename so a concurrent reader never sees a half-written
    JSON. Matches the pattern used by jobs.watchlist.save_snapshot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def migrate_audit_to_epochs(
    audit_dir: Path,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Idempotent migration. Creates ``epochs.json`` if absent;
    returns the existing index if present.

    Acquires ``fcntl.flock`` on the live audit file before reading
    so a concurrent appender cannot grow the chain mid-migration.
    The lock is released before writing ``epochs.json`` (the sidecar
    is a distinct file, no contention).

    If the live chain fails end-to-end verification, the migration
    REFUSES to register it as epoch 1 and raises RuntimeError. Do
    not formalize broken state into the new index.

    Args:
        audit_dir: directory containing audit.jsonl (and optionally
                   *.bak / *.jsonl orphan files).
        dry_run:   if True, returns the index that WOULD be written
                   without touching epochs.json on disk.

    Returns the index dict (existing or newly built).
    """
    audit_dir = audit_dir.resolve()
    epochs_path = audit_dir / "epochs.json"
    live = audit_dir / "audit.jsonl"

    # Idempotency check — before locking, before verifying.
    if epochs_path.exists() and not dry_run:
        try:
            existing = json.loads(epochs_path.read_text(encoding="utf-8"))
            existing["__migration_status"] = "already_migrated"
            return existing
        except (json.JSONDecodeError, OSError) as exc:
            raise RuntimeError(
                f"epochs.json exists but cannot be parsed: {exc}. "
                "Manual inspection required before re-migrating."
            ) from exc

    # Lock the live audit file (using the same cross-platform lock
    # adapter the append path uses, so migration and live appends
    # cannot interleave on either POSIX or Windows). If the file
    # doesn't exist yet, create empty.
    audit_dir.mkdir(parents=True, exist_ok=True)
    live.touch(exist_ok=True)

    with live.open("a", encoding="utf-8", newline="\n") as handle:
        acquire_exclusive_blocking(handle.fileno())
        try:
            # End-to-end verification of the live chain. Refuse to
            # formalize a broken chain into epochs.json.
            diag = verify_audit_chain_diag(live)
            line_count = _count_lines(live)

            if line_count == 0:
                first_record = None
                started_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
                first_hash = None
            else:
                if not diag["valid"]:
                    raise RuntimeError(
                        f"Live audit chain at {live} is invalid: "
                        f"{diag.get('reason')} at line {diag.get('first_bad_line')}. "
                        f"Migration refuses to register a broken chain as epoch 1. "
                        f"Repair or quarantine before re-running."
                    )
                first_record = _read_first_record(live)
                started_at = (
                    first_record.get("timestamp")
                    if first_record else
                    datetime.now(timezone.utc).replace(microsecond=0).isoformat()
                )
                first_hash = (
                    first_record.get("current_hash")
                    or first_record.get("record_hash")
                ) if first_record else None

            epoch_1 = {
                "epoch": 1,
                "file": "audit.jsonl",
                "started_at": started_at,
                "closed_at": None,
                "first_record_hash": first_hash,
                "last_record_hash": None,
                "record_count_at_close": None,
                "status": "open",
                "chain_root_type": "genesis",
            }

            # Orphan classification. Cross-reference uses the live
            # chain's first prev_hash (genesis = 0*64 for our case).
            live_first_prev = (
                first_record.get("prev_hash") or first_record.get("previous_hash")
            ) if first_record else None

            orphans = []
            for orphan_path in _discover_orphan_candidates(audit_dir):
                orphans.append(classify_orphan(orphan_path, live_first_prev))

            index = {
                "schema_version": "v1",
                "schema_fingerprint": EPOCHS_SCHEMA_FINGERPRINT,
                "epochs": [epoch_1],
                "orphan_chains": orphans,
                "registered_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            }

            if dry_run:
                index["__migration_status"] = "dry_run"
                return index

            _atomic_write_json(epochs_path, index)
            index["__migration_status"] = "migrated"
            return index

        finally:
            release_exclusive(handle.fileno())


def _main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m backend.investment_analytics.audit_migrate",
        description="Create data/audit/epochs.json (idempotent).",
    )
    parser.add_argument(
        "--audit-dir",
        default=None,
        help="Audit directory (defaults to <repo>/data/audit/).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the index that WOULD be written without touching epochs.json.",
    )
    args = parser.parse_args(argv)

    if args.audit_dir:
        audit_dir = Path(args.audit_dir)
    else:
        repo_root = Path(__file__).resolve().parent.parent.parent
        audit_dir = repo_root / "data" / "audit"

    index = migrate_audit_to_epochs(audit_dir, dry_run=args.dry_run)
    print(json.dumps(index, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_main(sys.argv[1:]))
