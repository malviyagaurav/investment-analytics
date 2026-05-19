"""Content-addressable evidence file store.

Heavy payloads (ranking snapshots, portfolio-health snapshots) live
here, externalized from the audit log. The audit record references
them via ``evidence_ref: {path, sha256, size_bytes}``.

## Write protocol (strict ordering, mandated by integrity model)

1. Caller pre-snapshots provenance ``inputs`` (so the audit record
   and the evidence file share the same provenance).
2. Caller generates a fresh ``run_id``.
3. Caller builds the FULL envelope (event metadata + payload +
   provenance) via ``evidence_envelope.build_event_envelope``.
4. Caller invokes ``write_evidence(audit_dir, kind, envelope)``:
   a. Sanitizes (PII redaction).
   b. Canonical JSON encoding to deterministic bytes.
   c. Atomic write (tmp + fsync + rename) to
      ``data/evidence/<kind>/<run_id>.json``.
   d. Reads bytes back from disk, verifies sha256 matches the
      pre-write hash — disk-corruption guard per the user's
      explicit integrity emphasis.
   e. Returns ``evidence_ref = {path, sha256, size_bytes}``.
5. Caller calls ``audit.append_audit_record`` with the LIGHTWEIGHT
   audit event including ``evidence_ref``, SAME ``run_id`` and SAME
   ``inputs`` so the audit envelope matches the evidence file's.

If step 5 fails, the evidence file is orphaned on disk. That is
acceptable — the integrity model permits orphan files but NEVER
permits a dangling audit reference (audit ref pointing at a missing
file). Cleanup of orphans is a separate operator task.

## Immutability

Same ``run_id`` twice → ``FileExistsError``. Evidence is immutable;
reruns generate fresh ``run_id`` per the architecture's
"no same-day overwrite" rule.

## Hashing

SHA-256 over RAW FILE BYTES — what was persisted, not a
re-canonicalized parse of the content. Disagreement between persisted
artifact and hashed artifact would silently break replay later.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional

from backend.investment_analytics.audit import sanitize_audit_event


def write_evidence(
    audit_dir: Path,
    evidence_kind: str,
    envelope: dict[str, Any],
) -> dict[str, Any]:
    """Persist an evidence envelope to a by-reference file.

    Args:
      audit_dir:     directory containing ``audit.jsonl`` (typically
                     ``<repo>/data/audit/``). Evidence is sibling at
                     ``<repo>/data/evidence/<kind>/``.
      evidence_kind: one of EVIDENCE_KINDS (used as subdirectory).
      envelope:      full envelope dict including ``run_id`` and
                     ``payload``. Will be sanitized before write.

    Returns:
      ``{"path": <relative to data/>, "sha256": hex, "size_bytes": int}``
      — the ``evidence_ref`` the caller embeds in the audit record.

    Raises:
      KeyError:        envelope lacks ``run_id`` (caller programming error).
      FileExistsError: a file already exists for this run_id (immutability).
      RuntimeError:    bytes read back from disk do not match the
                       computed sha256 (disk corruption or filesystem bug).
    """
    run_id = envelope.get("run_id")
    if not run_id:
        raise KeyError(
            "envelope must contain 'run_id' before write_evidence. "
            "Generate a UUID4 or pass run_id= explicitly to build_event_envelope."
        )

    data_dir = audit_dir.parent  # data/
    kind_dir = data_dir / "evidence" / evidence_kind
    kind_dir.mkdir(parents=True, exist_ok=True)
    target = kind_dir / f"{run_id}.json"

    if target.exists():
        raise FileExistsError(
            f"Evidence file already exists: {target}. "
            f"Evidence is immutable — generate a new run_id for a rerun."
        )

    sanitized = sanitize_audit_event(envelope)
    file_bytes = json.dumps(
        sanitized,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
    ).encode("utf-8")
    expected_sha = hashlib.sha256(file_bytes).hexdigest()
    size_bytes = len(file_bytes)

    # Atomic write: tmp + fsync + rename. Matches the cache/snapshot
    # discipline already in use elsewhere in the project.
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(file_bytes)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(target)

    # Verify after write — the integrity model requires that what we
    # claim was persisted matches what is actually on disk before the
    # audit record references it.
    with target.open("rb") as handle:
        persisted = handle.read()
    actual_sha = hashlib.sha256(persisted).hexdigest()
    if actual_sha != expected_sha:
        # Hash mismatch is rare but must NEVER produce a dangling ref.
        # The file is already on disk; leave it as forensic evidence
        # and refuse to return a ref the caller would chain into audit.
        raise RuntimeError(
            f"Evidence file hash mismatch after write: {target}. "
            f"Expected {expected_sha}, got {actual_sha}. "
            f"Possible disk corruption or filesystem inconsistency. "
            f"File preserved for forensic inspection; do NOT chain "
            f"this run_id into the audit log."
        )

    # ``.as_posix()`` forces forward-slash separators on all platforms;
    # ``str(Path)`` would emit backslash on Windows, making the audit
    # chain's ``evidence_ref["path"]`` non-byte-canonical across hosts
    # and unreplay-able on POSIX given a Windows-emitted ref.
    rel_path = target.relative_to(data_dir).as_posix()
    return {"path": rel_path, "sha256": actual_sha, "size_bytes": size_bytes}


def emit_evidence(
    audit_log_path: Path,
    evidence_kind: str,
    audit_event: dict[str, Any],
    payload: Any,
    *,
    parent_run_id: Optional[str] = None,
) -> dict[str, Any]:
    """Write evidence file FIRST, then append audit record with ref.

    The evidence file carries the FULL envelope (audit_event fields +
    ``payload`` + all provenance). The audit record carries ONLY the
    lightweight ``audit_event`` + ``evidence_ref``. Both share the
    same ``run_id``, ``parent_run_id``, ``evidence_kind``, and
    ``inputs`` so replay can correlate them deterministically.

    Ordering invariant (user-mandated): evidence file lands on disk
    BEFORE the audit ref is appended. If the audit append fails the
    evidence file is orphaned — acceptable. The reverse (audit ref
    referencing a missing file) is NEVER acceptable.

    The ``inputs`` block — including ``cache_fingerprint`` — is
    captured by the first envelope build and re-used in the audit
    append so both records reflect the same snapshot of provenance.
    ``generated_at`` legitimately differs between them: the evidence
    timestamp marks persistence; the audit timestamp marks the chain
    append.

    Args:
      audit_log_path: typically ``<repo>/data/audit/audit.jsonl``.
      evidence_kind:  one of EVIDENCE_KINDS (drives the subdirectory).
      audit_event:    lightweight event dict (event_type, counts,
                      subject_token, etc.) — NO heavy payload here.
      payload:        the heavy result dict (ranking_to_dict output,
                      portfolio_health_to_dict output, etc.).
      parent_run_id:  lineage primitive when this emission derives
                      from a prior run.

    Returns:
      The audit record dict returned by ``append_audit_record``.
    """
    from backend.investment_analytics.audit import append_audit_record
    from backend.investment_analytics.evidence_envelope import build_event_envelope

    full_envelope = build_event_envelope(
        {**audit_event, "payload": payload},
        evidence_kind=evidence_kind,
        parent_run_id=parent_run_id,
    )
    run_id = full_envelope["run_id"]
    captured_inputs = full_envelope["inputs"]

    audit_dir = audit_log_path.parent
    evidence_ref = write_evidence(audit_dir, evidence_kind, full_envelope)

    return append_audit_record(
        audit_log_path,
        {**audit_event, "evidence_ref": evidence_ref},
        evidence_kind=evidence_kind,
        run_id=run_id,
        parent_run_id=parent_run_id,
        inputs=captured_inputs,
    )
