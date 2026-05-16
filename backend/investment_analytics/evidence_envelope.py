"""Evidence envelope — provenance shell wrapped around every audit event.

The envelope is added INSIDE the audit record's ``event`` dict,
alongside (NOT nesting) the caller's event fields. Backward
compatible with all existing callers: ``event["event_type"]``,
``event["subject_token"]``, etc. remain accessible at the same
paths. New fields land alongside them.

## Envelope shape

```
event: {
  // caller's original fields (unchanged, additive)
  "event_type":   "<existing taxonomy>",
  "subject_token": "...",
  // ... whatever the caller put in ...

  // NEW envelope fields (always present after step 3)
  "evidence_kind":               "<one of EVIDENCE_KINDS>" | null,
  "run_id":                      str (UUID4 default, loose format),
  "parent_run_id":               str | null,
  "generated_at":                iso8601 str,
  "methodology":                 { methodology_schema_version, equity_metric, ... },
  "envelope_schema_version":     "v1",
  "envelope_schema_fingerprint": sha256(canonical-json-schema-definition),
  "inputs":                      { code_sha, python_version, ... }
}
```

## evidence_kind semantics

``evidence_kind`` is a closed enum of 7 values (see EVIDENCE_KINDS).
``None`` is a valid value with explicit meaning: the audit event is
provenance-tracked but intentionally unclassified — it does NOT fit
one of the seven canonical kinds, AND the caller knows this. This
distinguishes from a missing-evidence-kind bug: callers MUST pass
either a valid string OR ``None`` explicitly; passing nothing
defaults to ``None`` (intentional unclassified) via the kwarg
default in ``build_event_envelope``.

Step 3 lands the envelope on ALL audit events (existing callers
get ``evidence_kind=None`` automatically). Step 4 migrates the
well-classified call sites (``portfolio_health_check``,
``rank_category``, etc.) to pass explicit ``evidence_kind`` and
move heavy payloads to by-reference evidence files.

## run_id / parent_run_id

UUID4 is the project default. The architecture deliberately does
NOT hardcode UUID universally forever — deterministic replay runs,
imported experiments, scheduler correlation, and batch lineage may
need alternate ID strategies. Validation is loose: any non-empty
string ≤128 chars with no whitespace is accepted.

``parent_run_id`` is the lineage primitive: same-day reruns,
replay results referencing their original, experiment runs derived
from earlier ones all link via this field.

## Why hash provenance into the chain

Methodology versions, code SHA, evidence kind and lineage ARE part
of the evidence — not unhashed metadata. If they were excluded from
the hash, an attacker (or a sloppy operator) could rewrite
provenance without breaking the chain. Including them means the
chain itself bears witness to the provenance.

## Schema fingerprint

``ENVELOPE_SCHEMA_FINGERPRINT`` is sha256 over a canonical-JSON
representation of the envelope's structural definition (including
the methodology dict keys, the inputs keys, and the EVIDENCE_KINDS
enum). Any change to that structural definition produces a new
fingerprint — explicit drift visibility for replay diagnostics.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from backend.investment_analytics.methodology import (
    METHODOLOGY_VERSIONS,
    METHODOLOGY_SCHEMA_VERSION,
    current_methodology,
)
from backend.investment_analytics.provenance import capture_provenance_inputs


# Closed enum. Adding a kind requires GOVERNANCE REVIEW (audit-schema +
# methodology + replay compatibility) — NOT a mandatory global
# schema_version bump (per the architecture decision 2026-05-15).
EVIDENCE_KINDS: frozenset[str] = frozenset({
    "ranking_snapshot",
    "portfolio_health_snapshot",
    "watchlist_run",
    "experiment_run",
    "replay_result",
    "drift_analysis",
    "regime_summary",
    "calibration_report",
})


ENVELOPE_SCHEMA_VERSION = "v1"


# Structural definition of the envelope — hashed to produce the
# fingerprint. Changing ANY of the listed keys/types/component-set
# changes the fingerprint, giving explicit drift visibility. The
# methodology-component list is embedded so adding/removing a
# component bumps the fingerprint even if the SHAPE looks the same.
_ENVELOPE_SCHEMA_DEFINITION = {
    "schema_version": ENVELOPE_SCHEMA_VERSION,
    "envelope_fields": {
        "evidence_kind":               "one of EVIDENCE_KINDS | null (intentionally unclassified)",
        "run_id":                      "str (UUID4 default; loose format, ≤128 chars, no whitespace)",
        "parent_run_id":               "str | null (same format rules as run_id)",
        "generated_at":                "iso8601 str (UTC)",
        "methodology":                 "methodology_dict",
        "envelope_schema_version":     "str",
        "envelope_schema_fingerprint": "str (sha256 hex)",
        "inputs":                      "inputs_dict",
    },
    "methodology_dict_keys": sorted(["methodology_schema_version", *METHODOLOGY_VERSIONS.keys()]),
    "methodology_schema_version": METHODOLOGY_SCHEMA_VERSION,
    "inputs_dict_keys": sorted([
        "code_sha",
        "python_version",
        "analyzer_version",
        "registry_hash",
        "registry_path",
        "cache_fingerprint",
    ]),
    "evidence_kinds": sorted(EVIDENCE_KINDS),
}


def _canonical_json_local(value: Any) -> str:
    """Local copy to avoid importing from audit (which imports from us).
    Matches audit._canonical_json byte-for-byte."""
    import json
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_local(value: str) -> str:
    import hashlib
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


ENVELOPE_SCHEMA_FINGERPRINT: str = _sha256_local(
    _canonical_json_local(_ENVELOPE_SCHEMA_DEFINITION)
)


def validate_evidence_kind(kind: Optional[str]) -> None:
    """Raise on unknown evidence_kind. ``None`` is intentionally
    valid (means 'unclassified'); not a bug, not a missing value."""
    if kind is None:
        return
    if not isinstance(kind, str):
        raise TypeError(
            f"evidence_kind must be str or None (intentionally unclassified), "
            f"got {type(kind).__name__}"
        )
    if kind not in EVIDENCE_KINDS:
        raise ValueError(
            f"Unknown evidence_kind: {kind!r}. "
            f"Must be one of {sorted(EVIDENCE_KINDS)} "
            f"or None (intentionally unclassified)."
        )


def _validate_id_format(value: str, field_name: str) -> None:
    """Loose validation. UUID4 is the project default but the
    architecture does NOT hardcode that universally — deterministic
    replays, imported experiments, scheduler correlation may need
    alternate ID strategies later. Accept anything reasonable."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, got {type(value).__name__}")
    if not value or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    if len(value) > 128:
        raise ValueError(f"{field_name} must be ≤128 chars (got {len(value)})")
    if any(c.isspace() for c in value):
        raise ValueError(f"{field_name} must not contain whitespace")


def _validate_or_generate_run_id(run_id: Optional[str]) -> str:
    if run_id is None:
        return str(uuid.uuid4())
    _validate_id_format(run_id, field_name="run_id")
    return run_id


def build_event_envelope(
    event: dict[str, Any],
    evidence_kind: Optional[str] = None,
    run_id: Optional[str] = None,
    parent_run_id: Optional[str] = None,
    inputs: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Wrap a raw event with the provenance envelope (additive).

    The original event's fields are preserved at the same paths; new
    envelope fields are added alongside. Returns a NEW dict; does NOT
    mutate the input.

    Args:
      event:         the caller's raw event dict (e.g. ``{"event_type": "rank_category", ...}``).
      evidence_kind: one of EVIDENCE_KINDS, or None for intentionally unclassified.
      run_id:        a stable id; if None, an auto-generated UUID4.
      parent_run_id: lineage primitive; None when this event has no parent.
      inputs:        pre-captured provenance inputs; if None, capture now.

    Validation errors raise. Methodology is snapshotted at call time
    (via current_methodology()) so later mutations to METHODOLOGY_VERSIONS
    do NOT change the snapshot embedded in the returned envelope.
    """
    validate_evidence_kind(evidence_kind)
    resolved_run_id = _validate_or_generate_run_id(run_id)
    if parent_run_id is not None:
        _validate_id_format(parent_run_id, field_name="parent_run_id")

    envelope = dict(event)  # additive copy; do not mutate caller's dict
    envelope["evidence_kind"] = evidence_kind
    envelope["run_id"] = resolved_run_id
    envelope["parent_run_id"] = parent_run_id
    envelope["generated_at"] = (
        datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    )
    envelope["methodology"] = current_methodology()
    envelope["envelope_schema_version"] = ENVELOPE_SCHEMA_VERSION
    envelope["envelope_schema_fingerprint"] = ENVELOPE_SCHEMA_FINGERPRINT
    envelope["inputs"] = inputs if inputs is not None else capture_provenance_inputs()
    return envelope
