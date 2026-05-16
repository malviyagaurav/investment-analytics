"""Governance-decision emitter.

emit_governance_decision is the single entry point for writing
typed ``governance_decision`` rows. It enforces:

  - Decision type is one of DECISION_TYPES (no free additions).
  - Operator identity is captured + typed.
  - Eligibility check is run BEFORE the decision is recorded.
  - If eligibility fails AND ``force_ineligible`` is False, the
    function REFUSES to emit and returns the typed eligibility
    refusal — the operator has not yet declared their intent.
  - If eligibility fails AND ``force_ineligible`` is True, the
    function requires ``override_reason`` ∈ OVERRIDE_REASONS
    (tightening #1) and records the override attestation in the
    payload.
  - Production state is snapshotted at decision time and bound
    into the hash chain.
  - The chain-append is the single side effect; production
    constants are NEVER mutated.

## Discipline (load-bearing)

The runner does NOT:
  - Mutate HIGH_CORRELATION_THRESHOLD or any other production
    constant.
  - Update threshold_recommendation.adoption_status by editing
    the prior row (audit rows are immutable). Future readers
    derive the effective adoption status from the LATEST
    governance_decision for a subject (linear chain scan).
  - Auto-promote without an explicit operator decision.
  - Auto-rollback. Rollback is an operator decision_type, not a
    timer or threshold.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.evidence.replay import find_record_by_run_id
from backend.evidence.store import emit_evidence
from backend.governance.config import (
    DECISION_TYPES,
    GOVERNANCE_DECISION_SCHEMA_VERSION,
    GOVERNANCE_ELIGIBILITY_VERSION,
    OVERRIDE_REASONS,
)
from backend.governance.eligibility import (
    check_eligibility,
    find_latest_calibration_for_subject,
)
from backend.governance.identity import build_operator_identity
from backend.governance.production_state import snapshot_production_state


ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_AUDIT_PATH = ROOT / "data" / "audit" / "audit.jsonl"


class EligibilityRefused(RuntimeError):
    """The eligibility check refused the decision and the operator
    did NOT pass force_ineligible. Carries the eligibility result
    so the caller can present typed refusal reasons to the user."""

    def __init__(self, message: str, eligibility_result: Dict[str, Any]):
        super().__init__(message)
        self.eligibility_result = eligibility_result


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _validate_decision_type(decision_type: str) -> None:
    if decision_type not in DECISION_TYPES:
        raise ValueError(
            f"decision_type must be one of {sorted(DECISION_TYPES)}, "
            f"got {decision_type!r}"
        )


def _validate_override_reason(reason: str) -> None:
    if reason not in OVERRIDE_REASONS:
        raise ValueError(
            f"override_reason must be one of {sorted(OVERRIDE_REASONS)}, "
            f"got {reason!r}"
        )


def _build_override_attestation(
    force_ineligible: bool,
    override_reason: Optional[str],
) -> Dict[str, Any]:
    """Tightening #1: an override is a distinct typed governance
    claim. The attestation block is present on EVERY decision
    (override_used=False when no override happened) so the replay
    surface is uniform and the schema_fingerprint is stable across
    forced and non-forced decisions."""
    if not force_ineligible:
        return {"override_used": False, "override_reason": None}
    if override_reason is None:
        raise ValueError(
            "force_ineligible=True requires override_reason ∈ "
            f"{sorted(OVERRIDE_REASONS)}"
        )
    _validate_override_reason(override_reason)
    return {"override_used": True, "override_reason": override_reason}


def _find_prior_approval_for_target(
    audit_path: Path, target_canonical_id: str,
) -> Optional[Dict[str, Any]]:
    """Linear scan for the most recent ``approve`` decision whose
    subject targets the same canonical id. Used by rollback paths
    to populate ``rollback_target_run_id`` automatically when not
    provided. Returns the inner event dict, or None."""
    if not audit_path.exists():
        return None
    import json
    latest: Optional[Dict[str, Any]] = None
    with audit_path.open("r", encoding="utf-8", newline="\n") as h:
        for line in h:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = record.get("event", {}) or {}
            if event.get("evidence_kind") != "governance_decision":
                continue
            if event.get("decision_type") != "approve":
                continue
            if event.get("subject_target_canonical_id") != target_canonical_id:
                continue
            latest = event
    return latest


def _build_payload(
    *,
    decision_type: str,
    subject_run_id: str,
    subject_event: Dict[str, Any],
    decided_at: str,
    operator_identity: Dict[str, Any],
    eligibility_result: Dict[str, Any],
    override_attestation: Dict[str, Any],
    production_state: Dict[str, Any],
    supersedes_run_id: Optional[str],
    rollback_target_run_id: Optional[str],
    rationale: str,
) -> Dict[str, Any]:
    target = eligibility_result.get("subject_target_canonical_id")
    reliability_run_id = eligibility_result.get("reliability_score_run_id")
    calibration_run_id = find_latest_calibration_for_subject(subject_event)
    return {
        "schema_version":               GOVERNANCE_DECISION_SCHEMA_VERSION,
        "governance_eligibility_version": GOVERNANCE_ELIGIBILITY_VERSION,
        "decision_type":                decision_type,
        "subject_run_id":               subject_run_id,
        "subject_evidence_kind":        "threshold_recommendation",
        "subject_target_canonical_id":  target,
        "decided_at":                   decided_at,
        "operator_identity":            operator_identity,
        "evidence_basis": {
            "threshold_recommendation_run_id": subject_run_id,
            "reliability_score_run_id":        reliability_run_id,
            "calibration_report_run_id":       calibration_run_id,
        },
        "eligibility_check": {
            "eligibility_passed":            eligibility_result["eligibility_passed"],
            "reliability_score_at_decision": eligibility_result[
                "reliability_score_at_decision"],
            "reliability_score_floor":       eligibility_result[
                "reliability_score_floor"],
            "refusal_reasons":               list(
                eligibility_result["refusal_reasons"]),
        },
        "override_attestation":         override_attestation,
        "production_state_at_decision": production_state,
        "supersedes_run_id":            supersedes_run_id,
        "rollback_target_run_id":       rollback_target_run_id,
        "non_semantic_metadata":        {"rationale": rationale},
        "methodology_kind":             "operator_decision",
    }


def _build_audit_event(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Lightweight audit event; heavy detail is in the by-reference
    payload. Includes the load-bearing typed fields that replay,
    queries, and refusal-symmetry consumers need without loading
    the evidence file."""
    return {
        "event_type":                    "governance_decision",
        "decision_type":                 payload["decision_type"],
        "subject_run_id":                payload["subject_run_id"],
        "subject_evidence_kind":         payload["subject_evidence_kind"],
        "subject_target_canonical_id":   payload["subject_target_canonical_id"],
        "eligibility_passed":            payload["eligibility_check"][
            "eligibility_passed"],
        "reliability_score_at_decision": payload["eligibility_check"][
            "reliability_score_at_decision"],
        "override_used":                 payload["override_attestation"][
            "override_used"],
        "supersedes_run_id":             payload["supersedes_run_id"],
        "rollback_target_run_id":        payload["rollback_target_run_id"],
        "schema_version":                payload["schema_version"],
    }


def emit_governance_decision(
    *,
    decision_type: str,
    subject_run_id: str,
    operator_id: str,
    operator_role: str = "owner",
    attestation_method: str = "local_terminal",
    supporting_evidence: Optional[List[str]] = None,
    rationale: str = "",
    force_ineligible: bool = False,
    override_reason: Optional[str] = None,
    supersedes_run_id: Optional[str] = None,
    rollback_target_run_id: Optional[str] = None,
    audit_path: Optional[Path] = None,
    emit: bool = True,
) -> Dict[str, Any]:
    """Emit one typed governance_decision row.

    Args:
      decision_type:           one of DECISION_TYPES.
      subject_run_id:          the threshold_recommendation being
                               decided on.
      operator_id:             non-empty typed declaration of WHO.
      operator_role:           one of OPERATOR_ROLES (default
                               ``owner``). ``automation`` is
                               REFUSED for approve / reject /
                               rollback — those require a human.
      attestation_method:      one of ATTESTATION_METHODS.
      supporting_evidence:     optional breadcrumbs (ticket IDs,
                               meeting notes).
      rationale:               free-text operator note. Stored in
                               non_semantic_metadata — does NOT
                               feed back into eligibility or
                               reliability decisions.
      force_ineligible:        when True, proceed even if
                               eligibility fails. Requires
                               override_reason ∈ OVERRIDE_REASONS.
      override_reason:         typed enum value; required iff
                               force_ineligible=True.
      supersedes_run_id:       prior governance_decision this one
                               supersedes. Linear chain — no
                               diamond merges.
      rollback_target_run_id:  prior approve decision being rolled
                               back. Required for decision_type
                               ``rollback`` (auto-discovered via
                               most-recent-approve scan if not
                               passed; if no prior approve exists,
                               raises ValueError — rollback with
                               no target is a programming bug).
      audit_path:              chain path (defaults to repo audit).
      emit:                    when False, returns the payload
                               without appending. Used by tests.

    Returns the audit record dict from emit_evidence (or the
    payload dict when emit=False).

    Raises:
      ValueError:           bad decision_type / operator_role /
                            override_reason / missing
                            override_reason when force_ineligible.
      EligibilityRefused:   eligibility failed and
                            force_ineligible=False. Carries the
                            typed eligibility result.
      RuntimeError:         rollback requested but no prior
                            approve exists for the target.
    """
    _validate_decision_type(decision_type)

    # Automation cannot adopt into production. The role check fires
    # BEFORE eligibility / production-state work — operator-role
    # discipline is a hard precondition, not an after-the-fact
    # gate.
    if (decision_type in {"approve", "reject", "rollback"}
            and operator_role == "automation"):
        raise ValueError(
            f"decision_type {decision_type!r} requires a human role; "
            f"automation may only emit request_review"
        )

    operator_identity = build_operator_identity(
        operator_id=operator_id,
        operator_role=operator_role,
        attestation_method=attestation_method,
        supporting_evidence=supporting_evidence,
    )

    audit_path = audit_path or DEFAULT_AUDIT_PATH

    # Subject lookup + eligibility (typed refusal, never raises).
    eligibility_result = check_eligibility(subject_run_id, audit_path)

    # Subject event for evidence_basis. May be None when
    # eligibility refused with subject_not_found.
    subject_record = find_record_by_run_id(audit_path, subject_run_id)
    subject_event = (subject_record or {}).get("event", {}) or {}

    # Build override attestation FIRST so we validate
    # force_ineligible+override_reason regardless of eligibility
    # outcome. A misspecified override should fail loudly even on
    # eligible subjects — the operator is making a typed claim
    # that should be syntactically clean.
    override_attestation = _build_override_attestation(
        force_ineligible=force_ineligible,
        override_reason=override_reason,
    )

    if not eligibility_result["eligibility_passed"] and not force_ineligible:
        # Hard refusal: do NOT emit. The operator has not declared
        # their intent to override.
        raise EligibilityRefused(
            f"eligibility refused for subject {subject_run_id!r}: "
            f"{eligibility_result['refusal_reasons']}",
            eligibility_result=eligibility_result,
        )

    target = eligibility_result.get("subject_target_canonical_id")

    # Production-state snapshot. The target may legitimately be
    # missing if the subject was not a threshold_recommendation
    # (eligibility refused with subject_unsupported). For
    # force_ineligible paths where target is unknown, we still
    # must emit something — record a typed null snapshot.
    if target is None:
        production_state: Dict[str, Any] = {
            "target_canonical_id": None,
            "current_value":       None,
            "value_byte_hash":     None,
            "source_attestation":  None,
            "snapshot_refusal":    "target_unknown_at_decision_time",
        }
    else:
        try:
            production_state = snapshot_production_state(target)
        except KeyError:
            production_state = {
                "target_canonical_id": target,
                "current_value":       None,
                "value_byte_hash":     None,
                "source_attestation":  None,
                "snapshot_refusal":    "target_not_registered_for_snapshot",
            }

    # Rollback discipline: a rollback row MUST cite the prior
    # approve it is undoing. Auto-discover when the caller did not
    # pass one; refuse if nothing exists.
    if decision_type == "rollback":
        if rollback_target_run_id is None:
            prior = (
                _find_prior_approval_for_target(audit_path, target)
                if target is not None else None
            )
            if prior is None:
                raise RuntimeError(
                    f"rollback refused: no prior approve decision exists "
                    f"for target {target!r}"
                )
            rollback_target_run_id = prior.get("run_id")

    payload = _build_payload(
        decision_type=decision_type,
        subject_run_id=subject_run_id,
        subject_event=subject_event,
        decided_at=_now_iso(),
        operator_identity=operator_identity,
        eligibility_result=eligibility_result,
        override_attestation=override_attestation,
        production_state=production_state,
        supersedes_run_id=supersedes_run_id,
        rollback_target_run_id=rollback_target_run_id,
        rationale=rationale,
    )
    audit_event = _build_audit_event(payload)

    if not emit:
        return {"event": audit_event, "payload": payload}

    return emit_evidence(
        audit_log_path=audit_path,
        evidence_kind="governance_decision",
        audit_event=audit_event,
        payload=payload,
        parent_run_id=supersedes_run_id or subject_run_id,
    )


# ── Query helpers ────────────────────────────────────────────────────


def find_governance_decisions(
    audit_path: Optional[Path] = None,
    *,
    subject_run_id: Optional[str] = None,
    subject_target_canonical_id: Optional[str] = None,
    decision_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Linear chain scan for governance_decision rows with optional
    filters. Returns inner event dicts in chain order."""
    audit_path = audit_path or DEFAULT_AUDIT_PATH
    if not audit_path.exists():
        return []
    import json
    out: List[Dict[str, Any]] = []
    with audit_path.open("r", encoding="utf-8", newline="\n") as h:
        for line in h:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = record.get("event", {}) or {}
            if event.get("evidence_kind") != "governance_decision":
                continue
            if (subject_run_id is not None
                    and event.get("subject_run_id") != subject_run_id):
                continue
            if (subject_target_canonical_id is not None
                    and event.get("subject_target_canonical_id")
                    != subject_target_canonical_id):
                continue
            if (decision_type is not None
                    and event.get("decision_type") != decision_type):
                continue
            out.append(event)
    return out


def latest_effective_decision(
    audit_path: Optional[Path] = None,
    *,
    subject_target_canonical_id: str,
) -> Optional[Dict[str, Any]]:
    """Return the LATEST (most-recent in chain order) decision for
    a given target. Used by readers that want the current
    governance state without re-deriving it from a forward scan.

    Note: this is a DERIVED VIEW. The chain is the source of
    truth; this is a convenience shortcut. It does NOT mutate the
    chain and does NOT pre-compute adoption_status."""
    decisions = find_governance_decisions(
        audit_path=audit_path,
        subject_target_canonical_id=subject_target_canonical_id,
    )
    if not decisions:
        return None
    return decisions[-1]
