"""Pure eligibility check for a specific threshold_recommendation.

Eligibility is engineered policy: given the chain state, decide
whether THIS PARTICULAR threshold_recommendation INSTANCE has met
the floor for typed operator approval. The decision is independent
of the operator's choice — eligibility says "may a human even
consider approving this?" The operator may still --force-ineligible,
but that path requires a typed OVERRIDE_REASONS attestation.

## Scope — recommendation-instance, not target

Eligibility is RECOMMENDATION-INSTANCE scoped. A single
target_canonical_id may produce many threshold_recommendations
over time (re-calibrations, supersessions, new methodology
versions). Each recommendation is its own subject and is
evaluated against the reliability_score rows that were emitted
ABOUT IT — i.e., scores whose ``target_run_id`` equals the
cited subject_run_id. Reliability scores for OTHER prior
recommendations on the SAME target are intentionally ignored:
inheriting a target-level endorsement would let stale evidence
silently authorize a fresh artifact, defeating the
recommendation-scoped lineage the architecture relies on.

The only target-level concern in this layer is
``target_unsupported`` — the system simply has no calibration
surface registered for the target at all.

## Function signature

  check_eligibility(
      subject_run_id: str,
      audit_path: Path,
  ) -> EligibilityResult

EligibilityResult is a typed payload:

  {
    "eligibility_passed":             bool,
    "subject_run_id":                 str,
    "subject_evidence_kind":          "threshold_recommendation" | None,
    "subject_target_canonical_id":    str | None,
    "reliability_score_run_id":       str | None,
    "reliability_score_at_decision":  float | None,
    "reliability_score_floor":        float,
    "refusal_reasons":                List[str],   # closed enum values
    "consulted_run_ids":              List[str],
  }

The function NEVER raises on missing substrate; refusal is typed
data. It DOES raise on programming errors (audit_path is not a
Path; subject_run_id is empty).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.calibration.config import CALIBRATION_TARGETS
from backend.evidence.replay import find_record_by_run_id
from backend.governance.config import (
    ELIGIBILITY_REFUSAL_REASONS,
    GOVERNANCE_ELIGIBILITY_VERSION,
    RELIABILITY_SCORE_FLOOR_FOR_PROMOTION,
)
from backend.reliability.runner import find_reliability_scores


def _validate_refusal_reason(reason: str) -> None:
    if reason not in ELIGIBILITY_REFUSAL_REASONS:
        raise ValueError(
            f"eligibility refusal_reason must be one of "
            f"{sorted(ELIGIBILITY_REFUSAL_REASONS)}, got {reason!r}"
        )


def check_eligibility(
    subject_run_id: str,
    audit_path: Path,
) -> Dict[str, Any]:
    """Return a typed EligibilityResult for the given
    threshold_recommendation run_id.

    The function performs a forward chain scan once to locate the
    subject, then a second scan via ``find_reliability_scores`` to
    locate the most recent reliability_score for the subject's
    target. Both scans are linear; single-machine + bounded chain
    keeps them cheap.

    Returns the typed result (never raises on missing substrate).
    """
    if not isinstance(subject_run_id, str) or not subject_run_id.strip():
        raise ValueError("subject_run_id must be a non-empty string")
    if not isinstance(audit_path, Path):
        raise TypeError("audit_path must be a Path")

    result: Dict[str, Any] = {
        "eligibility_passed":            False,
        "subject_run_id":                subject_run_id,
        "subject_evidence_kind":         None,
        "subject_target_canonical_id":   None,
        "reliability_score_run_id":      None,
        "reliability_score_at_decision": None,
        "reliability_score_floor":       RELIABILITY_SCORE_FLOOR_FOR_PROMOTION,
        "refusal_reasons":               [],
        "consulted_run_ids":             [],
        "eligibility_version":           GOVERNANCE_ELIGIBILITY_VERSION,
    }

    record = find_record_by_run_id(audit_path, subject_run_id)
    if record is None:
        result["refusal_reasons"].append("subject_not_found")
        return result
    event = record.get("event", {}) or {}
    kind = event.get("evidence_kind")
    result["subject_evidence_kind"] = kind
    result["consulted_run_ids"].append(subject_run_id)

    if kind != "threshold_recommendation":
        result["refusal_reasons"].append("subject_unsupported")
        return result

    if event.get("recommended_value") is None:
        result["refusal_reasons"].append("subject_is_refusal")
        return result

    target = event.get("target_canonical_id")
    result["subject_target_canonical_id"] = target
    if target not in CALIBRATION_TARGETS:
        result["refusal_reasons"].append("target_unsupported")
        return result

    # RECOMMENDATION-INSTANCE SCOPING (load-bearing): filter
    # scores by ``target_run_id == subject_run_id``, not just by
    # the shared target_canonical_id. A score is "about" a
    # recommendation iff it cites that recommendation's run_id as
    # its target. Scores for prior recommendations on the same
    # target do NOT transfer. Chain order is append, so the LAST
    # matching row is the most recent reliability_score for THIS
    # recommendation instance.
    scores = find_reliability_scores(
        audit_path=audit_path,
        target_canonical_id=target,
    )
    matching = [s for s in scores if s.get("target_run_id") == subject_run_id]
    if not matching:
        result["refusal_reasons"].append("no_reliability_score")
        return result
    latest = matching[-1]
    score_run_id = latest.get("run_id")
    result["reliability_score_run_id"] = score_run_id
    if score_run_id:
        result["consulted_run_ids"].append(score_run_id)

    overall_score = latest.get("overall_score")
    result["reliability_score_at_decision"] = overall_score
    if overall_score is None:
        result["refusal_reasons"].append("reliability_score_is_refusal")
        return result

    if overall_score < RELIABILITY_SCORE_FLOOR_FOR_PROMOTION:
        result["refusal_reasons"].append("reliability_below_floor")
        return result

    # All checks passed.
    for reason in result["refusal_reasons"]:
        _validate_refusal_reason(reason)
    result["eligibility_passed"] = True
    return result


def find_latest_calibration_for_subject(
    subject_event: Dict[str, Any],
) -> Optional[str]:
    """Return the calibration_report run_id the subject derives
    from, if recorded. Convenience used by the runner to fill
    ``evidence_basis``. Returns None when the field is absent."""
    cal = subject_event.get("derived_from_calibration_report_run_id")
    if isinstance(cal, str) and cal.strip():
        return cal
    return None
