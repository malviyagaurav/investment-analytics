"""Operator identity capture.

Operator identity is a typed claim about WHO authorized the
decision. Single-machine, single-operator reality means we cannot
verify identity cryptographically at this layer — but we CAN
capture the operator's typed declaration, the typed role, and the
attestation method, and bind all three into the audit-hashed
envelope so the decision is forensically attributable.

## What this module does not do

It does not authenticate. It does not verify signatures. It does
not check that the operator_id matches an OS user. Those guarantees
require a different deployment posture (mTLS, hardware key,
remote audit broker). What this module guarantees is that the
operator's typed declaration is captured at the moment of decision
and bound into the hash chain — so any later attempt to rewrite
WHO approved breaks the chain.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.governance.config import (
    ATTESTATION_METHODS,
    OPERATOR_ROLES,
)


def build_operator_identity(
    operator_id: str,
    operator_role: str,
    attestation_method: str = "local_terminal",
    supporting_evidence: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build the operator_identity block embedded in every
    governance_decision payload.

    Args:
      operator_id:         typed declaration of WHO is approving
                           (e.g. "owner@local"). Non-empty string.
      operator_role:       one of OPERATOR_ROLES. ``automation`` is
                           NOT permitted for approve / reject /
                           rollback — those require a human.
      attestation_method:  one of ATTESTATION_METHODS. Day-one
                           default is local_terminal.
      supporting_evidence: optional list of free-text breadcrumbs
                           (ticket IDs, meeting notes, etc.). Kept
                           as a list of strings so each entry is
                           individually addressable on replay.

    Raises ValueError for unknown role / method, or empty
    operator_id.
    """
    if not isinstance(operator_id, str) or not operator_id.strip():
        raise ValueError(
            "operator_id must be a non-empty string declaring who is "
            "authorizing the decision"
        )
    if operator_role not in OPERATOR_ROLES:
        raise ValueError(
            f"operator_role must be one of {sorted(OPERATOR_ROLES)}, "
            f"got {operator_role!r}"
        )
    if attestation_method not in ATTESTATION_METHODS:
        raise ValueError(
            f"attestation_method must be one of {sorted(ATTESTATION_METHODS)}, "
            f"got {attestation_method!r}"
        )
    evidence_list: List[str] = []
    if supporting_evidence:
        for entry in supporting_evidence:
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError(
                    "supporting_evidence entries must be non-empty strings"
                )
            evidence_list.append(entry.strip())

    return {
        "operator_id":         operator_id.strip(),
        "operator_role":       operator_role,
        "attestation_method":  attestation_method,
        "supporting_evidence": evidence_list,
    }
