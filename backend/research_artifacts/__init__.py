"""Research-artifact layer — Step 13.

Activates ``threshold_recommendation`` as a first-class evidence
kind: a typed, lineage-linked projection of a
``calibration_report``'s recommendation into a promotion-candidate
artifact with its own lifecycle status.

## Why this layer exists separately from calibration

A ``calibration_report`` (Step 11) is one act of inference: "given
this regime substrate, the p95 of these samples is X." Its
recommendation FIELD is the output of that inference.

A ``threshold_recommendation`` is a PROMOTION CANDIDATE: "the system
currently proposes value X for target Y, within scope Z, derived
from calibration_report W." It has its own adoption_status
lifecycle (proposed → under_review → adopted/rejected/superseded),
its own supersedes chain across methodology updates, and is what
Step 15's promotion gating will read.

The split keeps calibration_report immutable / one-shot while
letting the recommendation accumulate adoption_status history.

## Load-bearing refusal-projection guard

emit AND replay both refuse to project a ``threshold_recommendation``
from a calibration_report whose ``recommendation`` field is null
(a typed refusal). This is the canonical anti-evidence-laundering
control: a typed wrapper cannot turn a non-recommendation into a
recommendation, at any layer.

## Scope (day-one)

  - ONE new evidence_kind: ``threshold_recommendation``
  - ONE permitted emit status: ``"proposed"``
  - Forward statuses (under_review/adopted/rejected/superseded)
    exist in ADOPTION_STATUSES so Step 15 does not need to extend
    the enum, but Step 13 itself refuses to emit them.
  - Linear acyclic supersedes chain (same discipline as Step 10
    regime_summary).
  - NO methodology component added.
  - NO METHODOLOGY_SCHEMA_VERSION bump.
  - NO mutation of production thresholds or any prior evidence row.

The other artifacts named in Step 13's kickoff (calibration_report,
drift_analysis, experiment_run, replay_result) are ALREADY first-
class evidence-bearing from earlier steps; Step 13's audit
confirmed they already meet the durability contract.
"""
from __future__ import annotations

from backend.research_artifacts.config import (
    ADOPTION_STATUSES,
    STEP_13_PERMITTED_EMIT_STATUSES,
    THRESHOLD_RECOMMENDATION_SCHEMA_VERSION,
)
from backend.research_artifacts.runner import (
    DEFAULT_AUDIT_PATH,
    emit_threshold_recommendation,
    find_threshold_recommendations,
)

__all__ = [
    "ADOPTION_STATUSES",
    "DEFAULT_AUDIT_PATH",
    "STEP_13_PERMITTED_EMIT_STATUSES",
    "THRESHOLD_RECOMMENDATION_SCHEMA_VERSION",
    "emit_threshold_recommendation",
    "find_threshold_recommendations",
]
