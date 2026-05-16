"""Governance closed enums + day-one defaults.

Step 15 is the human-decision evidence layer. Every governance
decision (approve, reject, request_review, rollback) becomes one
typed ``governance_decision`` audit row + by-reference evidence
file. The layer NEVER mutates production state — production
mutation is a separate, explicit, code-edit operation outside the
chain. What governance records is the OPERATOR'S DECLARATION about
production, not production itself.

## Anti-laundering controls in this file

  - DECISION_TYPES is closed (4 values, no free additions).
  - ELIGIBILITY_REFUSAL_REASONS is closed.
  - OPERATOR_ROLES is closed (single-machine, single-operator
    reality; the typed role is still recorded for replay).
  - ATTESTATION_METHODS is closed — local_terminal is the only
    day-one value; future remote / mTLS / hardware-key paths
    extend the enum and bump the schema_version.
  - OVERRIDE_REASONS is closed (tightening #1 from review): a
    forced approval is NOT merely "eligibility failed"; it is a
    distinct typed governance claim that must survive replay.
    Collapsing override semantics into eligibility-bypass would
    let operator intent vanish from the audit surface.
  - RELIABILITY_SCORE_FLOOR_FOR_PROMOTION is the engineered
    eligibility threshold. Bumping it bumps the
    ``governance_eligibility`` methodology component (registered
    in methodology.py), which surfaces on replay as
    ``methodology_changed``.

## What governance_decision does NOT do

  - It does not mutate production constants (e.g.
    HIGH_CORRELATION_THRESHOLD remains byte-unchanged).
  - It does not auto-promote a threshold_recommendation.
  - It does not retroactively rewrite prior decisions —
    supersession is via an explicit ``supersedes_run_id`` link.
  - It does not silently fail eligibility — the
    ``--force-ineligible`` path is explicit AND records an
    override_attestation.
"""
from __future__ import annotations


GOVERNANCE_DECISION_SCHEMA_VERSION = "v1"
GOVERNANCE_ELIGIBILITY_VERSION = "v1"


# Closed enum. Every governance decision falls into exactly one of
# these four types. Adding a type requires bumping the schema
# version and registering replay handler behavior.
DECISION_TYPES: frozenset = frozenset({
    "approve",         # operator declares the recommendation adopted
    "reject",          # operator declares the recommendation rejected
    "request_review",  # operator parks the recommendation under review
    "rollback",        # operator rolls back a prior approval
})


# Closed enum. Each value is a typed reason eligibility refused.
# The eligibility check is a PURE function over chain state; these
# enumerate the exhaustive set of refusal modes it can return.
#
# CRITICAL FRAMING: eligibility is RECOMMENDATION-INSTANCE scoped,
# NOT target-scoped. A single target_canonical_id may have many
# threshold_recommendations over its lifetime — each one is its
# own subject and earns its own eligibility evaluation against
# the reliability_score rows whose ``target_run_id`` matches it.
# A previous recommendation passing eligibility does NOT propagate
# to a newer recommendation; that would let a stale endorsement
# silently authorize a fresh artifact.
#
#   subject_not_found            — cited threshold_recommendation
#                                  run_id is absent from the chain
#   subject_unsupported          — cited row is not a
#                                  threshold_recommendation
#   subject_is_refusal           — cited threshold_recommendation
#                                  is itself a typed refusal
#                                  (recommended_value=null)
#   no_reliability_score         — no reliability_score row exists
#                                  FOR THIS RECOMMENDATION INSTANCE
#                                  (i.e., no row whose target_run_id
#                                  matches the cited subject_run_id —
#                                  scores attached to OTHER prior
#                                  recommendations for the same
#                                  target do NOT satisfy this)
#   reliability_score_is_refusal — the latest reliability_score for
#                                  this RECOMMENDATION INSTANCE is
#                                  itself a refusal (overall_score=null)
#   reliability_below_floor      — latest score for this instance
#                                  < RELIABILITY_SCORE_FLOOR_FOR_PROMOTION
#   target_unsupported           — target_canonical_id not in
#                                  CALIBRATION_TARGETS (target-level
#                                  concern: the system has no calibration
#                                  surface registered for this target
#                                  at all)
ELIGIBILITY_REFUSAL_REASONS: frozenset = frozenset({
    "subject_not_found",
    "subject_unsupported",
    "subject_is_refusal",
    "no_reliability_score",
    "reliability_score_is_refusal",
    "reliability_below_floor",
    "target_unsupported",
})


# Closed enum. Single-machine, single-operator reality — but the
# typed role surfaces on replay and downstream audit so future
# multi-operator regimes don't require schema extension to
# distinguish humans from automated approval simulations.
OPERATOR_ROLES: frozenset = frozenset({
    "owner",       # the deployment operator (default day-one role)
    "reviewer",    # secondary reviewer, advisory only
    "automation",  # NOT for production approvals — emits
                   # request_review only; reserved for future
                   # batched-evaluation runs that should never
                   # adopt into production unattended.
})


# Closed enum. How the operator's identity was attested at decision
# time. Day-one: only local_terminal (the operator was at the
# laptop, typing the CLI command). Future attestation methods
# (hardware key, mTLS, remote signed payload) extend this enum
# and bump the schema version — making attestation policy itself
# part of the replay-visible drift surface.
ATTESTATION_METHODS: frozenset = frozenset({
    "local_terminal",
})


# Closed enum (TIGHTENING #1 — review directive). A forced
# approval, request_review, or rollback that bypasses eligibility
# is not merely "the check failed and we proceeded anyway." It is
# a distinct governance claim by the operator that the eligibility
# result was insufficient grounds to block the decision.
#
#   operator_judgment            — operator overrides on subjective
#                                  / domain-specific grounds not
#                                  encoded in eligibility policy.
#   reliability_pipeline_fault   — operator overrides because the
#                                  reliability scoring pipeline is
#                                  known-broken (substrate gap,
#                                  emit-side defect under repair).
#   temporary_substrate_gap      — operator overrides because the
#                                  substrate is known thin during
#                                  bootstrap / re-baselining and
#                                  the eligibility floor would
#                                  permanently block all decisions.
#
# Adding a reason requires a governance review — the enum is
# deliberately small because the override path is the ONE place
# typed system policy can be bypassed by a human, and the
# vocabulary must remain bounded.
OVERRIDE_REASONS: frozenset = frozenset({
    "operator_judgment",
    "reliability_pipeline_fault",
    "temporary_substrate_gap",
})


# Engineered eligibility floor for the reliability_score. Below
# this number the system refuses to consider the recommendation
# eligible for approval — the operator can still --force-ineligible
# but must attest with a typed OVERRIDE_REASONS value.
#
# Day-one floor: 0.50. Reasoning: scores below 0.50 mean MORE THAN
# HALF the weighted dimensions either refused or scored low; the
# claim's epistemic backing is materially thinner than the
# alternative of "no recommendation at all." Bumping this value
# bumps GOVERNANCE_ELIGIBILITY_VERSION, surfacing on replay.
RELIABILITY_SCORE_FLOOR_FOR_PROMOTION = 0.50
