"""Research-artifact configuration: closed enums + schema versions.

Step 13's single concern is ``threshold_recommendation`` — a typed,
lineage-linked projection of a ``calibration_report.recommendation``
field into its own evidence row. The other artifacts named in
Step 13's kickoff (calibration_report, drift_analysis,
experiment_run, replay_result) are ALREADY first-class evidence
kinds from earlier steps; the audit confirmed they meet the
"durable, evidence-bearing, lineage-linked, replayable, queryable"
contract.

## Why a separate evidence_kind for a recommendation

A calibration_report records one act of inference: "given this
regime substrate, the p95 of these samples is X." Its recommendation
field is the OUTPUT of that inference.

A threshold_recommendation records a PROMOTION CANDIDATE: "the
system currently proposes X as the threshold value for target Y,
within scope Z, derived from calibration_report W." It has its own
lifecycle (proposed → under_review → adopted/rejected/superseded),
its own supersedes chain across methodology updates, and is what
Step 15's promotion gating will read.

Splitting these two concepts:
  * keeps calibration_report immutable / one-shot
  * lets the recommendation accumulate adoption_status history
  * gives Step 15 a clean typed surface to query
  * keeps the "no auto-promotion" boundary explicit at the type
    level (calibration emits inference; recommendation emits
    promotion candidacy)

## Closed enums

  ADOPTION_STATUSES — fixed at 5 values. Day-one Step 13 emits ONLY
  "proposed"; the other four exist for Step 15's promotion lifecycle.
  Enumerating all of them now means Step 15 does NOT need to extend
  the enum (which would shift envelope_schema_fingerprint).
"""
from __future__ import annotations


THRESHOLD_RECOMMENDATION_SCHEMA_VERSION = "v1"


# Closed enum. Day-one emits only "proposed". The remaining values
# exist so Step 15's promotion lifecycle does NOT need to extend
# the enum at a later date — extension would shift
# envelope_schema_fingerprint and force replay re-classification of
# every prior recommendation row.
ADOPTION_STATUSES: frozenset = frozenset({
    "proposed",       # initial state — emitted from a calibration_report
    "under_review",   # operator / Step 15 shadow runner is evaluating
    "adopted",        # promotion happened — production threshold matches
    "rejected",       # operator / Step 15 decided NOT to adopt
    "superseded",     # replaced by a newer threshold_recommendation
})


# Step 13 itself only emits "proposed". Subsequent transitions
# (under_review, adopted, rejected, superseded) are Step 15's
# domain. Used by the runner to refuse forward-status emissions
# until that machinery exists, keeping the no-auto-promotion
# boundary explicit at the type level.
STEP_13_PERMITTED_EMIT_STATUSES: frozenset = frozenset({"proposed"})
