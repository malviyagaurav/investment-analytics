"""Methodology version registry.

Per-component versioned identifiers stored inside every audit
event's envelope under ``methodology``. The audit record's hash
includes these versions, so the version snapshot at write time is
permanently bound to the decision that was recorded.

## Bump rules

MUST bump a component's version (vN → v(N+1)):
  - Threshold change affecting classification (e.g. confidence
    5y → 6y, coverage 70% → 65%).
  - Formula change (adding/removing a metric, altering a formula,
    swapping a benchmark proxy default).
  - Tie-breaker rule change in dominance ordering.
  - Gate/filter change that alters what surfaces (material-
    improvement threshold, correlation threshold).
  - Bug fix where the fix changes outputs (old runs were
    effectively under a different methodology; record that fact).

MUST NOT bump:
  - Behavior-preserving refactor (renames, file moves, package
    splits, type hints).
  - Comment/docstring/log changes.
  - Performance changes that produce identical outputs (caching,
    vectorization, parallel speedups).
  - Test additions or test-only fixture changes.
  - Schema-version bumps to the response wire format that don't
    change underlying decisions.

## Composability

Each component is independent. A ``coverage_integrity`` bump does
NOT require ``equity_metric`` to bump. Audit records carry the full
versions dict so downstream replay knows exactly which combination
was active.

## Schema versioning

``methodology_schema_version`` versions the SHAPE of the
methodology dict itself, separately from individual component
versions. Bumped when:
  - A component is added or removed (granularity changes).
  - The methodology dict gains/loses top-level metadata.
  - The component naming convention changes.

This separation matters for replay tooling later: replay can
distinguish "different versions of the same component" from
"different structural understanding of what a methodology IS."
"""
from __future__ import annotations


# v2: added "regime_classifier" component (Step 10).
# v3: added "calibration_engine" component (Step 11).
# v4: added "reliability_weighting" component (Step 14). Per the
# schema-version bump rules above, adding a component bumps the
# schema version. Existing audit rows are byte-untouched; new rows
# carry the v4 dict and a new ENVELOPE_SCHEMA_FINGERPRINT, surfacing
# the structural change as explicit drift on replay.
METHODOLOGY_SCHEMA_VERSION = "v4"


# Source of truth. Each component is independently versioned per the
# bump rules above. Adding a new component or removing one bumps
# METHODOLOGY_SCHEMA_VERSION as well.
METHODOLOGY_VERSIONS: dict[str, str] = {
    "equity_metric":         "v1",
    "debt_metric":           "v1",
    "gold_metric":           "v1",
    "confidence":            "v1",
    "coverage_integrity":    "v1",
    "alternative_gate":      "v1",
    "correlation_detection": "v1",
    "decision_engine":       "v1",
    "regime_classifier":     "v1",
    "calibration_engine":    "v1",
    "reliability_weighting": "v1",
}


def current_methodology() -> dict[str, str]:
    """Return a fresh snapshot of the methodology versions with the
    methodology-schema version included.

    The result is a COPY — callers cannot mutate the source of truth
    by mutating the return value. This is load-bearing: the envelope
    embeds this dict into each audit record, and the record's hash
    must reflect the methodology that was ACTIVE at write time. If
    callers shared a reference, a later mutation would change the
    apparent methodology of already-written records.
    """
    snapshot = {"methodology_schema_version": METHODOLOGY_SCHEMA_VERSION}
    snapshot.update(METHODOLOGY_VERSIONS)
    return snapshot
