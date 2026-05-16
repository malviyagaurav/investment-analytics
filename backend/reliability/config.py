"""Reliability scoring — closed enums + weighting tables.

Step 14 is a typed AGGREGATION layer over existing evidence, not a
new inference layer. Every weight here is engineered, deterministic,
and versioned under the ``reliability_weighting`` methodology
component. Bumping a weight bumps the version; replay surfaces it
as a typed drift driver via the existing methodology-changed
machinery.

## Anti-laundering controls in this file

  - SCORING_DIMENSIONS is closed (8 values, no free additions).
  - RELIABILITY_REFUSAL_REASONS is closed.
  - REPLAY_STATE_WEIGHTS treats ``invalid_replay`` ASYMMETRICALLY
    relative to ``unreproducible``:
      - exact_match / semantically_equivalent / expected_divergence
        all count fully (1.0) — typed honest replay outcomes
      - unreproducible counts HALF (0.5) — substrate may have
        decayed or been archived; the original claim's epistemic
        plausibility is not impugned
      - invalid_replay counts NEGATIVELY (-1.0) — the claim's
        integrity cannot currently be justified; this is a real
        reliability hit, not a substrate gap
    The asymmetry is load-bearing: collapsing these into one
    penalty class lets chain-integrity violations hide inside
    benign substrate-decay statistics.
  - K_REFUSAL_FLOOR = 4: if half or more of the 8 dimensions
    individually refuse, the aggregate refuses too — refusing to
    average a score from substrates that are mostly absent.
"""
from __future__ import annotations


RELIABILITY_SCORE_SCHEMA_VERSION = "v1"
RELIABILITY_WEIGHTING_VERSION    = "v1"

DEFAULT_SCORING_WINDOW_DAYS = 90


# Closed enum. Each value maps to a pure-function in dimensions.py
# that returns a DimensionResult. Adding a dimension requires a
# bump of the ``reliability_weighting`` methodology component AND a
# corresponding pure-function implementation under governance review.
SCORING_DIMENSIONS: frozenset = frozenset({
    "evidence_stability",
    "replay_stability",
    "regime_sensitivity",
    "calibration_coverage_quality",
    "drift_exposure",
    "supersession_churn",
    "refusal_frequency",
    "methodology_volatility",
})


# Closed enum. Refusal IS evidence per the Step 11 pattern.
#
# Per Step 14 tightening review:
#   ``dimension_execution_failed`` is distinct from
#   ``insufficient_substrate``. The first means the scoring code
#   itself raised — a real defect, not an honest refusal. Collapsing
#   them into one bucket would launder implementation bugs as
#   epistemic refusals, violating the anti-laundering posture.
#   Operators inspecting a payload can distinguish "we tried and
#   the substrate wasn't there" from "we tried and our code broke."
RELIABILITY_REFUSAL_REASONS: frozenset = frozenset({
    "no_target_evidence",          # target has no recent rows
    "no_replay_history",           # no replay_result rows available
    "insufficient_substrate",      # honest: data not present / not enough
    "target_unsupported",          # target_canonical_id not registered
    "target_run_id_missing",       # target row not in chain
    "target_is_refusal",           # cited target is itself a refusal
    "all_dimensions_refused",      # every dimension refused individually
    "dimension_execution_failed",  # scoring code raised — implementation bug
})


# Day-one weighting table for threshold_recommendation targets.
# Engineered, not learned. Sums to 1.00. Versioned under
# ``reliability_weighting`` methodology component — bumping any
# weight bumps that version, surfaces on replay.
#
# Operationally: replay_stability + regime_sensitivity +
# calibration_coverage_quality are the heaviest because they speak
# most directly to "can the system justify this claim now?" —
# stability of past replays, robustness across regimes, and the
# epistemic depth of the underlying calibration substrate.
WEIGHTING_TABLE_THRESHOLD_RECOMMENDATION: dict = {
    "evidence_stability":           0.10,
    "replay_stability":             0.20,
    "regime_sensitivity":           0.20,
    "calibration_coverage_quality": 0.20,
    "drift_exposure":               0.10,
    "supersession_churn":           0.05,
    "refusal_frequency":            0.10,
    "methodology_volatility":       0.05,
}


# ── Asymmetric replay-state weighting (the Step 14 tightening) ──────
#
# Per the Step 14 governance directive:
#   invalid_replay must heavily penalize replay_stability;
#   unreproducible should penalize less aggressively and expose
#   the cause in basis.
#
# These weights are applied per-replay-row when computing the
# replay_stability dimension. The dimension's basis records the
# raw count by state PLUS the unreproducible_causes histogram so
# operators can see WHY substrate has decayed.
REPLAY_STATE_WEIGHTS: dict = {
    "exact_match":            1.0,
    "semantically_equivalent": 1.0,
    "expected_divergence":     1.0,   # typed driver explained — honest drift
    "unreproducible":          0.5,   # soft penalty; substrate decay
    "invalid_replay":         -1.0,   # HARD penalty; chain integrity hit
}


# Aggregate refusal threshold. If K_REFUSAL_FLOOR or more of the 8
# dimensions refuse individually, the aggregate score refuses with
# RELIABILITY_REFUSAL_REASONS = "insufficient_substrate" (if no
# rows at all) or "all_dimensions_refused" (specific dimensions
# refused for typed reasons). Day-one floor: 4 (half of 8).
K_REFUSAL_FLOOR = 4


# Default substrate-availability floors for individual dimensions.
# Below these counts, the dimension refuses (returns score=None
# with a typed refusal_reason on the DimensionResult).
MIN_REPLAYS_FOR_REPLAY_STABILITY     = 3
MIN_RECOMMENDATIONS_FOR_CHURN        = 2
MIN_CALIBRATIONS_FOR_REFUSAL_FREQ    = 2
MIN_REGIMES_FOR_REGIME_SENSITIVITY   = 2
MIN_DRIFT_ROWS_FOR_EXPOSURE          = 1
DRIFT_EXPOSURE_TRANSITION_FLOOR      = 5   # transitions ≥ this drives score → 0
