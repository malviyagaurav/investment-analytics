"""Reliability scoring — Step 14.

Activates ``reliability_score`` as a first-class evidence kind: a
typed aggregation over existing evidence that scores epistemic
credibility for a target's claims, without mutating production
state or auto-promoting any threshold.

## Scope (day-one)

ONE target evidence_kind: ``threshold_recommendation``.

EIGHT closed scoring dimensions, each a pure function over typed
chain evidence:

  evidence_stability           — supersedes density on recent rows
  replay_stability             — weighted replay-state histogram with
                                 ASYMMETRIC penalty (invalid_replay
                                 hard-penalized, unreproducible
                                 soft-penalized)
  regime_sensitivity           — CV of per-regime recommendation values
  calibration_coverage_quality — high/medium coverage_quality fraction
  drift_exposure               — regime_transition density
  supersession_churn           — supersedes density across target +
                                 underlying regimes
  refusal_frequency            — calibration refusal density
  methodology_volatility       — distinct methodology snapshots

## Load-bearing controls

  - Aggregation = engineered weighted average (Option A);
    weights versioned under ``reliability_weighting`` methodology.
  - Refusal floor K=4: if ≥4 dimensions refuse, aggregate refuses.
  - Refusal IS evidence — refusal payloads carry the SAME shape as
    recommendation payloads (symmetric work per Step 11/13).
  - Asymmetric replay_state weighting: invalid_replay = -1.0,
    unreproducible = +0.5, others = +1.0. Chain-integrity hits
    cannot hide inside benign substrate-decay statistics.
  - NO production mutation. NO threshold_recommendation status
    transitions. NO auto-promotion. Step 15's domain.

## Anti-evidence-laundering posture

Every score carries its raw basis (counts/ratios) in the payload.
Every dimension lists the run_ids it consulted. The aggregate cites
its full ``derived_from_run_ids`` set. A reader can verify the math
by inspection — no opaque heuristic.
"""
from __future__ import annotations

from backend.reliability.config import (
    DEFAULT_SCORING_WINDOW_DAYS,
    K_REFUSAL_FLOOR,
    RELIABILITY_REFUSAL_REASONS,
    RELIABILITY_SCORE_SCHEMA_VERSION,
    RELIABILITY_WEIGHTING_VERSION,
    REPLAY_STATE_WEIGHTS,
    SCORING_DIMENSIONS,
    WEIGHTING_TABLE_THRESHOLD_RECOMMENDATION,
)
from backend.reliability.dimensions import (
    DIMENSION_FUNCTIONS,
    DimensionResult,
    calibration_coverage_quality,
    drift_exposure,
    evidence_stability,
    methodology_volatility,
    refusal_frequency,
    regime_sensitivity,
    replay_stability,
    supersession_churn,
)
from backend.reliability.runner import (
    DEFAULT_AUDIT_PATH,
    find_reliability_scores,
    score_target,
)

__all__ = [
    "DEFAULT_AUDIT_PATH",
    "DEFAULT_SCORING_WINDOW_DAYS",
    "DIMENSION_FUNCTIONS",
    "DimensionResult",
    "K_REFUSAL_FLOOR",
    "RELIABILITY_REFUSAL_REASONS",
    "RELIABILITY_SCORE_SCHEMA_VERSION",
    "RELIABILITY_WEIGHTING_VERSION",
    "REPLAY_STATE_WEIGHTS",
    "SCORING_DIMENSIONS",
    "WEIGHTING_TABLE_THRESHOLD_RECOMMENDATION",
    "calibration_coverage_quality",
    "drift_exposure",
    "evidence_stability",
    "find_reliability_scores",
    "methodology_volatility",
    "refusal_frequency",
    "regime_sensitivity",
    "replay_stability",
    "score_target",
    "supersession_churn",
]
