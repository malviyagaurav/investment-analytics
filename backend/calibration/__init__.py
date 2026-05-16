"""Calibration engine — Step 11.

Activates ``calibration_report`` as a first-class evidence kind:
per-regime weighted-percentile recommendations for engineered
thresholds, with symmetric typed refusals when substrate is
insufficient.

## Phase 2 anchor

Step 11 is the inflection point where the system transitions from
"structurally rigorous + logically disciplined" to "empirically
grounded." Engineered thresholds become measured distributions —
but ONLY through evidence-bearing, regime-partitioned, replayable
analysis. A calibration_report is INFERENCE FROM EVIDENCE, not
evidence itself.

## Load-bearing constraints

  - NEVER mutates production thresholds. A calibration_report is a
    typed proposal. Promotion is Step 15's domain.
  - NEVER pools across regimes. Top-level recommendation = median
    of per-regime recommendations. Pooling would produce the
    compression artifact Step 10 was designed to prevent.
  - NEVER auto-promotes from data_driven_variant. The methodology_kind
    label exists to be honored by downstream consumers.
  - NEVER omits the calibration_basis on refusal. Refusal is symmetric
    work to recommendation. The schema invariant is enforced at
    construction.

## Anti-evidence-laundering controls

  - REFUSAL_REASONS closed enum — no free-text refusal rationales
  - CALIBRATION_TARGETS closed enum — no untyped calibration targets
  - calibration_scope.valid_within_regimes non-empty ↔ recommendation
    non-null (anti-universalization)
  - calibration_basis records observation_count, effective_weight,
    coverage_quality_mix per regime even on refusal
  - derived_from_run_ids cites the regime_summary rows the
    calibration was built on; replay surfaces
    regime_dependency_superseded when those rows have been
    superseded by newer methodology

## Scope (day-one, Step 11)

ONE target: HIGH_CORRELATION_THRESHOLD on portfolio_health.
ONE weighting scheme: classification_confidence-weighted.
ONE aggregation rule: median across per-regime recommendations.
ONE engine version: v1, rule-based (sort + weighted percentile).

No ML, no optimization, no fitting, no black-box statistics. The
entire engine is transparent and replayable.
"""
from __future__ import annotations

from backend.calibration.config import (
    ACCEPTED_COVERAGE_QUALITIES,
    CALIBRATION_ENGINE_VERSION,
    CALIBRATION_EQUITY_UNIVERSE_CATEGORIES,
    CALIBRATION_REPORT_SCHEMA_VERSION,
    CALIBRATION_TARGETS,
    DEFAULT_PERCENTILE_BY_TARGET,
    MIN_EFFECTIVE_WEIGHT_PER_REGIME,
    MIN_FUND_HISTORY_POINTS_FOR_CALIBRATION,
    MIN_OBSERVATIONS_PER_REGIME,
    REFUSAL_REASONS,
)
from backend.calibration.percentiles import (
    unweighted_median,
    weighted_median,
    weighted_percentile,
)
from backend.calibration.runner import (
    find_calibration_reports,
    run_calibration,
)
from backend.calibration.sampling import (
    RegimeBucket,
    assemble_per_regime_samples,
    passes_coverage_floors,
)
from backend.calibration.targets import (
    CalibrationTarget,
    DEFAULT_CACHE_DIR,
    REGISTERED_TARGETS,
    get_target,
)

__all__ = [
    "ACCEPTED_COVERAGE_QUALITIES",
    "CALIBRATION_ENGINE_VERSION",
    "CALIBRATION_EQUITY_UNIVERSE_CATEGORIES",
    "CALIBRATION_REPORT_SCHEMA_VERSION",
    "CALIBRATION_TARGETS",
    "CalibrationTarget",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_PERCENTILE_BY_TARGET",
    "MIN_EFFECTIVE_WEIGHT_PER_REGIME",
    "MIN_FUND_HISTORY_POINTS_FOR_CALIBRATION",
    "MIN_OBSERVATIONS_PER_REGIME",
    "REFUSAL_REASONS",
    "REGISTERED_TARGETS",
    "RegimeBucket",
    "assemble_per_regime_samples",
    "find_calibration_reports",
    "get_target",
    "passes_coverage_floors",
    "run_calibration",
    "unweighted_median",
    "weighted_median",
    "weighted_percentile",
]
