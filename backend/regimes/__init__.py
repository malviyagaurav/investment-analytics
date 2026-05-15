"""Regime detection & drift analysis — Step 10.

Activates ``regime_summary`` and ``drift_analysis`` as first-class
evidence kinds. The day-one classifier is rule-based on Nifty 50
realized volatility (the only universe-level equity signal with
>10y of continuous history in the cache today).

## Phase 2 anchor

This is the partitioning layer that determines whether Step 11
calibration outputs are stationary, transferable, and
decision-relevant. Without typed regime context, calibration averages
ZIRP, COVID, inflation shock, and tightening cycles into a single
distribution and presents the result as "empirical."

## Governance controls (load-bearing for Phase 2)

  - REGIME_CLASSES is a closed enum — no free-text labels.
  - taxonomy_version and regime_classifier_version are independent
    drift axes (renaming a class is taxonomy drift; recalibrating
    band thresholds is classifier drift).
  - classification_semantics is hardcoded to "descriptive_not_causal"
    — a formal anti-overinterpretation control. Future layers cannot
    silently treat a realized-vol label as a causal economic claim.
  - signal_quality.coverage_quality surfaces the epistemic depth of
    the data so downstream calibration can weight strong vs weak
    classifications.
  - classification_confidence erodes near band boundaries — a "barely
    crossed" classification carries lower authority than a decisive
    one.
  - regime_summary rows are immutable: a methodology update that
    wants to re-classify a historical window must explicitly
    ``supersedes_run_id`` the prior claim. Both rows live forever.
  - drift_analysis honors indeterminate as a CONFIDENCE state, not a
    market state: transitions involving indeterminate sides are
    flagged with regime_transition=False to keep calibration
    statistics clean.

## Scope (day-one, Step 10)

  - ONE classifier methodology: rule-based vol bands on Nifty 50.
  - ONE drift metric: rolling_vol_shift between two windows.
  - 5-class taxonomy: low_vol / normal_vol / high_vol / crisis_vol
    / indeterminate.

Cross-segment regimes, rate-cycle / macro classes, and data-driven
band cut points are deferred. Step 11 calibration produces the
data-driven cut points; bumping the classifier methodology to v2
will surface as expected_divergence on replay of older rows.
"""
from __future__ import annotations

from backend.regimes.classifier import (
    RegimeClassification,
    classification_to_payload,
    classify_window,
)
from backend.regimes.config import (
    CLASSIFICATION_SEMANTICS,
    DEFAULT_CLASSIFIER_PARAMS,
    DETERMINATE_REGIME_CLASSES,
    DRIFT_ANALYSIS_SCHEMA_VERSION,
    REGIME_CLASSES,
    REGIME_SUMMARY_SCHEMA_VERSION,
    REGIME_TAXONOMY_VERSION,
    ClassifierParams,
)
from backend.regimes.drift import (
    DriftResult,
    compute_drift,
    drift_to_payload,
)
from backend.regimes.runner import (
    emit_drift_analysis,
    emit_regime_summary,
    find_regime_summaries,
)

__all__ = [
    "CLASSIFICATION_SEMANTICS",
    "ClassifierParams",
    "DEFAULT_CLASSIFIER_PARAMS",
    "DETERMINATE_REGIME_CLASSES",
    "DRIFT_ANALYSIS_SCHEMA_VERSION",
    "DriftResult",
    "REGIME_CLASSES",
    "REGIME_SUMMARY_SCHEMA_VERSION",
    "REGIME_TAXONOMY_VERSION",
    "RegimeClassification",
    "classification_to_payload",
    "classify_window",
    "compute_drift",
    "drift_to_payload",
    "emit_drift_analysis",
    "emit_regime_summary",
    "find_regime_summaries",
]
