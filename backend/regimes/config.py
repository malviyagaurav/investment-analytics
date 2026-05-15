"""Regime taxonomy + classifier configuration.

Closed enum + versioned defaults for the day-one rule-based
classifier. Step 11 calibration will produce data-driven cut points
that bump ``regime_classifier`` from v1 → v2 (a methodology-version
bump, NOT a taxonomy bump). Taxonomy changes (renaming a class,
adding/removing a class, changing class semantics) bump
``REGIME_TAXONOMY_VERSION`` independently — those are two different
drift surfaces.

## Why two version axes

The classifier methodology and the regime taxonomy can drift
independently:

  - Calibrating new vol bands while keeping the same classes →
    bumps ``regime_classifier`` to v2, taxonomy stays v1.
  - Renaming ``crisis_vol`` → ``extreme_vol`` while keeping the
    math unchanged → bumps taxonomy to v2, classifier stays v1.
  - Adding ``trending_up`` as a new class with new math → bumps
    both.

Treating them as one version would hide which kind of change
happened. Replay/calibration/reliability tooling can compare each
axis independently and surface the right diagnostic.
"""
from __future__ import annotations

from dataclasses import dataclass


# Closed enum. Adding a class is a taxonomy bump (REGIME_TAXONOMY_VERSION).
#
# Day-one membership (Step 10): vol-only classes derived from realized
# annualized volatility on Nifty 50. Macro-causal classes
# (tightening_cycle, inflationary_rotation, etc.) are NOT reserved —
# we don't have rate/macro signals typed yet, and pre-reserving names
# we can't honestly populate is itself an evidence-laundering vector.
REGIME_CLASSES: frozenset = frozenset({
    "low_vol",        # ann. vol < low_threshold
    "normal_vol",     # low_threshold ≤ ann. vol < high_threshold
    "high_vol",       # high_threshold ≤ ann. vol < crisis_threshold
    "crisis_vol",     # ann. vol ≥ crisis_threshold
    "indeterminate",  # confidence state — insufficient coverage to classify
})


# Classes that participate in regime transitions. indeterminate is a
# CONFIDENCE state, not a market state — treating it as a regime would
# pollute drift statistics. The drift module enforces this.
DETERMINATE_REGIME_CLASSES: frozenset = REGIME_CLASSES - {"indeterminate"}


# Taxonomy version. Bumped on any change to REGIME_CLASSES membership
# OR to the documented semantics of a class. Independent of the
# classifier methodology version (which lives in METHODOLOGY_VERSIONS
# under "regime_classifier").
REGIME_TAXONOMY_VERSION = "v1"


# Schema version for the regime_summary / drift_analysis payloads
# themselves (the wire shape, separately from the taxonomy semantics).
REGIME_SUMMARY_SCHEMA_VERSION = "v1"
DRIFT_ANALYSIS_SCHEMA_VERSION = "v1"


# Classification semantics — formal anti-overinterpretation control.
# Surfaces on every regime_summary payload so downstream layers
# (calibration, reliability scoring, eventual narrative generation)
# cannot silently treat realized-vol bands as causal claims about
# economic state.
CLASSIFICATION_SEMANTICS = "descriptive_not_causal"


@dataclass(frozen=True)
class ClassifierParams:
    """Engineered cut points for the v1 rule-based classifier.

    These are REASONED defaults, not calibrated. The classifier emits
    them inline (via ``classification_basis.applied_bands``) on every
    regime_summary so a future replay sees exactly which thresholds
    were active at classification time. Step 11 calibration will
    propose data-driven replacements; promoting them bumps
    ``regime_classifier`` to v2.
    """
    signal_scheme_code:    int   = 120716  # UTI Nifty 50 Index (deepest history)
    low_threshold_pct:     float = 12.0
    high_threshold_pct:    float = 20.0
    crisis_threshold_pct:  float = 35.0
    min_coverage_days:     int   = 60      # below this → indeterminate
    boundary_confidence_margin_pct: float = 10.0
    # ↑ a window's vol within ±N% of a band boundary erodes its
    # classification_confidence — encodes the "barely crossed vs
    # decisively crossed" distinction so calibration later doesn't
    # treat edge cases as authoritative.


DEFAULT_CLASSIFIER_PARAMS = ClassifierParams()


# Coverage-quality bands derived from history depth + missingness.
# Used in regime_summary.signal_quality.coverage_quality.
#
#   high:   5+ years of history AND <8% missing in the window
#   medium: 2+ years of history AND <20% missing in the window
#   low:    everything else (still classifiable, but downstream
#           weighting should reflect lower epistemic standing)
#
# All counts are in TRADING-DAY OBSERVATIONS. The "missing %" cutoffs
# account for NSE's ~13 holidays/year (~5% of weekdays) given the
# day-one weekday-only approximation — a clean window with full
# attendance baselines at ~5% "missing" under that approximation.
# When a real Indian-market calendar is plugged in later, these
# cutoffs can be tightened (a methodology bump under the
# regime_classifier component would surface the change on replay).
HIGH_COVERAGE_HISTORY_DAYS    = 252 * 5
MEDIUM_COVERAGE_HISTORY_DAYS  = 252 * 2
HIGH_COVERAGE_MISSING_PCT     = 8.0
MEDIUM_COVERAGE_MISSING_PCT   = 20.0
