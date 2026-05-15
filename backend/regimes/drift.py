"""Drift metric — rolling-vol shift between two regime windows.

Pure function. Composes two ``RegimeClassification`` objects (one per
window) into a typed drift result with explicit transition handling
and a magnitude-aware ``transition_confidence``.

## Indeterminate handling (tightening per Step 10 governance review)

A transition is recognized ONLY when BOTH windows resolve to a
determinate REGIME_CLASS. indeterminate is a *confidence* state, not
a market state — treating ``normal_vol → indeterminate`` or
``indeterminate → crisis_vol`` as a regime transition would
contaminate downstream calibration statistics with edge cases that
are actually missing-data artefacts.

When either side is indeterminate:
  - ``regime_transition`` is False (regardless of label difference).
  - ``transition_confidence`` is 0.0.
  - The drift row is still emitted (it's evidence of "we tried" and
    carries forensic value), but downstream layers can filter on the
    transition flag.

## transition_confidence — magnitude-aware

For a recognized transition:

    transition_confidence = min(
        min(a.classification_confidence, b.classification_confidence),
        magnitude_factor,
    )

where ``magnitude_factor`` measures how decisively each side sits
beyond the boundary that separates a's class from b's class —
normalized by ``boundary_confidence_margin_pct`` and saturated at 1.0.

Without this factor, ``19.99% → 20.01%`` (technical band crossing)
and ``8% → 40%`` (structural regime shift) would land at identical
transition_confidence whenever both windows happened to share equal
classification_confidence. That would let Step 11 calibration treat
threshold-edge jitter as authoritative regime evidence — a real
evidence-laundering vector.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from backend.regimes.classifier import RegimeClassification
from backend.regimes.config import (
    DETERMINATE_REGIME_CLASSES,
    DRIFT_ANALYSIS_SCHEMA_VERSION,
    REGIME_TAXONOMY_VERSION,
)


@dataclass(frozen=True)
class DriftResult:
    """Pure result of compute_drift. Carries enough provenance to be
    emitted as a drift_analysis evidence payload — including the
    taxonomy + classifier versions active at compute time so replay
    can detect (and explicitly classify) taxonomy / classifier drift
    on re-derivation."""
    drift_kind:                  str
    window_a:                    Dict[str, Any]
    window_b:                    Dict[str, Any]
    vol_delta_pct:               float
    regime_transition:           bool
    transition_confidence:       float
    magnitude_band:              str
    signal_kind:                 str
    taxonomy_version:            str = REGIME_TAXONOMY_VERSION
    regime_classifier_version:   str = "v1"
    schema_version:              str = DRIFT_ANALYSIS_SCHEMA_VERSION


def _magnitude_band(vol_delta_pct: float) -> str:
    """Coarse magnitude bucket. Calibration (Step 11) may replace
    these cut points; they are engineered defaults today."""
    abs_delta = abs(vol_delta_pct)
    if abs_delta < 2.0:
        return "minor"
    if abs_delta < 8.0:
        return "notable"
    return "regime_change"


_CLASS_ORDER = {
    "low_vol":    0,
    "normal_vol": 1,
    "high_vol":   2,
    "crisis_vol": 3,
}


def _crossed_boundaries(
    a_class: str, b_class: str, params_basis: dict,
) -> Tuple[Optional[float], Optional[float]]:
    """Return ``(a_side_boundary, b_side_boundary)`` — the band edges
    that separate a's class from the directionally adjacent class
    toward b, and symmetrically for b. Returns ``(None, None)`` when
    either class isn't in the ordered determinate set.

    For an upward transition (a's class index < b's class index),
    a's boundary is the UPPER edge of a's class; b's boundary is the
    LOWER edge of b's class. Reversed for downward transitions.
    """
    bands = params_basis.get("applied_bands") or {}
    ordered_thresholds = [
        bands.get("low_threshold_pct"),
        bands.get("high_threshold_pct"),
        bands.get("crisis_threshold_pct"),
    ]
    if None in ordered_thresholds:
        return (None, None)
    a_idx = _CLASS_ORDER.get(a_class)
    b_idx = _CLASS_ORDER.get(b_class)
    if a_idx is None or b_idx is None:
        return (None, None)
    if a_idx == b_idx:
        return (None, None)
    if b_idx > a_idx:
        # upward
        return (ordered_thresholds[a_idx], ordered_thresholds[b_idx - 1])
    # downward
    return (ordered_thresholds[a_idx - 1], ordered_thresholds[b_idx])


def _magnitude_factor(
    a: RegimeClassification, b: RegimeClassification,
) -> float:
    """Decisiveness of the transition — how far each side sits past
    the band boundary it has to cross. Bounded by the WEAKER side
    (the one closer to its boundary) and normalized by the
    boundary_confidence_margin_pct from the classification basis.
    Saturated at 1.0.

    A 19.99→20.01 crossing yields a tiny factor; an 8→40 crossing
    yields near 1.0.
    """
    vol_a = a.classification_basis.get("annualized_vol_pct")
    vol_b = b.classification_basis.get("annualized_vol_pct")
    margin = a.classification_basis.get("boundary_confidence_margin_pct")
    if vol_a is None or vol_b is None or not margin or margin <= 0:
        return 1.0  # no usable magnitude signal — defer to side-confidence
    a_boundary, b_boundary = _crossed_boundaries(
        a.regime_class, b.regime_class, a.classification_basis,
    )
    if a_boundary is None or b_boundary is None:
        return 1.0
    d_a = abs(vol_a - a_boundary)
    d_b = abs(vol_b - b_boundary)
    return min(1.0, min(d_a, d_b) / margin)


def _transition_confidence(
    a: RegimeClassification,
    b: RegimeClassification,
    is_transition: bool,
) -> float:
    """Confidence ∈ [0, 1] that the transition is real.

    Not-a-transition → 0.0.

    Transition: ``min(weaker_side_classification_confidence,
    magnitude_factor)``. Both must be decisive for the transition
    itself to be considered decisive.
    """
    if not is_transition:
        return 0.0
    weaker_side = min(a.classification_confidence,
                      b.classification_confidence)
    mag = _magnitude_factor(a, b)
    return round(min(weaker_side, mag), 4)


def compute_drift(
    a: RegimeClassification,
    b: RegimeClassification,
    *,
    drift_kind: str = "rolling_vol_shift",
    signal_kind: str = "nifty50_realized_vol",
) -> DriftResult:
    """Compose two classifications into a drift result.

    Both inputs must be RegimeClassification instances (already
    produced by classify_window). The drift module does NOT re-classify
    — it composes existing classifications, so the same windows always
    produce the same drift result given the same classifications.
    """
    vol_a = a.classification_basis.get("annualized_vol_pct") or 0.0
    vol_b = b.classification_basis.get("annualized_vol_pct") or 0.0
    vol_delta = round(vol_b - vol_a, 4)

    # Per the tightening: a transition only counts when BOTH windows
    # resolve to determinate classes AND the classes differ.
    both_determinate = (
        a.regime_class in DETERMINATE_REGIME_CLASSES
        and b.regime_class in DETERMINATE_REGIME_CLASSES
    )
    is_transition = both_determinate and a.regime_class != b.regime_class
    confidence = _transition_confidence(a, b, is_transition)

    def _window_dict(c: RegimeClassification) -> Dict[str, Any]:
        return {
            "start_date":              c.window_start_date,
            "end_date":                c.window_end_date,
            "regime_class":            c.regime_class,
            "annualized_vol_pct":      c.classification_basis.get(
                "annualized_vol_pct"),
            "classification_confidence": c.classification_confidence,
            "regime_summary_run_id":   None,  # filled by the runner
        }

    return DriftResult(
        drift_kind=drift_kind,
        window_a=_window_dict(a),
        window_b=_window_dict(b),
        vol_delta_pct=vol_delta,
        regime_transition=is_transition,
        transition_confidence=confidence,
        magnitude_band=_magnitude_band(vol_delta),
        signal_kind=signal_kind,
        taxonomy_version=a.taxonomy_version,
        regime_classifier_version=a.regime_classifier_version,
    )


def drift_to_payload(
    d: DriftResult,
    *,
    window_a_run_id: Optional[str] = None,
    window_b_run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Plain-dict view for embedding in a drift_analysis evidence
    payload. ``window_*_run_id`` parameters let the runner stitch in
    the regime_summary lineage links when the windows have been
    classified-and-emitted."""
    a = dict(d.window_a)
    b = dict(d.window_b)
    a["regime_summary_run_id"] = window_a_run_id
    b["regime_summary_run_id"] = window_b_run_id
    return {
        "schema_version":             d.schema_version,
        "taxonomy_version":           d.taxonomy_version,
        "regime_classifier_version":  d.regime_classifier_version,
        "drift_kind":                 d.drift_kind,
        "signal_kind":                d.signal_kind,
        "window_a":                   a,
        "window_b":                   b,
        "shift": {
            "vol_delta_pct":          d.vol_delta_pct,
            "regime_transition":      d.regime_transition,
            "transition_confidence":  d.transition_confidence,
            "magnitude_band":         d.magnitude_band,
        },
    }
