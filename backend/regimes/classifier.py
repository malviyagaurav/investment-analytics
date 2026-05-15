"""Window-level regime classifier (v1, rule-based).

Pure function over the Nifty 50 NAV cache. Slices the daily-return
series to ``[start, end]``, computes annualized vol, maps to a
REGIME_CLASSES value with explicit confidence + signal-quality
provenance.

Emits nothing. Mutates nothing. The runner wraps this with the
``emit_evidence`` path. Replay handler re-invokes with the recorded
parameters and lands on ``exact_match`` (or ``expected_divergence``
if methodology/taxonomy bumped between record and replay).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import pstdev
from typing import Any, Dict, List, Optional, Tuple

import hashlib

from backend.regimes.config import (
    CLASSIFICATION_SEMANTICS,
    ClassifierParams,
    DEFAULT_CLASSIFIER_PARAMS,
    HIGH_COVERAGE_HISTORY_DAYS,
    HIGH_COVERAGE_MISSING_PCT,
    MEDIUM_COVERAGE_HISTORY_DAYS,
    MEDIUM_COVERAGE_MISSING_PCT,
    REGIME_SUMMARY_SCHEMA_VERSION,
    REGIME_TAXONOMY_VERSION,
)


# Default path to the NAV cache directory. Tests redirect to a tmp dir.
ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CACHE_DIR = ROOT / "data" / "cache"


@dataclass(frozen=True)
class RegimeClassification:
    """Pure result of classify_window. Carries everything needed to
    construct the regime_summary payload — the runner just wraps it
    with envelope metadata and emits.

    Confidence model (split, per Step 10 governance review):

      classification_stability  — derived from the coverage of THIS
        window AND background history quality. Reflects how
        statistically stable the underlying vol estimate is.
      boundary_separation       — derived from proximity to the
        nearest band boundary. Reflects how decisive the class
        assignment is given the bands.
      classification_confidence — combined scalar. Equals
        ``min(boundary_separation_with_floor, classification_stability)``
        where the floor (0.5) applies ONLY when the underlying
        estimate is stable (high stability AND high coverage_quality).
        Without the floor, a calm stable market window with vol
        numerically near a static boundary would land at near-zero
        confidence for purely geometric reasons — penalizing it as
        though it were epistemically weak when it is not.
    """
    regime_class:                 str
    window_start_date:            str
    window_end_date:              str
    window_coverage_days:         int
    classification_confidence:    float
    classification_stability:     float
    boundary_separation:          float
    classification_basis:         Dict[str, Any]
    signal_quality:               Dict[str, Any]
    classification_semantics:     str  = CLASSIFICATION_SEMANTICS
    taxonomy_version:             str  = REGIME_TAXONOMY_VERSION
    regime_classifier_version:    str  = "v1"
    schema_version:               str  = REGIME_SUMMARY_SCHEMA_VERSION


def _parse_iso(d: str) -> date:
    return datetime.fromisoformat(d).date()


def _parse_dd_mm_yyyy(s: str) -> Optional[date]:
    """Parse the mfapi.in DD-MM-YYYY format the NAV cache stores."""
    try:
        return datetime.strptime(s.strip(), "%d-%m-%Y").date()
    except (ValueError, AttributeError):
        return None


def _load_signal_navs(
    cache_dir: Path, scheme_code: int,
) -> List[Tuple[date, float]]:
    """Read the signal scheme's NAV history from cache and return
    [(date, nav)] sorted oldest-first. Returns an empty list when the
    cache file is missing or malformed — the caller treats empty as
    'no signal available' → indeterminate."""
    path = cache_dir / f"{scheme_code}.json"
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    data = raw.get("response", {}).get("data", []) or []
    out: List[Tuple[date, float]] = []
    for entry in data:
        d = _parse_dd_mm_yyyy(entry.get("date", ""))
        if d is None:
            continue
        try:
            nav = float(entry.get("nav", ""))
        except (ValueError, TypeError):
            continue
        if nav <= 0:
            continue
        out.append((d, nav))
    out.sort(key=lambda x: x[0])
    return out


def _window_slice(
    series: List[Tuple[date, float]], start: date, end: date,
) -> List[Tuple[date, float]]:
    return [(d, v) for d, v in series if start <= d <= end]


def _annualized_vol_pct(window: List[Tuple[date, float]]) -> Optional[float]:
    if len(window) < 2:
        return None
    daily_returns: List[float] = []
    for i in range(1, len(window)):
        prev = window[i - 1][1]
        if prev > 0:
            daily_returns.append((window[i][1] / prev) - 1.0)
    if len(daily_returns) < 2:
        return None
    return pstdev(daily_returns) * math.sqrt(252) * 100.0


def _bucket_by_bands(vol_pct: float, params: ClassifierParams) -> str:
    if vol_pct >= params.crisis_threshold_pct:
        return "crisis_vol"
    if vol_pct >= params.high_threshold_pct:
        return "high_vol"
    if vol_pct >= params.low_threshold_pct:
        return "normal_vol"
    return "low_vol"


def _classification_confidence(
    vol_pct: float,
    coverage_days: int,
    signal_quality: Dict[str, Any],
    params: ClassifierParams,
) -> Tuple[float, float, float]:
    """Return (combined_confidence, classification_stability, boundary_separation).

    boundary_separation — distance to nearest band boundary, scaled
      by ``boundary_confidence_margin_pct``, saturated at 1.0. A
      window with vol 0.5% from a boundary and margin=10 yields 0.05.

    classification_stability — coverage of the WINDOW relative to
      ``min_coverage_days`` (linearly from 0.0 → 1.0 across
      [min_coverage, 2*min_coverage]).

    combined_confidence — ``min(effective_boundary_separation,
      classification_stability)`` where ``effective_boundary_separation``
      floors at 0.5 ONLY when the underlying estimate is stable
      (window coverage ≥ MEDIUM_COVERAGE_HISTORY_DAYS observations AND
      signal_quality.coverage_quality == "high"). The floor prevents
      a stable calm-market window with vol numerically near a boundary
      from registering as epistemically weak.
    """
    boundaries = [
        params.low_threshold_pct,
        params.high_threshold_pct,
        params.crisis_threshold_pct,
    ]
    margin = params.boundary_confidence_margin_pct
    nearest = min(abs(vol_pct - b) for b in boundaries)
    boundary_separation = min(1.0, nearest / margin) if margin > 0 else 1.0

    if coverage_days < params.min_coverage_days:
        stability = 0.0
    elif coverage_days >= 2 * params.min_coverage_days:
        stability = 1.0
    else:
        span = params.min_coverage_days
        stability = 0.5 + 0.5 * (
            (coverage_days - params.min_coverage_days) / span
        )

    # Stable estimate floor — boundary geometry is not epistemic
    # weakness when:
    #   1. classification_stability is fully saturated (the window
    #      has ≥ 2 * min_coverage_days of observations, enough for
    #      the vol estimate to be statistically usable), AND
    #   2. the background history depth puts coverage_quality at "high".
    # When both hold, the boundary-separation component cannot drag
    # the combined confidence below 0.5.
    stable_estimate = (
        stability >= 1.0
        and signal_quality.get("coverage_quality") == "high"
    )
    if stable_estimate:
        effective_boundary_separation = max(0.5, boundary_separation)
    else:
        effective_boundary_separation = boundary_separation

    combined = min(effective_boundary_separation, stability)
    return (round(combined, 4),
            round(stability, 4),
            round(boundary_separation, 4))


def _compute_signal_quality(
    full_series: List[Tuple[date, float]],
    window: List[Tuple[date, float]],
    window_start: date,
    window_end: date,
) -> Dict[str, Any]:
    """Quality of the signal underpinning the classification.

    All counts are in TRADING-DAY OBSERVATIONS, not calendar days —
    consistent with the classifier itself (which operates on
    observations) and with the HIGH/MEDIUM_COVERAGE_HISTORY_DAYS
    cutpoints (defined as 252 * N).

    history_depth_days: count of NAV observations from earliest
    cached point through window_end inclusive. "How many trading
    days of data back up this classification."

    missing_data_pct: percentage of expected trading days in the
    window (Mon-Fri) that don't have a NAV point. Holiday-blind
    approximation (a future enrichment can plug in a real calendar).

    coverage_quality: derived bucket so downstream layers can filter
    on a typed signal without re-deriving the math.
    """
    if not full_series:
        return {
            "history_depth_days": 0,
            "missing_data_pct":  100.0,
            "coverage_quality":  "low",
        }
    # Count actual NAV observations from start of cache to window_end —
    # same unit as HIGH/MEDIUM_COVERAGE_HISTORY_DAYS (trading days,
    # 252/year). Previously this was (window_end - earliest).days
    # which is CALENDAR days; comparing calendar days to a trading-day
    # threshold introduced silent skew.
    history_depth_days = sum(1 for d, _ in full_series if d <= window_end)

    # Expected trading days = weekdays in [start, end]. Approximation —
    # ignores Indian market holidays. For the day-one signal it is
    # close enough; a future enrichment can plug in a real calendar.
    expected_weekdays = 0
    cursor = window_start
    while cursor <= window_end:
        if cursor.weekday() < 5:
            expected_weekdays += 1
        cursor = cursor + timedelta(days=1)
    observed = len(window)
    if expected_weekdays > 0:
        missing_pct = max(0.0, (expected_weekdays - observed) / expected_weekdays * 100.0)
    else:
        missing_pct = 100.0
    missing_pct = round(missing_pct, 2)

    if (history_depth_days >= HIGH_COVERAGE_HISTORY_DAYS
            and missing_pct < HIGH_COVERAGE_MISSING_PCT):
        coverage_quality = "high"
    elif (history_depth_days >= MEDIUM_COVERAGE_HISTORY_DAYS
            and missing_pct < MEDIUM_COVERAGE_MISSING_PCT):
        coverage_quality = "medium"
    else:
        coverage_quality = "low"

    return {
        "history_depth_days": history_depth_days,
        "missing_data_pct":   missing_pct,
        "coverage_quality":   coverage_quality,
    }


def classify_window(
    window_start_date: str,
    window_end_date: str,
    *,
    params: Optional[ClassifierParams] = None,
    cache_dir: Optional[Path] = None,
) -> RegimeClassification:
    """Classify a [start, end] window into a REGIME_CLASSES value.

    Pure function — reads the NAV cache, returns a typed result, does
    NOT emit evidence. The runner wraps with emit_evidence.

    indeterminate is returned when:
      - the signal cache file is missing or unreadable
      - the window slice has fewer than min_coverage_days observations
      - the vol computation has < 2 daily return points
    """
    p = params or DEFAULT_CLASSIFIER_PARAMS
    cdir = cache_dir or DEFAULT_CACHE_DIR

    start = _parse_iso(window_start_date)
    end = _parse_iso(window_end_date)
    if end < start:
        raise ValueError(
            f"window_end_date {window_end_date} precedes "
            f"window_start_date {window_start_date}"
        )

    full_series = _load_signal_navs(cdir, p.signal_scheme_code)
    window = _window_slice(full_series, start, end)
    coverage_days = len(window)
    signal_quality = _compute_signal_quality(full_series, window, start, end)

    basis: Dict[str, Any] = {
        "signal_kind":         "nifty50_realized_vol",
        "signal_scheme_code":  p.signal_scheme_code,
        "annualized_vol_pct":  None,
        "applied_bands": {
            "low_threshold_pct":     p.low_threshold_pct,
            "high_threshold_pct":    p.high_threshold_pct,
            "crisis_threshold_pct":  p.crisis_threshold_pct,
        },
        "min_coverage_days":   p.min_coverage_days,
        "boundary_confidence_margin_pct": p.boundary_confidence_margin_pct,
    }

    if coverage_days < p.min_coverage_days:
        return RegimeClassification(
            regime_class="indeterminate",
            window_start_date=window_start_date,
            window_end_date=window_end_date,
            window_coverage_days=coverage_days,
            classification_confidence=0.0,
            classification_stability=0.0,
            boundary_separation=0.0,
            classification_basis={**basis,
                                  "indeterminate_reason": "insufficient_coverage"},
            signal_quality=signal_quality,
        )

    vol_pct = _annualized_vol_pct(window)
    if vol_pct is None:
        return RegimeClassification(
            regime_class="indeterminate",
            window_start_date=window_start_date,
            window_end_date=window_end_date,
            window_coverage_days=coverage_days,
            classification_confidence=0.0,
            classification_stability=0.0,
            boundary_separation=0.0,
            classification_basis={**basis,
                                  "indeterminate_reason": "vol_uncomputable"},
            signal_quality=signal_quality,
        )

    basis["annualized_vol_pct"] = round(vol_pct, 4)
    regime = _bucket_by_bands(vol_pct, p)
    confidence, stability, separation = _classification_confidence(
        vol_pct, coverage_days, signal_quality, p,
    )

    return RegimeClassification(
        regime_class=regime,
        window_start_date=window_start_date,
        window_end_date=window_end_date,
        window_coverage_days=coverage_days,
        classification_confidence=confidence,
        classification_stability=stability,
        boundary_separation=separation,
        classification_basis=basis,
        signal_quality=signal_quality,
    )


def regime_signature(c: RegimeClassification) -> Dict[str, Any]:
    """Canonical identity tuple for a regime interpretation context.

    Step 11 calibration and Step 14 reliability scoring will repeatedly
    need to join per-regime statistics across many regime_summary rows.
    A canonical signature avoids brittle multi-field joins: two
    classifications under identical (signal, taxonomy, classifier,
    window definition) → identical signature → composable.

    The window_hash component covers (window_start, window_end,
    signal_scheme_code, applied_bands) so two classifications of the
    SAME window under the SAME methodology + taxonomy converge on
    identical signatures across emissions.
    """
    basis = c.classification_basis or {}
    window_components = {
        "window_start_date":  c.window_start_date,
        "window_end_date":    c.window_end_date,
        "signal_scheme_code": basis.get("signal_scheme_code"),
        "applied_bands":      basis.get("applied_bands"),
    }
    canonical = json.dumps(
        window_components,
        sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    )
    window_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "signal_kind":         basis.get("signal_kind"),
        "taxonomy_version":    c.taxonomy_version,
        "classifier_version":  c.regime_classifier_version,
        "window_hash":         window_hash,
    }


def classification_to_payload(c: RegimeClassification) -> Dict[str, Any]:
    """Plain-dict view for embedding in a regime_summary evidence payload."""
    return {
        "schema_version":             c.schema_version,
        "taxonomy_version":           c.taxonomy_version,
        "regime_classifier_version":  c.regime_classifier_version,
        "classification_semantics":   c.classification_semantics,
        "regime_class":               c.regime_class,
        "window_start_date":          c.window_start_date,
        "window_end_date":            c.window_end_date,
        "window_coverage_days":       c.window_coverage_days,
        "classification_confidence":  c.classification_confidence,
        "classification_stability":   c.classification_stability,
        "boundary_separation":        c.boundary_separation,
        "classification_basis":       dict(c.classification_basis),
        "signal_quality":             dict(c.signal_quality),
        "regime_signature":           regime_signature(c),
    }
