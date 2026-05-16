"""Per-dimension scoring functions.

Each function takes ``(target_canonical_id, target_run_id,
audit_path, window_days)`` and returns a ``DimensionResult`` with:

  - score ∈ [0.0, 1.0]  OR  score=None with a typed refusal_reason
  - basis dict carrying the raw counts/ratios used
  - run_ids list citing every evidence row consulted

All functions are PURE (no side effects, no audit emission).
Deterministic given the chain content + window. No ML, no fitting,
no probabilistic theater.

## Why basis is mandatory

A score without its basis is opaque — the canonical
evidence-laundering vector. Every DimensionResult carries the raw
counts/ratios that produced the score so an operator can verify
the math by inspection. Step 14's anti-laundering posture
demands this.

## Refusal as evidence

A dimension that cannot compute a score (insufficient substrate,
unsupported target shape, etc.) returns ``score=None`` with a
typed refusal_reason. This carries forensic visibility forward
into the reliability_score payload — the score's evidentiary
basis is fully traceable.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.reliability.config import (
    DRIFT_EXPOSURE_TRANSITION_FLOOR,
    MIN_CALIBRATIONS_FOR_REFUSAL_FREQ,
    MIN_DRIFT_ROWS_FOR_EXPOSURE,
    MIN_RECOMMENDATIONS_FOR_CHURN,
    MIN_REGIMES_FOR_REGIME_SENSITIVITY,
    MIN_REPLAYS_FOR_REPLAY_STABILITY,
    REPLAY_STATE_WEIGHTS,
)


@dataclass(frozen=True)
class DimensionResult:
    """Pure result of one scoring dimension. Carried into the
    reliability_score payload as a typed sub-record."""
    dimension:        str
    score:            Optional[float]
    basis:            Dict[str, Any] = field(default_factory=dict)
    refusal_reason:   Optional[str]  = None
    run_ids:          Tuple[str, ...] = ()


# ── Chain access (shared utilities) ──────────────────────────────────


def _iter_chain(audit_path: Path):
    """Yield parsed audit records line-by-line. Skips malformed."""
    if not audit_path.exists():
        return
    with audit_path.open("r", encoding="utf-8") as h:
        for line in h:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _within_window(timestamp_str: str, window_start: datetime) -> bool:
    """True if the row's timestamp is at or after window_start.
    Malformed timestamps default to "in window" (conservative —
    don't silently exclude a row we can't parse)."""
    try:
        ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts >= window_start
    except (ValueError, AttributeError):
        return True


def _gather_rows(
    audit_path: Path, *, evidence_kind: str, window_start: datetime,
    target_canonical_id: Optional[str] = None,
) -> List[dict]:
    """Filter chain rows by evidence_kind + window + optional target."""
    out: List[dict] = []
    for record in _iter_chain(audit_path):
        event = record.get("event", {})
        if event.get("evidence_kind") != evidence_kind:
            continue
        if not _within_window(record.get("timestamp", ""), window_start):
            continue
        if target_canonical_id is not None:
            # target_canonical_id is on calibration / threshold rows;
            # may be elsewhere on a "target" field for calibrations.
            t = event.get("target_canonical_id") or event.get("target")
            if t != target_canonical_id:
                continue
        out.append(record)
    return out


def _load_evidence_payload(
    audit_path: Path, evidence_ref: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Read by-reference payload if available; return None on miss."""
    if not evidence_ref:
        return None
    data_dir = audit_path.parent.parent
    rel = evidence_ref.get("path")
    if not rel:
        return None
    p = Path(rel)
    if not p.is_absolute():
        p = data_dir / rel
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("payload")
    except (json.JSONDecodeError, OSError):
        return None


# ── Dimension 1: evidence_stability ──────────────────────────────────


def evidence_stability(
    *, target_canonical_id: str, target_run_id: str,
    audit_path: Path, window_start: datetime,
) -> DimensionResult:
    """Fraction of threshold_recommendation rows for this target in
    the window that did NOT supersede a prior row. High score = stable
    chain, few methodology updates churning the artifact.

    score = 1 - (supersede_count / max(1, total))
    """
    rows = _gather_rows(
        audit_path,
        evidence_kind="threshold_recommendation",
        window_start=window_start,
        target_canonical_id=target_canonical_id,
    )
    events = [r.get("event", {}) for r in rows]
    total = len(events)
    if total < MIN_RECOMMENDATIONS_FOR_CHURN:
        return DimensionResult(
            dimension="evidence_stability", score=None,
            basis={"total_rows": total,
                   "minimum_required": MIN_RECOMMENDATIONS_FOR_CHURN},
            refusal_reason="insufficient_substrate",
            run_ids=tuple(e["run_id"] for e in events if "run_id" in e),
        )
    supersede_count = sum(1 for e in events if e.get("supersedes_run_id"))
    score = 1.0 - (supersede_count / total)
    return DimensionResult(
        dimension="evidence_stability",
        score=round(score, 6),
        basis={
            "total_rows":       total,
            "supersede_count":  supersede_count,
            "supersede_ratio":  round(supersede_count / total, 6),
        },
        run_ids=tuple(e["run_id"] for e in events if "run_id" in e),
    )


# ── Dimension 2: replay_stability (asymmetric tightening here) ──────


def replay_stability(
    *, target_canonical_id: str, target_run_id: str,
    audit_path: Path, window_start: datetime,
) -> DimensionResult:
    """Weighted average of REPLAY_STATE_WEIGHTS over recent
    replay_result rows whose ORIGINAL artifact's
    ``target_canonical_id`` matches the scored target.

    The asymmetry is load-bearing:
      exact_match / semantically_equivalent / expected_divergence
        contribute +1.0 (typed honest replay outcomes)
      unreproducible contributes +0.5 (substrate decay, not
        necessarily the claim's fault)
      invalid_replay contributes -1.0 (claim integrity cannot
        currently be justified — chain integrity hit)

    ## Cross-target exclusion (Step 14 tightening)

    Each replay_result is filtered through its
    ``payload.audit_event_ref.run_id`` — we fetch the ORIGINAL
    replayed artifact and check whether its
    ``target_canonical_id`` (or ``target`` field on
    calibration_report) matches the scored target. Replay rows
    that target a different target_canonical_id are EXCLUDED;
    rows whose original cannot be resolved are also excluded
    (conservative — if we can't prove same-target lineage, we
    don't count it).

    Without this exclusion, reliability scores for target X would
    silently incorporate replay history of target Y, contaminating
    per-target diagnostics.

    Basis exposes:
      - state histogram (of INCLUDED rows only)
      - unreproducible_causes histogram (the typed reasons)
      - invalid_replay_causes histogram
      - rejected_cross_target_count (forensic visibility into
        how many replays were filtered out for target mismatch)
    """
    from backend.evidence.replay import find_record_by_run_id

    rows = _gather_rows(
        audit_path, evidence_kind="replay_result",
        window_start=window_start,
    )
    relevant: List[Tuple[str, dict, Optional[dict]]] = []  # (run_id, event, payload)
    rejected_cross_target = 0
    rejected_unresolvable = 0

    for record in rows:
        event = record.get("event", {})
        payload = _load_evidence_payload(audit_path, event.get("evidence_ref"))
        if payload is None:
            rejected_unresolvable += 1
            continue
        original = payload.get("audit_event_ref", {}) or {}
        original_kind = original.get("evidence_kind")
        original_run_id = original.get("run_id")
        # Only threshold_recommendation / calibration_report replays
        # can plausibly correspond to a scored threshold target.
        if original_kind not in (
            "threshold_recommendation", "calibration_report",
        ):
            continue
        if not original_run_id:
            rejected_unresolvable += 1
            continue
        # Resolve original artifact to verify same-target lineage.
        original_record = find_record_by_run_id(audit_path, original_run_id)
        if original_record is None:
            rejected_unresolvable += 1
            continue
        original_event = original_record.get("event", {})
        # threshold_recommendation uses ``target_canonical_id``;
        # calibration_report uses ``target``. Both name the same concept.
        original_target = (original_event.get("target_canonical_id")
                           or original_event.get("target"))
        if original_target != target_canonical_id:
            rejected_cross_target += 1
            continue
        relevant.append((event.get("run_id", ""), event, payload))
    total = len(relevant)
    if total < MIN_REPLAYS_FOR_REPLAY_STABILITY:
        return DimensionResult(
            dimension="replay_stability", score=None,
            basis={
                "total_replays":                 total,
                "minimum_required":              MIN_REPLAYS_FOR_REPLAY_STABILITY,
                "rejected_cross_target_count":   rejected_cross_target,
                "rejected_unresolvable_count":   rejected_unresolvable,
            },
            refusal_reason="no_replay_history",
            run_ids=tuple(rid for rid, _, _ in relevant if rid),
        )

    histogram: Dict[str, int] = {}
    unreproducible_causes: Dict[str, int] = {}
    invalid_replay_causes: Dict[str, int] = {}
    weighted_total = 0.0
    for rid, event, payload in relevant:
        state = event.get("state") or payload.get("state") or "unknown"
        histogram[state] = histogram.get(state, 0) + 1
        weight = REPLAY_STATE_WEIGHTS.get(state, 0.0)
        weighted_total += weight
        if state == "unreproducible":
            reason = (payload.get("reason") or event.get("reason")
                      or "unspecified")
            unreproducible_causes[reason] = (
                unreproducible_causes.get(reason, 0) + 1
            )
        elif state == "invalid_replay":
            reason = (payload.get("reason") or event.get("reason")
                      or "unspecified")
            invalid_replay_causes[reason] = (
                invalid_replay_causes.get(reason, 0) + 1
            )
    raw = weighted_total / total
    score = max(0.0, min(1.0, raw))
    return DimensionResult(
        dimension="replay_stability",
        score=round(score, 6),
        basis={
            "total_replays":                 total,
            "state_histogram":               histogram,
            "weighted_total":                round(weighted_total, 6),
            "raw_score_pre_clamp":           round(raw, 6),
            "unreproducible_causes":         unreproducible_causes,
            "invalid_replay_causes":         invalid_replay_causes,
            "applied_state_weights":         dict(REPLAY_STATE_WEIGHTS),
            "rejected_cross_target_count":   rejected_cross_target,
            "rejected_unresolvable_count":   rejected_unresolvable,
        },
        run_ids=tuple(rid for rid, _, _ in relevant if rid),
    )


# ── Dimension 3: regime_sensitivity ─────────────────────────────────


def regime_sensitivity(
    *, target_canonical_id: str, target_run_id: str,
    audit_path: Path, window_start: datetime,
) -> DimensionResult:
    """For the cited calibration_report, look at per-regime
    recommendation values in samples_by_regime. Higher dispersion
    across regimes = higher regime sensitivity = LOWER score.

    score = 1 / (1 + coefficient_of_variation)
    """
    # Find the target_run_id (a threshold_recommendation), follow
    # derived_from_calibration_report_run_id to the calibration.
    target_payload = _find_target_payload(audit_path, target_run_id)
    if target_payload is None:
        return DimensionResult(
            dimension="regime_sensitivity", score=None,
            basis={"target_run_id": target_run_id},
            refusal_reason="target_run_id_missing",
        )
    calibration_run_id = target_payload.get(
        "derived_from_calibration_report_run_id"
    )
    if not calibration_run_id:
        return DimensionResult(
            dimension="regime_sensitivity", score=None,
            basis={"reason": "target has no cited calibration"},
            refusal_reason="insufficient_substrate",
        )
    cal_payload = _find_payload_by_run_id(audit_path, calibration_run_id)
    if cal_payload is None:
        return DimensionResult(
            dimension="regime_sensitivity", score=None,
            basis={"calibration_run_id": calibration_run_id,
                   "reason": "cited calibration not in chain"},
            refusal_reason="insufficient_substrate",
            run_ids=(calibration_run_id,),
        )
    samples_by_regime = (cal_payload.get("calibration_basis", {})
                                     .get("samples_by_regime", {}) or {})
    per_regime_values = [
        entry.get("percentile_value")
        for entry in samples_by_regime.values()
        if entry.get("percentile_value") is not None
           and entry.get("status") == "included"
    ]
    if len(per_regime_values) < MIN_REGIMES_FOR_REGIME_SENSITIVITY:
        return DimensionResult(
            dimension="regime_sensitivity", score=None,
            basis={"included_regime_count": len(per_regime_values),
                   "minimum_required": MIN_REGIMES_FOR_REGIME_SENSITIVITY,
                   "calibration_run_id": calibration_run_id},
            refusal_reason="insufficient_substrate",
            run_ids=(calibration_run_id,),
        )
    mean_value = statistics.mean(per_regime_values)
    if mean_value == 0:
        cv = 0.0
    else:
        stdev = (statistics.pstdev(per_regime_values)
                 if len(per_regime_values) > 1 else 0.0)
        cv = abs(stdev / mean_value)
    score = 1.0 / (1.0 + cv)
    return DimensionResult(
        dimension="regime_sensitivity",
        score=round(score, 6),
        basis={
            "calibration_run_id":     calibration_run_id,
            "included_regime_count":  len(per_regime_values),
            "per_regime_values":      [round(v, 6) for v in per_regime_values],
            "mean_value":             round(mean_value, 6),
            "coefficient_of_variation": round(cv, 6),
        },
        run_ids=(calibration_run_id,),
    )


# ── Dimension 4: calibration_coverage_quality ───────────────────────


def calibration_coverage_quality(
    *, target_canonical_id: str, target_run_id: str,
    audit_path: Path, window_start: datetime,
) -> DimensionResult:
    """Fraction of regimes consulted by the cited calibration that
    had high or medium coverage_quality (excluding low-coverage
    refusals). Reflects the depth of the underlying substrate."""
    target_payload = _find_target_payload(audit_path, target_run_id)
    if target_payload is None:
        return DimensionResult(
            dimension="calibration_coverage_quality", score=None,
            basis={"target_run_id": target_run_id},
            refusal_reason="target_run_id_missing",
        )
    calibration_run_id = target_payload.get(
        "derived_from_calibration_report_run_id"
    )
    if not calibration_run_id:
        return DimensionResult(
            dimension="calibration_coverage_quality", score=None,
            basis={"reason": "no cited calibration"},
            refusal_reason="insufficient_substrate",
        )
    cal_payload = _find_payload_by_run_id(audit_path, calibration_run_id)
    if cal_payload is None:
        return DimensionResult(
            dimension="calibration_coverage_quality", score=None,
            basis={"calibration_run_id": calibration_run_id},
            refusal_reason="insufficient_substrate",
            run_ids=(calibration_run_id,),
        )
    samples_by_regime = (cal_payload.get("calibration_basis", {})
                                     .get("samples_by_regime", {}) or {})
    total = len(samples_by_regime)
    if total == 0:
        return DimensionResult(
            dimension="calibration_coverage_quality", score=None,
            basis={"calibration_run_id": calibration_run_id,
                   "samples_by_regime_count": 0},
            refusal_reason="insufficient_substrate",
            run_ids=(calibration_run_id,),
        )
    quality_count: Dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    acceptable = 0
    for entry in samples_by_regime.values():
        mix = entry.get("coverage_quality_mix", {}) or {}
        # Take the dominant coverage_quality for that regime: prefer
        # high → medium → low.
        if mix.get("high", 0) > 0:
            dom = "high"
        elif mix.get("medium", 0) > 0:
            dom = "medium"
        else:
            dom = "low"
        quality_count[dom] = quality_count.get(dom, 0) + 1
        if dom in ("high", "medium"):
            acceptable += 1
    score = acceptable / total
    return DimensionResult(
        dimension="calibration_coverage_quality",
        score=round(score, 6),
        basis={
            "calibration_run_id":    calibration_run_id,
            "total_regimes":         total,
            "acceptable_regimes":    acceptable,
            "quality_count":         quality_count,
        },
        run_ids=(calibration_run_id,),
    )


# ── Dimension 5: drift_exposure ─────────────────────────────────────


def drift_exposure(
    *, target_canonical_id: str, target_run_id: str,
    audit_path: Path, window_start: datetime,
) -> DimensionResult:
    """Drift_analysis rows with regime_transition=True in the window.
    More transitions = more environmental instability = LOWER score.
    Score = 1 - min(1, transitions / DRIFT_EXPOSURE_TRANSITION_FLOOR)
    """
    rows = _gather_rows(
        audit_path, evidence_kind="drift_analysis",
        window_start=window_start,
    )
    total = len(rows)
    if total < MIN_DRIFT_ROWS_FOR_EXPOSURE:
        return DimensionResult(
            dimension="drift_exposure", score=None,
            basis={"total_drift_rows": total,
                   "minimum_required": MIN_DRIFT_ROWS_FOR_EXPOSURE},
            refusal_reason="insufficient_substrate",
        )
    transitions = sum(1 for r in rows
                      if r.get("event", {}).get("regime_transition"))
    score = 1.0 - min(1.0, transitions / DRIFT_EXPOSURE_TRANSITION_FLOOR)
    return DimensionResult(
        dimension="drift_exposure",
        score=round(score, 6),
        basis={
            "total_drift_rows":  total,
            "transitions":       transitions,
            "transition_floor":  DRIFT_EXPOSURE_TRANSITION_FLOOR,
        },
        run_ids=tuple(r.get("event", {}).get("run_id", "")
                      for r in rows if r.get("event", {}).get("run_id")),
    )


# ── Dimension 6: supersession_churn ─────────────────────────────────


def supersession_churn(
    *, target_canonical_id: str, target_run_id: str,
    audit_path: Path, window_start: datetime,
) -> DimensionResult:
    """Same target window as evidence_stability but framed as churn:
    HIGH churn = LOW score. Distinct from evidence_stability because
    it counts threshold_recommendation AND regime_summary supersession
    activity affecting this target.

    score = 1 - min(1, total_supersedes / max(1, total_window_rows))
    """
    thresh_rows = _gather_rows(
        audit_path, evidence_kind="threshold_recommendation",
        window_start=window_start,
        target_canonical_id=target_canonical_id,
    )
    regime_rows = _gather_rows(
        audit_path, evidence_kind="regime_summary",
        window_start=window_start,
    )
    total = len(thresh_rows) + len(regime_rows)
    if total == 0:
        return DimensionResult(
            dimension="supersession_churn", score=None,
            basis={"total_rows": 0},
            refusal_reason="insufficient_substrate",
        )
    supersedes = sum(
        1 for r in thresh_rows
        if r.get("event", {}).get("supersedes_run_id")
    ) + sum(
        1 for r in regime_rows
        if r.get("event", {}).get("supersedes_run_id")
    )
    score = 1.0 - min(1.0, supersedes / max(1, total))
    return DimensionResult(
        dimension="supersession_churn",
        score=round(score, 6),
        basis={
            "threshold_rows":     len(thresh_rows),
            "regime_rows":        len(regime_rows),
            "supersede_count":    supersedes,
        },
    )


# ── Dimension 7: refusal_frequency ──────────────────────────────────


def refusal_frequency(
    *, target_canonical_id: str, target_run_id: str,
    audit_path: Path, window_start: datetime,
) -> DimensionResult:
    """Fraction of calibration_report rows for this target that
    emitted a typed refusal in the window. Refusal IS evidence but
    HIGH refusal rate = LOW operational credibility for the
    recommendation derived from those calibrations.

    score = 1 - (refusals / max(1, total_calibrations))
    """
    rows = _gather_rows(
        audit_path, evidence_kind="calibration_report",
        window_start=window_start,
        target_canonical_id=target_canonical_id,
    )
    total = len(rows)
    if total < MIN_CALIBRATIONS_FOR_REFUSAL_FREQ:
        return DimensionResult(
            dimension="refusal_frequency", score=None,
            basis={"total_calibrations": total,
                   "minimum_required": MIN_CALIBRATIONS_FOR_REFUSAL_FREQ},
            refusal_reason="insufficient_substrate",
            run_ids=tuple(r.get("event", {}).get("run_id", "")
                          for r in rows if r.get("event", {}).get("run_id")),
        )
    refusal_histogram: Dict[str, int] = {}
    refusal_count = 0
    for r in rows:
        event = r.get("event", {})
        reason = event.get("refusal_reason")
        if reason:
            refusal_count += 1
            refusal_histogram[reason] = refusal_histogram.get(reason, 0) + 1
    score = 1.0 - (refusal_count / total)
    return DimensionResult(
        dimension="refusal_frequency",
        score=round(score, 6),
        basis={
            "total_calibrations":  total,
            "refusal_count":       refusal_count,
            "refusal_histogram":   refusal_histogram,
        },
        run_ids=tuple(r.get("event", {}).get("run_id", "")
                      for r in rows if r.get("event", {}).get("run_id")),
    )


# ── Dimension 8: methodology_volatility ─────────────────────────────


def methodology_volatility(
    *, target_canonical_id: str, target_run_id: str,
    audit_path: Path, window_start: datetime,
) -> DimensionResult:
    """Distinct methodology snapshots across recent calibration +
    threshold_recommendation rows for this target. More distinct
    methodologies = MORE volatility = LOWER score.

    score = 1 / distinct_methodologies (clamped to 1.0 when no
    rows).
    """
    cal_rows = _gather_rows(
        audit_path, evidence_kind="calibration_report",
        window_start=window_start,
        target_canonical_id=target_canonical_id,
    )
    thresh_rows = _gather_rows(
        audit_path, evidence_kind="threshold_recommendation",
        window_start=window_start,
        target_canonical_id=target_canonical_id,
    )
    all_rows = cal_rows + thresh_rows
    if not all_rows:
        return DimensionResult(
            dimension="methodology_volatility", score=None,
            basis={"total_rows": 0},
            refusal_reason="insufficient_substrate",
        )
    distinct_methodologies: set = set()
    for r in all_rows:
        meth = r.get("event", {}).get("methodology", {})
        # Canonicalize the methodology dict to a hashable form.
        if isinstance(meth, dict):
            distinct_methodologies.add(
                json.dumps(meth, sort_keys=True, separators=(",", ":")),
            )
    n = max(1, len(distinct_methodologies))
    score = 1.0 / n
    return DimensionResult(
        dimension="methodology_volatility",
        score=round(score, 6),
        basis={
            "total_rows":               len(all_rows),
            "distinct_methodologies":   n,
        },
    )


# ── Dimension registry ───────────────────────────────────────────────


DIMENSION_FUNCTIONS = {
    "evidence_stability":            evidence_stability,
    "replay_stability":              replay_stability,
    "regime_sensitivity":            regime_sensitivity,
    "calibration_coverage_quality":  calibration_coverage_quality,
    "drift_exposure":                drift_exposure,
    "supersession_churn":            supersession_churn,
    "refusal_frequency":             refusal_frequency,
    "methodology_volatility":        methodology_volatility,
}


# ── Helpers for target / payload lookup ─────────────────────────────


def _find_target_payload(
    audit_path: Path, target_run_id: str,
) -> Optional[Dict[str, Any]]:
    """Locate the by-reference evidence payload for the target
    threshold_recommendation."""
    return _find_payload_by_run_id(audit_path, target_run_id)


def _find_payload_by_run_id(
    audit_path: Path, run_id: str,
) -> Optional[Dict[str, Any]]:
    """Walk the chain for the row with this run_id; load its
    evidence payload. Returns None on miss."""
    for record in _iter_chain(audit_path):
        event = record.get("event", {})
        if event.get("run_id") != run_id:
            continue
        return _load_evidence_payload(audit_path, event.get("evidence_ref"))
    return None
