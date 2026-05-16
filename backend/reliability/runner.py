"""Reliability-scoring orchestrator.

Aggregates the 8 dimension functions into one ``reliability_score``
audit row. Symmetric refusal semantics: a refusal payload carries
the SAME shape as a recommendation payload — only the conclusion
field differs.

## Aggregation strategy

Engineered weighted average per ``WEIGHTING_TABLE_*`` in config.py.
Weights are versioned under the ``reliability_weighting``
methodology component; bumping any weight bumps that version and
surfaces as ``methodology_changed`` on replay.

## Refusal floor (K=4)

If ≥ K dimensions individually refuse, the aggregate refuses with
``all_dimensions_refused`` (or ``insufficient_substrate`` when no
substrate at all). The aggregate is computed over the dimensions
that DID score; if too many refused, the aggregate isn't trustworthy
and the layer refuses to emit a number.

## Production isolation (load-bearing)

The runner NEVER mutates:
  - production threshold constants (HIGH_CORRELATION_THRESHOLD etc.)
  - METHODOLOGY_VERSIONS
  - threshold_recommendation.adoption_status
  - any other existing audit row

Only side effect: ONE reliability_score evidence row + its
by-reference payload file.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.calibration.config import CALIBRATION_TARGETS
from backend.evidence.replay import find_record_by_run_id
from backend.evidence.store import emit_evidence
from backend.reliability.config import (
    DEFAULT_SCORING_WINDOW_DAYS,
    K_REFUSAL_FLOOR,
    RELIABILITY_REFUSAL_REASONS,
    RELIABILITY_SCORE_SCHEMA_VERSION,
    RELIABILITY_WEIGHTING_VERSION,
    SCORING_DIMENSIONS,
    WEIGHTING_TABLE_THRESHOLD_RECOMMENDATION,
)
from backend.reliability.dimensions import (
    DIMENSION_FUNCTIONS,
    DimensionResult,
)


ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_AUDIT_PATH = ROOT / "data" / "audit" / "audit.jsonl"


# ── Helpers ──────────────────────────────────────────────────────────


def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _validate_refusal_reason(reason: str) -> None:
    if reason not in RELIABILITY_REFUSAL_REASONS:
        raise ValueError(
            f"refusal_reason must be one of {sorted(RELIABILITY_REFUSAL_REASONS)}, "
            f"got {reason!r}"
        )


def _lookup_target(
    audit_path: Path, target_run_id: str,
) -> Tuple[Optional[dict], Optional[str]]:
    """Find the cited target_run_id in the chain. Returns
    (audit_event_dict, refusal_reason). The refusal_reason is set
    when the lookup fails or the target is itself a refusal."""
    record = find_record_by_run_id(audit_path, target_run_id)
    if record is None:
        return None, "target_run_id_missing"
    event = record.get("event", {})
    if event.get("evidence_kind") != "threshold_recommendation":
        return None, "target_unsupported"
    # A threshold_recommendation that itself is a refusal? — Step 13
    # made threshold_recommendation always carry a non-null
    # recommended_value (refusal projection is refused at emit and
    # replay). So we don't expect this, but guard defensively.
    if event.get("recommended_value") is None:
        return None, "target_is_refusal"
    return event, None


def _build_basis_dim_record(d: DimensionResult) -> Dict[str, Any]:
    """Per-dimension payload-level record."""
    return {
        "dimension":       d.dimension,
        "score":           d.score,
        "basis":           d.basis,
        "refusal_reason":  d.refusal_reason,
        "run_ids_count":   len(d.run_ids),
    }


def _weighted_aggregate(
    dimensions: List[DimensionResult],
    weights: Dict[str, float],
) -> Optional[float]:
    """Sum(score * weight) / sum(weight) over dimensions that scored.
    Returns None if no dimension scored."""
    total_weight = 0.0
    total_score = 0.0
    for d in dimensions:
        if d.score is None:
            continue
        w = weights.get(d.dimension, 0.0)
        if w <= 0:
            continue
        total_weight += w
        total_score += d.score * w
    if total_weight <= 0:
        return None
    return round(total_score / total_weight, 6)


# ── Public API ───────────────────────────────────────────────────────


def score_target(
    target_canonical_id: str,
    target_run_id: str,
    *,
    audit_path: Optional[Path] = None,
    scoring_window_days: int = DEFAULT_SCORING_WINDOW_DAYS,
    supersedes_run_id: Optional[str] = None,
    emit: bool = True,
) -> Dict[str, Any]:
    """Compute the reliability_score for a target artifact.

    Args:
      target_canonical_id: the canonical target (e.g.
        ``portfolio_health.correlation.HIGH_CORRELATION_THRESHOLD``).
      target_run_id: run_id of the threshold_recommendation being
        scored.
      audit_path: chain path.
      scoring_window_days: how far back to gather substrate.
      supersedes_run_id: optional prior reliability_score this one
        supersedes (linear chain enforcement).
      emit: when False, returns the payload without emitting. Used
        by the replay handler.

    Returns the audit record dict from emit_evidence (or just the
    payload dict when emit=False).

    Refusal: emits a row with ``overall_score=null`` and a typed
    ``overall_refusal_reason`` when target is missing, unsupported,
    or ≥K dimensions individually refuse. Refusal payloads carry
    the SAME shape (per-dimension scores + basis + run_ids
    consulted) so the work is forensically traceable.

    The layer NEVER mutates production state.
    """
    audit_path = audit_path or DEFAULT_AUDIT_PATH
    window_start = _now_utc() - timedelta(days=scoring_window_days)
    scored_at = _now_utc().isoformat()

    # ── Target validation ───────────────────────────────────────────
    if target_canonical_id not in CALIBRATION_TARGETS:
        return _emit_refusal(
            audit_path=audit_path,
            target_canonical_id=target_canonical_id,
            target_run_id=target_run_id,
            scoring_window_days=scoring_window_days,
            scored_at=scored_at,
            refusal_reason="target_unsupported",
            dimensions=[],
            supersedes_run_id=supersedes_run_id,
            emit=emit,
            extra_basis={"registered_targets": sorted(CALIBRATION_TARGETS)},
        )

    target_event, target_refusal = _lookup_target(audit_path, target_run_id)
    if target_refusal is not None:
        return _emit_refusal(
            audit_path=audit_path,
            target_canonical_id=target_canonical_id,
            target_run_id=target_run_id,
            scoring_window_days=scoring_window_days,
            scored_at=scored_at,
            refusal_reason=target_refusal,
            dimensions=[],
            supersedes_run_id=supersedes_run_id,
            emit=emit,
        )

    # ── Compute every dimension (each is pure, refusal-safe) ────────
    dim_results: List[DimensionResult] = []
    for dimension_name in sorted(SCORING_DIMENSIONS):
        fn = DIMENSION_FUNCTIONS[dimension_name]
        try:
            result = fn(
                target_canonical_id=target_canonical_id,
                target_run_id=target_run_id,
                audit_path=audit_path,
                window_start=window_start,
            )
        except Exception as exc:  # noqa: BLE001
            # A dimension that RAISED is an IMPLEMENTATION DEFECT,
            # not an honest substrate refusal. Surfacing it as
            # ``insufficient_substrate`` would launder a real bug
            # into an epistemic refusal — exactly the anti-laundering
            # posture the architecture is designed to refuse.
            #
            # Step 14 tightening: ``dimension_execution_failed`` is a
            # distinct typed refusal reason. The exception details are
            # captured inline so operators can diagnose without
            # re-running.
            result = DimensionResult(
                dimension=dimension_name, score=None,
                basis={
                    "exception_type":    type(exc).__name__,
                    "exception_message": str(exc),
                    "dimension":         dimension_name,
                },
                refusal_reason="dimension_execution_failed",
            )
        dim_results.append(result)

    # ── Aggregate or refuse based on K-floor ────────────────────────
    refused_count = sum(1 for d in dim_results if d.score is None)
    if refused_count >= K_REFUSAL_FLOOR:
        # Distinguish "no substrate at all" from "specific dimensions
        # refused for typed reasons."
        if all(d.score is None for d in dim_results):
            refusal_reason = "all_dimensions_refused"
        else:
            refusal_reason = "insufficient_substrate"
        return _emit_refusal(
            audit_path=audit_path,
            target_canonical_id=target_canonical_id,
            target_run_id=target_run_id,
            scoring_window_days=scoring_window_days,
            scored_at=scored_at,
            refusal_reason=refusal_reason,
            dimensions=dim_results,
            supersedes_run_id=supersedes_run_id,
            emit=emit,
        )

    weights = WEIGHTING_TABLE_THRESHOLD_RECOMMENDATION
    overall_score = _weighted_aggregate(dim_results, weights)
    if overall_score is None:
        # No dimension contributed any score under a non-zero weight.
        return _emit_refusal(
            audit_path=audit_path,
            target_canonical_id=target_canonical_id,
            target_run_id=target_run_id,
            scoring_window_days=scoring_window_days,
            scored_at=scored_at,
            refusal_reason="all_dimensions_refused",
            dimensions=dim_results,
            supersedes_run_id=supersedes_run_id,
            emit=emit,
        )

    # ── Build success payload ───────────────────────────────────────
    derived_from = _collect_consulted_run_ids(dim_results)
    payload = _build_payload(
        target_canonical_id=target_canonical_id,
        target_run_id=target_run_id,
        target_evidence_kind="threshold_recommendation",
        scoring_window_days=scoring_window_days,
        scored_at=scored_at,
        overall_score=overall_score,
        overall_refusal_reason=None,
        dimensions=dim_results,
        derived_from_run_ids=derived_from,
        supersedes_run_id=supersedes_run_id,
        weights=weights,
    )
    audit_event = _build_audit_event(payload)
    return _emit_or_return(
        audit_path=audit_path,
        audit_event=audit_event,
        payload=payload,
        emit=emit,
        parent_run_id=supersedes_run_id or target_run_id,
    )


# ── Refusal emit (symmetric work) ───────────────────────────────────


def _emit_refusal(
    *,
    audit_path: Path,
    target_canonical_id: str,
    target_run_id: str,
    scoring_window_days: int,
    scored_at: str,
    refusal_reason: str,
    dimensions: List[DimensionResult],
    supersedes_run_id: Optional[str],
    emit: bool,
    extra_basis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Refusal path: emit a reliability_score row with overall_score
    null, typed refusal_reason, and the SAME per-dimension shape as
    a recommendation row (symmetric work per Step 11/13 pattern)."""
    _validate_refusal_reason(refusal_reason)
    derived_from = _collect_consulted_run_ids(dimensions)
    payload = _build_payload(
        target_canonical_id=target_canonical_id,
        target_run_id=target_run_id,
        target_evidence_kind="threshold_recommendation",
        scoring_window_days=scoring_window_days,
        scored_at=scored_at,
        overall_score=None,
        overall_refusal_reason=refusal_reason,
        dimensions=dimensions,
        derived_from_run_ids=derived_from,
        supersedes_run_id=supersedes_run_id,
        weights=WEIGHTING_TABLE_THRESHOLD_RECOMMENDATION,
        extra_basis=extra_basis,
    )
    audit_event = _build_audit_event(payload)
    return _emit_or_return(
        audit_path=audit_path,
        audit_event=audit_event,
        payload=payload,
        emit=emit,
        parent_run_id=supersedes_run_id or target_run_id,
    )


# ── Payload + audit_event construction ──────────────────────────────


def _build_payload(
    *,
    target_canonical_id: str,
    target_run_id: str,
    target_evidence_kind: str,
    scoring_window_days: int,
    scored_at: str,
    overall_score: Optional[float],
    overall_refusal_reason: Optional[str],
    dimensions: List[DimensionResult],
    derived_from_run_ids: List[str],
    supersedes_run_id: Optional[str],
    weights: Dict[str, float],
    extra_basis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "schema_version":          RELIABILITY_SCORE_SCHEMA_VERSION,
        "weighting_table_version": RELIABILITY_WEIGHTING_VERSION,
        "target_canonical_id":     target_canonical_id,
        "target_run_id":           target_run_id,
        "target_evidence_kind":    target_evidence_kind,
        "scoring_window_days":     scoring_window_days,
        "scored_at":               scored_at,
        "overall_score":           overall_score,
        "overall_refusal_reason":  overall_refusal_reason,
        "dimensions":              [_build_basis_dim_record(d)
                                    for d in dimensions],
        "applied_weights":         dict(weights),
        "derived_from_run_ids":    derived_from_run_ids,
        "supersedes_run_id":       supersedes_run_id,
        "methodology_kind":        "data_driven_variant",
        "extra_basis":             extra_basis or {},
    }


def _build_audit_event(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Lightweight audit_event; heavy detail lives in the
    by-reference payload."""
    return {
        "event_type":              "reliability_score",
        "target_canonical_id":     payload["target_canonical_id"],
        "target_run_id":           payload["target_run_id"],
        "target_evidence_kind":    payload["target_evidence_kind"],
        "scoring_window_days":     payload["scoring_window_days"],
        "overall_score":           payload["overall_score"],
        "overall_refusal_reason":  payload["overall_refusal_reason"],
        "weighting_table_version": payload["weighting_table_version"],
        "dimension_count":         len(payload["dimensions"]),
        "refused_dimension_count": sum(
            1 for d in payload["dimensions"] if d["score"] is None
        ),
        "supersedes_run_id":       payload["supersedes_run_id"],
        "schema_version":          payload["schema_version"],
    }


def _emit_or_return(
    *,
    audit_path: Path,
    audit_event: Dict[str, Any],
    payload: Dict[str, Any],
    emit: bool,
    parent_run_id: Optional[str],
) -> Dict[str, Any]:
    if not emit:
        return {"event": audit_event, "payload": payload}
    return emit_evidence(
        audit_log_path=audit_path,
        evidence_kind="reliability_score",
        audit_event=audit_event,
        payload=payload,
        parent_run_id=parent_run_id,
    )


def _collect_consulted_run_ids(
    dimensions: List[DimensionResult],
) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for d in dimensions:
        for rid in d.run_ids:
            if rid and rid not in seen:
                seen.add(rid)
                out.append(rid)
    return out


# ── Query helper ─────────────────────────────────────────────────────


def find_reliability_scores(
    audit_path: Optional[Path] = None,
    *,
    target_canonical_id: Optional[str] = None,
    overall_refusal_reason: Optional[str] = None,
    only_recommendations: bool = False,
    only_refusals: bool = False,
) -> List[dict]:
    """Linear chain scan for reliability_score rows with optional
    filters. Returns inner event dicts."""
    audit_path = audit_path or DEFAULT_AUDIT_PATH
    if not audit_path.exists():
        return []
    if only_recommendations and only_refusals:
        return []
    rows: List[dict] = []
    with audit_path.open("r", encoding="utf-8", newline="\n") as h:
        for line in h:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = record.get("event", {})
            if event.get("evidence_kind") != "reliability_score":
                continue
            if (target_canonical_id is not None
                    and event.get("target_canonical_id") != target_canonical_id):
                continue
            if (overall_refusal_reason is not None
                    and event.get("overall_refusal_reason") != overall_refusal_reason):
                continue
            if only_recommendations and event.get("overall_score") is None:
                continue
            if only_refusals and event.get("overall_refusal_reason") is None:
                continue
            rows.append(event)
    return rows
