"""Calibration runner — emits calibration_report rows with strict
schema invariants on recommendation vs refusal.

The runner is the only thing that touches the audit chain. The
substrate fetchers (targets.py), aggregator (sampling.py), and
percentile math (percentiles.py) are pure — the runner orchestrates
them, applies coverage floors per regime, computes per-regime
recommendations, aggregates across regimes (median), packages
provenance into the typed payload, and emits exactly one row.

## Schema invariants (load-bearing)

Both branches emit the SAME shape; only the conclusion differs.

  recommendation present:
    - recommendation:        non-null float
    - refusal_reason:        null
    - calibration_scope.valid_within_regimes:  non-empty
    - calibration_basis:                       fully populated

  recommendation absent:
    - recommendation:        null
    - refusal_reason:        non-null (from REFUSAL_REASONS enum)
    - calibration_scope.valid_within_regimes:  empty
    - calibration_basis:                       fully populated
      (refusal is symmetric work: same forensic depth)

A refusal is itself an evidence-bearing claim. The system examined
the substrate, applied the coverage rules, and judged it
insufficient. That examination work must be visible in the audit
chain just as recommendation work is.

## Cross-regime aggregation

Top-level recommendation = median across per-regime recommendations.
NOT a pooled percentile across all samples (that would be the
compression artifact Step 10's regime partitioning was designed to
prevent). Per-regime values remain inspectable in
``calibration_basis.samples_by_regime``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.calibration.config import (
    ACCEPTED_COVERAGE_QUALITIES,
    CALIBRATION_ENGINE_VERSION,
    CALIBRATION_REPORT_SCHEMA_VERSION,
    CALIBRATION_TARGETS,
    MIN_EFFECTIVE_WEIGHT_PER_REGIME,
    MIN_OBSERVATIONS_PER_REGIME,
    REFUSAL_REASONS,
)
from backend.calibration.percentiles import (
    unweighted_median,
    weighted_percentile,
)
from backend.calibration.sampling import (
    RegimeBucket,
    assemble_per_regime_samples,
    passes_coverage_floors,
)
from backend.calibration.targets import (
    DEFAULT_CACHE_DIR,
    REGISTERED_TARGETS,
    get_target,
)
from backend.evidence.store import emit_evidence

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_AUDIT_PATH = ROOT / "data" / "audit" / "audit.jsonl"


# ── Refusal helpers ──────────────────────────────────────────────────


def _empty_basis(target_id: str, percentile: float) -> Dict[str, Any]:
    """Calibration_basis shape for refusals where no substrate was
    examined (target_unsupported / insufficient_substrate). Still
    structured — refusal evidence must carry the same forensic
    shape as recommendation evidence, even when most fields are
    empty by construction."""
    return {
        "target_canonical_id":          target_id,
        "regime_signatures_consulted":  [],
        "regime_summary_run_ids":       [],
        "observation_count":            0,
        "effective_weight_total":       0.0,
        "weighting_scheme":             "classification_confidence_weighted",
        "percentile_used":              percentile,
        "minimum_coverage_required":    MIN_OBSERVATIONS_PER_REGIME,
        "minimum_effective_weight":     MIN_EFFECTIVE_WEIGHT_PER_REGIME,
        "accepted_coverage_qualities":  sorted(ACCEPTED_COVERAGE_QUALITIES),
        "samples_by_regime":            {},
    }


def _build_basis(
    target_id: str,
    percentile: float,
    buckets: List[RegimeBucket],
    consulted: List[str],
    samples_by_regime: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Full calibration_basis when substrate was examined."""
    all_signatures = [b.regime_signature for b in buckets]
    return {
        "target_canonical_id":          target_id,
        "regime_signatures_consulted":  all_signatures,
        "regime_summary_run_ids":       sorted(consulted),
        "observation_count":            sum(
            b.raw_observation_count for b in buckets
        ),
        "effective_weight_total":       round(sum(
            b.effective_weight_total for b in buckets
        ), 6),
        "weighting_scheme":             "classification_confidence_weighted",
        "percentile_used":              percentile,
        "minimum_coverage_required":    MIN_OBSERVATIONS_PER_REGIME,
        "minimum_effective_weight":     MIN_EFFECTIVE_WEIGHT_PER_REGIME,
        "accepted_coverage_qualities":  sorted(ACCEPTED_COVERAGE_QUALITIES),
        "samples_by_regime":            samples_by_regime,
    }


def _emit(
    audit_path: Path,
    target_id: str,
    payload: Dict[str, Any],
    parent_run_id: Optional[str],
) -> Dict[str, Any]:
    """Single emit path. The audit_event is lightweight; heavy
    provenance lives in the by-reference evidence payload."""
    audit_event = {
        "event_type":                  "calibration_report",
        "target":                      target_id,
        "recommendation":              payload["recommendation"],
        "refusal_reason":              payload["refusal_reason"],
        "calibration_engine_version":  CALIBRATION_ENGINE_VERSION,
        "valid_within_regimes_count":  len(
            payload["calibration_scope"]["valid_within_regimes"]
        ),
        "excluded_regimes_count":      len(
            payload["calibration_scope"]["excluded_regimes"]
        ),
        "derivation_depth":            payload["derivation_depth"],
        "schema_version":              CALIBRATION_REPORT_SCHEMA_VERSION,
    }
    return emit_evidence(
        audit_log_path=audit_path,
        evidence_kind="calibration_report",
        audit_event=audit_event,
        payload=payload,
        parent_run_id=parent_run_id,
    )


# ── Public API ───────────────────────────────────────────────────────


def run_calibration(
    target: str,
    *,
    audit_path: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    percentile: Optional[float] = None,
    regime_scope: Optional[List[Dict[str, Any]]] = None,
    parent_run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the calibration engine for a registered target and emit
    exactly one calibration_report row.

    Returns the emit_evidence audit record dict. The recommendation
    (or refusal) lives in record["event"] and in the by-reference
    evidence payload.

    The runner refuses cleanly — production thresholds are NEVER
    mutated. A calibration_report is a typed proposal; promotion is
    Step 15's domain.
    """
    audit_path = audit_path or DEFAULT_AUDIT_PATH
    cache_dir = cache_dir or DEFAULT_CACHE_DIR

    # ── Refusal: target_unsupported (closed-enum + registry check) ──
    if target not in CALIBRATION_TARGETS or target not in REGISTERED_TARGETS:
        pctl = percentile if percentile is not None else 95.0
        payload = _refusal_payload(
            target_id=target,
            refusal_reason="target_unsupported",
            percentile=pctl,
            basis=_empty_basis(target, pctl),
            derived_from_run_ids=[],
            derivation_depth=0,
        )
        return _emit(audit_path, target, payload, parent_run_id)

    target_obj = get_target(target)
    pctl = percentile if percentile is not None else target_obj.default_percentile

    # ── Refusal: regime_indeterminate (scope explicitly asks for it) ──
    if regime_scope is not None:
        for sig in regime_scope:
            # A scope entry pointing at indeterminate by construction
            # (no signature has regime_class field — this is defensive
            # for malformed callers). Indeterminate as a calibration
            # target makes no sense; refuse symmetrically.
            pass  # placeholder — indeterminate is filtered at sample time

    # ── Assemble per-regime samples ──────────────────────────────────
    buckets, consulted_run_ids = assemble_per_regime_samples(
        audit_path=audit_path,
        target=target_obj,
        cache_dir=cache_dir,
        regime_scope=regime_scope,
    )

    # ── Refusal: insufficient_substrate ──────────────────────────────
    # No regime_summary rows in chain → no buckets to evaluate.
    # The basis still records that we tried (consulted_run_ids may be
    # empty; that's the honest answer).
    if not buckets:
        payload = _refusal_payload(
            target_id=target,
            refusal_reason="insufficient_substrate",
            percentile=pctl,
            basis=_build_basis(
                target_id=target,
                percentile=pctl,
                buckets=[],
                consulted=consulted_run_ids,
                samples_by_regime={},
            ),
            derived_from_run_ids=[],
            derivation_depth=0,
        )
        return _emit(audit_path, target, payload, parent_run_id)

    # ── Per-regime coverage evaluation ───────────────────────────────
    valid_signatures: List[Dict[str, Any]] = []
    excluded_signatures: List[Dict[str, Any]] = []
    per_regime_recommendations: List[float] = []
    samples_by_regime: Dict[str, Dict[str, Any]] = {}
    derived_from_run_ids: List[str] = []

    for bucket in buckets:
        sig_key = json.dumps(bucket.regime_signature,
                             sort_keys=True, separators=(",", ":"))
        passed, refusal = passes_coverage_floors(bucket)
        per_regime_entry: Dict[str, Any] = {
            "regime_class":             bucket.regime_class,
            "n":                        bucket.raw_observation_count,
            "effective_weight":         bucket.effective_weight_total,
            "coverage_quality_mix":     bucket.coverage_quality_mix,
            "contributing_run_ids":     list(bucket.contributing_run_ids),
        }
        if passed:
            value = round(weighted_percentile(list(bucket.samples), pctl), 6)
            per_regime_entry["percentile_value"] = value
            per_regime_entry["status"] = "included"
            valid_signatures.append(bucket.regime_signature)
            per_regime_recommendations.append(value)
            derived_from_run_ids.extend(bucket.contributing_run_ids)
        else:
            per_regime_entry["percentile_value"] = None
            per_regime_entry["status"] = "excluded"
            per_regime_entry["per_regime_refusal_reason"] = refusal
            excluded_signatures.append(bucket.regime_signature)
        samples_by_regime[sig_key] = per_regime_entry

    basis = _build_basis(
        target_id=target,
        percentile=pctl,
        buckets=buckets,
        consulted=consulted_run_ids,
        samples_by_regime=samples_by_regime,
    )

    # ── Refusal: no regime passed coverage ───────────────────────────
    # All buckets failed; aggregate refusal reason is the most common
    # per-regime refusal (or insufficient_coverage as conservative
    # default). The per-regime detail in samples_by_regime preserves
    # the granular reasons.
    if not valid_signatures:
        # Aggregate refusal reason: prefer insufficient_coverage; if
        # all regimes failed on confidence_floor_unmet, surface that
        # instead. Operator-actionable signal.
        all_failed_on_confidence = all(
            samples_by_regime[k].get("per_regime_refusal_reason")
            == "confidence_floor_unmet"
            for k in samples_by_regime
        )
        aggregate_refusal = (
            "confidence_floor_unmet" if all_failed_on_confidence
            else "insufficient_coverage"
        )
        payload = _refusal_payload(
            target_id=target,
            refusal_reason=aggregate_refusal,
            percentile=pctl,
            basis=basis,
            derived_from_run_ids=[],  # no regime contributed
            derivation_depth=0,
        )
        # Excluded signatures still recorded in scope so consumers can
        # see what was examined.
        payload["calibration_scope"]["excluded_regimes"] = excluded_signatures
        return _emit(audit_path, target, payload, parent_run_id)

    # ── Recommendation path ──────────────────────────────────────────
    top_level_recommendation = round(
        unweighted_median(per_regime_recommendations), 6,
    )
    derivation_depth = 1  # one hop from regime_summary lineage
    payload = _recommendation_payload(
        target_id=target,
        recommendation=top_level_recommendation,
        percentile=pctl,
        valid_signatures=valid_signatures,
        excluded_signatures=excluded_signatures,
        basis=basis,
        derived_from_run_ids=sorted(set(derived_from_run_ids)),
        derivation_depth=derivation_depth,
    )
    return _emit(audit_path, target, payload, parent_run_id)


# ── Payload builders with schema invariants ──────────────────────────


def _recommendation_payload(
    target_id: str,
    recommendation: float,
    percentile: float,
    valid_signatures: List[Dict[str, Any]],
    excluded_signatures: List[Dict[str, Any]],
    basis: Dict[str, Any],
    derived_from_run_ids: List[str],
    derivation_depth: int,
) -> Dict[str, Any]:
    """Construct a recommendation payload with the schema invariant
    enforced: recommendation non-null ↔ refusal_reason null ↔
    valid_within_regimes non-empty."""
    if not valid_signatures:
        raise ValueError(
            "recommendation payload requires non-empty valid_signatures"
        )
    return {
        "schema_version":              CALIBRATION_REPORT_SCHEMA_VERSION,
        "calibration_engine_version":  CALIBRATION_ENGINE_VERSION,
        "target":                      target_id,
        "target_canonical_id":         target_id,
        "recommendation":              recommendation,
        "refusal_reason":              None,
        "calibration_scope": {
            "valid_within_regimes": valid_signatures,
            "excluded_regimes":     excluded_signatures,
            "assumed_stationarity": (
                "Per-regime recommendations are emitted on the assumption "
                "that realized-vol regimes capture the relevant "
                "environmental partition. Cross-regime aggregation uses "
                "the median of per-regime values — NOT a pooled "
                "percentile across regimes."
            ),
            "known_limitations": [
                "engineered_coverage_floors_not_calibrated",
                "weighting_scheme_v1_uniform_substrate",
                "universe_restricted_to_deep_history_equity_funds",
            ],
        },
        "calibration_basis":         basis,
        "methodology_kind":          "data_driven_variant",
        "derived_from_run_ids":      derived_from_run_ids,
        "derivation_depth":          derivation_depth,
    }


def _refusal_payload(
    target_id: str,
    refusal_reason: str,
    percentile: float,
    basis: Dict[str, Any],
    derived_from_run_ids: List[str],
    derivation_depth: int,
) -> Dict[str, Any]:
    """Construct a refusal payload with the schema invariant enforced:
    recommendation null ↔ refusal_reason non-null ↔ valid_within_regimes
    empty. Refusal is symmetric work — calibration_basis is recorded
    with the same fields as recommendation."""
    if refusal_reason not in REFUSAL_REASONS:
        raise ValueError(
            f"refusal_reason must be one of {sorted(REFUSAL_REASONS)}, "
            f"got {refusal_reason!r}"
        )
    return {
        "schema_version":              CALIBRATION_REPORT_SCHEMA_VERSION,
        "calibration_engine_version":  CALIBRATION_ENGINE_VERSION,
        "target":                      target_id,
        "target_canonical_id":         target_id,
        "recommendation":              None,
        "refusal_reason":              refusal_reason,
        "calibration_scope": {
            "valid_within_regimes": [],
            "excluded_regimes":     [],
            "assumed_stationarity": "n/a — refusal payload",
            "known_limitations": [
                f"refused_with_reason:{refusal_reason}",
            ],
        },
        "calibration_basis":         basis,
        "methodology_kind":          "data_driven_variant",
        "derived_from_run_ids":      derived_from_run_ids,
        "derivation_depth":          derivation_depth,
    }


def find_calibration_reports(
    audit_path: Optional[Path] = None,
    *,
    target: Optional[str] = None,
    only_recommendations: bool = False,
    only_refusals: bool = False,
) -> List[dict]:
    """Linear chain scan for calibration_report rows. Optional
    filters for target / recommendation-vs-refusal. Returns inner
    event dicts (not full audit records)."""
    audit_path = audit_path or DEFAULT_AUDIT_PATH
    if not audit_path.exists():
        return []
    if only_recommendations and only_refusals:
        return []
    matches: List[dict] = []
    with audit_path.open("r", encoding="utf-8") as h:
        for line in h:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = record.get("event", {})
            if event.get("evidence_kind") != "calibration_report":
                continue
            if target is not None and event.get("target") != target:
                continue
            if only_recommendations and event.get("recommendation") is None:
                continue
            if only_refusals and event.get("refusal_reason") is None:
                continue
            matches.append(event)
    return matches
