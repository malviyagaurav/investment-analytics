"""Confidence-weighted per-regime sample assembly.

Loads regime_summary rows from the audit chain, partitions samples
by regime_signature, applies confidence weighting, and surfaces
per-regime coverage attestations the runner uses for refusal /
recommendation decisions.

## Why this lives apart from the runner

The runner orchestrates emit + governance + lineage. The sampler is
pure: same chain + same cache → same per-regime samples. Keeping
this layer pure means the replay handler can re-derive samples
deterministically and the unit tests can exercise the sampling
algorithm without going through emit_evidence.

## Confidence weighting

Each sample drawn from a regime's window carries the raw weight the
substrate fetcher returned (1.0 for HIGH_CORRELATION_THRESHOLD —
uniform substrate weighting) multiplied by the regime_summary's
``classification_confidence``. A regime classified at confidence
0.1 contributes ~10% of the per-sample weight a regime classified
at confidence 1.0 contributes.

A future weighting scheme could replace this with something more
elaborate (e.g., temporal decay, signal_quality multipliers). Today's
scheme is intentionally simple: trust the regime exactly as much as
its classifier said it should be trusted.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.calibration.config import (
    ACCEPTED_COVERAGE_QUALITIES,
    MIN_EFFECTIVE_WEIGHT_PER_REGIME,
    MIN_OBSERVATIONS_PER_REGIME,
)
from backend.calibration.targets import CalibrationTarget, DEFAULT_CACHE_DIR


@dataclass(frozen=True)
class RegimeBucket:
    """Per-regime aggregation result. Inspected by the runner to
    decide which regimes pass coverage and which become explicit
    refusals.

    Carries enough provenance to be embedded in calibration_basis
    without re-derivation — every field is forensically meaningful.
    """
    regime_signature:        Dict[str, Any]
    regime_class:            str
    contributing_run_ids:    Tuple[str, ...]
    samples:                 Tuple[Tuple[float, float], ...]  # (value, effective_weight)
    raw_observation_count:   int
    effective_weight_total:  float
    coverage_quality_mix:    Dict[str, int]  # {"high": N, "medium": N, "low": N}


def _iter_regime_summaries(audit_path: Path):
    """Yield (run_id, event_dict, payload_dict) for every
    regime_summary row in the chain. The payload is loaded eagerly
    here (rather than lazy) because the sampler needs the
    classification_basis (window dates, applied_bands) and
    signal_quality which only live in the by-reference evidence
    payload — not in the lightweight audit_event row.
    """
    if not audit_path.exists():
        return
    data_dir = audit_path.parent.parent
    with audit_path.open("r", encoding="utf-8") as h:
        for line in h:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = record.get("event", {})
            if event.get("evidence_kind") != "regime_summary":
                continue
            ev_ref = event.get("evidence_ref") or {}
            ev_path_rel = ev_ref.get("path")
            if not ev_path_rel:
                continue
            ev_path = data_dir / ev_path_rel
            if not ev_path.exists():
                continue
            try:
                envelope = json.loads(ev_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            payload = envelope.get("payload", {})
            yield event["run_id"], event, payload


def _is_superseded(target_run_id: str, audit_path: Path) -> bool:
    """True if any regime_summary in the chain has
    supersedes_run_id == target_run_id. Walks the chain once;
    cheap on single-machine bounded scales."""
    if not audit_path.exists():
        return False
    with audit_path.open("r", encoding="utf-8") as h:
        for line in h:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = record.get("event", {})
            if event.get("evidence_kind") != "regime_summary":
                continue
            if event.get("supersedes_run_id") == target_run_id:
                return True
    return False


def assemble_per_regime_samples(
    audit_path: Path,
    target: CalibrationTarget,
    *,
    cache_dir: Optional[Path] = None,
    regime_scope: Optional[List[Dict[str, Any]]] = None,
    skip_superseded: bool = True,
) -> Tuple[List[RegimeBucket], List[str]]:
    """Walk the chain, partition samples by regime_signature, apply
    confidence weighting, return per-regime buckets.

    Args:
      audit_path:       chain path.
      target:           registered calibration target whose
                        substrate_fetcher produces raw samples per
                        regime window.
      cache_dir:        NAV cache directory (the fetcher reads it).
      regime_scope:     optional list of regime_signature dicts to
                        restrict aggregation to. None = include all.
      skip_superseded:  when True, exclude regime_summary rows whose
                        run_id has been superseded by a newer row.
                        The current claim is the surviving one.

    Returns:
      (buckets, consulted_run_ids)
        buckets:             list of RegimeBucket, one per
                             regime_signature observed in scope.
                             "indeterminate" buckets are EXCLUDED
                             (a confidence state, not a market
                             state — never participates in
                             calibration).
        consulted_run_ids:   every regime_summary run_id the sampler
                             examined, in chain order. Forensic
                             value: the refusal_reason
                             ``insufficient_substrate`` payload
                             records this list (often empty on
                             day-one).
    """
    cdir = cache_dir or DEFAULT_CACHE_DIR

    # signature_key -> bucket-in-progress
    grouped: Dict[str, Dict[str, Any]] = {}
    consulted: List[str] = []

    for run_id, event, payload in _iter_regime_summaries(audit_path):
        consulted.append(run_id)

        if skip_superseded and _is_superseded(run_id, audit_path):
            continue

        regime_class = payload.get("regime_class")
        if regime_class == "indeterminate":
            continue  # confidence state, not market state

        signature = payload.get("regime_signature") or {}
        if not signature:
            continue

        if regime_scope is not None:
            if signature not in regime_scope:
                continue

        confidence = float(payload.get("classification_confidence") or 0.0)
        if confidence <= 0:
            continue

        coverage_quality = (payload.get("signal_quality") or {}).get(
            "coverage_quality", "low",
        )

        window_start = payload.get("window_start_date")
        window_end = payload.get("window_end_date")
        if not window_start or not window_end:
            continue

        raw_samples = target.substrate_fetcher(window_start, window_end, cdir)

        sig_key = json.dumps(signature, sort_keys=True, separators=(",", ":"))
        bucket = grouped.setdefault(sig_key, {
            "regime_signature": signature,
            "regime_class": regime_class,
            "contributing_run_ids": [],
            "samples": [],
            "raw_observation_count": 0,
            "effective_weight_total": 0.0,
            "coverage_quality_mix": {"high": 0, "medium": 0, "low": 0},
        })
        bucket["contributing_run_ids"].append(run_id)
        bucket["coverage_quality_mix"][coverage_quality] = (
            bucket["coverage_quality_mix"].get(coverage_quality, 0) + 1
        )
        for value, raw_weight in raw_samples:
            effective_weight = float(raw_weight) * confidence
            if effective_weight <= 0:
                continue
            bucket["samples"].append((float(value), effective_weight))
            bucket["raw_observation_count"] += 1
            bucket["effective_weight_total"] += effective_weight

    out: List[RegimeBucket] = []
    for sig_key in sorted(grouped):  # deterministic iteration order
        b = grouped[sig_key]
        out.append(RegimeBucket(
            regime_signature=b["regime_signature"],
            regime_class=b["regime_class"],
            contributing_run_ids=tuple(b["contributing_run_ids"]),
            samples=tuple(b["samples"]),
            raw_observation_count=b["raw_observation_count"],
            effective_weight_total=round(b["effective_weight_total"], 6),
            coverage_quality_mix=dict(b["coverage_quality_mix"]),
        ))
    return out, consulted


def passes_coverage_floors(bucket: RegimeBucket) -> Tuple[bool, Optional[str]]:
    """Apply the three coverage floors. Returns (passed, refusal_reason).

    The three checks are independent — a regime can fail more than
    one. We return the FIRST failing check by priority (substrate
    count → effective weight → coverage quality) so the refusal
    reason is single-valued and operator-actionable.
    """
    if bucket.raw_observation_count < MIN_OBSERVATIONS_PER_REGIME:
        return False, "insufficient_coverage"
    if bucket.effective_weight_total < MIN_EFFECTIVE_WEIGHT_PER_REGIME:
        return False, "confidence_floor_unmet"
    high_or_medium = (
        bucket.coverage_quality_mix.get("high", 0)
        + bucket.coverage_quality_mix.get("medium", 0)
    )
    if high_or_medium == 0:
        return False, "confidence_floor_unmet"  # all-"low" coverage
    return True, None
