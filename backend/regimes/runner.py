"""Regime evidence emitters with supersedes-chain enforcement.

Two public entry points:

  emit_regime_summary  — classify a window, persist a regime_summary
                         evidence row, enforce immutability.
  emit_drift_analysis  — compose two classifications into a typed
                         drift_analysis row.

Both call ``emit_evidence`` exactly once each. No snapshots written,
no sidecar updated, no methodology mutated. The only side effect is
the single audit row + by-reference evidence file.

## Immutability via supersedes_run_id

A regime_summary covers an immutable historical window. If a future
methodology wants to re-classify that same window under improved
rules, the emit path requires an explicit ``supersedes_run_id``
pointing at the prior claim. Silent shadowing is refused — the
chain MUST be able to surface both the original and the supersedes
row so consumers can choose to honor history or accept the update.

Enforcement: at emit time we scan existing regime_summary rows for
ones covering the same (start, end) window. If a prior row exists:
  - and supersedes_run_id is None  → raise.
  - and supersedes_run_id matches  → emit (both rows live forever).
  - and supersedes_run_id doesn't match → raise (linkage mismatch).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.evidence.replay import find_record_by_run_id
from backend.evidence.store import emit_evidence
from backend.regimes.classifier import (
    DEFAULT_CACHE_DIR,
    RegimeClassification,
    classification_to_payload,
    classify_window,
)
from backend.regimes.config import (
    ClassifierParams,
    DEFAULT_CLASSIFIER_PARAMS,
)
from backend.regimes.drift import (
    DriftResult,
    compute_drift,
    drift_to_payload,
)

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_AUDIT_PATH = ROOT / "data" / "audit" / "audit.jsonl"


# ── Chain scans ──────────────────────────────────────────────────────


def _iter_audit_records(audit_path: Path):
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


def find_regime_summaries(
    audit_path: Optional[Path] = None,
    *,
    regime_class: Optional[str] = None,
    window_start_date: Optional[str] = None,
    window_end_date: Optional[str] = None,
    include_superseded: bool = True,
) -> List[dict]:
    """Scan the chain for regime_summary audit-event rows, optionally
    filtered. Returns inner event dicts (not the full audit record).

    ``include_superseded`` toggles whether rows that have been
    superseded by a later regime_summary appear in the result. Default
    True (full history view); calibration consumers typically pass
    False to see only the latest claim per window.
    """
    audit_path = audit_path or DEFAULT_AUDIT_PATH
    rows: List[dict] = []
    for record in _iter_audit_records(audit_path):
        event = record.get("event", {})
        if event.get("evidence_kind") != "regime_summary":
            continue
        if regime_class and event.get("regime_class") != regime_class:
            continue
        if (window_start_date is not None
                and event.get("window_start_date") != window_start_date):
            continue
        if (window_end_date is not None
                and event.get("window_end_date") != window_end_date):
            continue
        rows.append(event)
    if include_superseded:
        return rows
    # Filter out rows that have been superseded by a later one.
    superseded: set = set()
    for r in rows:
        sid = r.get("supersedes_run_id")
        if sid:
            superseded.add(sid)
    return [r for r in rows if r["run_id"] not in superseded]


def _existing_summaries_for_window(
    audit_path: Path, start_date: str, end_date: str,
) -> List[dict]:
    return find_regime_summaries(
        audit_path=audit_path,
        window_start_date=start_date,
        window_end_date=end_date,
        include_superseded=True,
    )


# ── Emitters ─────────────────────────────────────────────────────────


def _classification_to_audit_event(
    c: RegimeClassification,
    supersedes_run_id: Optional[str],
) -> Dict[str, Any]:
    """Lightweight audit-event view. Heavy provenance lives in the
    by-reference evidence payload."""
    return {
        "event_type":                 "regime_summary",
        "regime_class":               c.regime_class,
        "window_start_date":          c.window_start_date,
        "window_end_date":            c.window_end_date,
        "window_coverage_days":       c.window_coverage_days,
        "classification_confidence":  c.classification_confidence,
        "taxonomy_version":           c.taxonomy_version,
        "regime_classifier_version":  c.regime_classifier_version,
        "classification_semantics":   c.classification_semantics,
        "coverage_quality":           c.signal_quality.get("coverage_quality"),
        "supersedes_run_id":          supersedes_run_id,
        "schema_version":             c.schema_version,
    }


def emit_regime_summary(
    window_start_date: str,
    window_end_date: str,
    *,
    params: Optional[ClassifierParams] = None,
    cache_dir: Optional[Path] = None,
    audit_path: Optional[Path] = None,
    supersedes_run_id: Optional[str] = None,
) -> dict:
    """Classify a window and emit exactly one regime_summary evidence
    row + evidence file.

    Immutability rule: if a prior regime_summary exists for the SAME
    (start, end) window, the caller MUST pass a ``supersedes_run_id``
    that matches one of the prior rows' run_ids. Otherwise refused.

    Returns the audit record dict from emit_evidence.
    """
    p = params or DEFAULT_CLASSIFIER_PARAMS
    cdir = cache_dir or DEFAULT_CACHE_DIR
    audit_path = audit_path or DEFAULT_AUDIT_PATH

    # Supersedes-chain enforcement happens BEFORE compute. We refuse
    # to spend cycles on a classification that the chain will reject.
    prior = _existing_summaries_for_window(
        audit_path, window_start_date, window_end_date,
    )
    if prior and supersedes_run_id is None:
        prior_ids = [p["run_id"] for p in prior]
        raise ValueError(
            f"regime_summary already exists for window "
            f"[{window_start_date}, {window_end_date}] "
            f"(run_id(s): {prior_ids}). Pass supersedes_run_id="
            f"<one of those> to record a methodology update; "
            f"emit refuses to silently shadow a prior claim."
        )
    if supersedes_run_id is not None:
        if not prior:
            raise ValueError(
                f"supersedes_run_id={supersedes_run_id!r} given but no "
                f"prior regime_summary exists for window "
                f"[{window_start_date}, {window_end_date}]"
            )
        prior_ids = {p["run_id"] for p in prior}
        if supersedes_run_id not in prior_ids:
            raise ValueError(
                f"supersedes_run_id={supersedes_run_id!r} does not match "
                f"any prior regime_summary for window "
                f"[{window_start_date}, {window_end_date}]. "
                f"Existing run_ids: {sorted(prior_ids)}"
            )
        # Also validate that the cited prior row actually exists in
        # the chain at all (defense in depth — covers chain edits).
        if find_record_by_run_id(audit_path, supersedes_run_id) is None:
            raise ValueError(
                f"supersedes_run_id={supersedes_run_id!r} not found in chain"
            )
        # Linearity: the cited row must not already be superseded by
        # someone else. Branching ("two rows both supersede A") makes
        # downstream lineage walkers ambiguous about which is the
        # current claim. Refuse it.
        for row in prior:
            if row.get("supersedes_run_id") == supersedes_run_id:
                raise ValueError(
                    f"run_id {supersedes_run_id!r} is already superseded by "
                    f"{row['run_id']!r}; supersedes chains must remain "
                    f"linear (no branching)"
                )
        # Acyclicity: walk the supersedes chain backward from the
        # cited row. Append-only + uuid run_ids make a cycle
        # structurally improbable, but if one ever existed (chain
        # edit, run_id reuse) the lineage walker would loop forever.
        # Detect and refuse here.
        _enforce_acyclic_supersedes_chain(audit_path, supersedes_run_id)

    classification = classify_window(
        window_start_date, window_end_date,
        params=p, cache_dir=cdir,
    )
    audit_event = _classification_to_audit_event(
        classification, supersedes_run_id,
    )
    payload = {
        **classification_to_payload(classification),
        "supersedes_run_id": supersedes_run_id,
    }
    return emit_evidence(
        audit_log_path=audit_path,
        evidence_kind="regime_summary",
        audit_event=audit_event,
        payload=payload,
        parent_run_id=supersedes_run_id,  # chains methodology updates
    )


def _enforce_acyclic_supersedes_chain(
    audit_path: Path, start_run_id: str,
) -> None:
    """Walk the supersedes chain backward from start_run_id; raise
    on any repeat visit. Defense in depth — the append-only + UUID
    invariants make a cycle structurally improbable, but a chain
    edit or reused run_id could create one, and a calibration
    consumer walking the chain would loop forever rather than fail.
    """
    visited: set = set()
    cursor: Optional[str] = start_run_id
    while cursor is not None:
        if cursor in visited:
            raise ValueError(
                f"supersedes chain contains a cycle at run_id={cursor!r}; "
                f"refusing to emit. visited: {sorted(visited)}"
            )
        visited.add(cursor)
        record = find_record_by_run_id(audit_path, cursor)
        if record is None:
            return
        cursor = record.get("event", {}).get("supersedes_run_id")


def emit_drift_analysis(
    a: RegimeClassification,
    b: RegimeClassification,
    *,
    audit_path: Optional[Path] = None,
    window_a_run_id: Optional[str] = None,
    window_b_run_id: Optional[str] = None,
) -> dict:
    """Compose two classifications into a drift_analysis row.

    The caller chooses how the classifications were produced (a fresh
    classify_window pair, or two prior regime_summary rows). If the
    classifications came from already-emitted regime_summary rows, pass
    their run_ids via ``window_*_run_id`` so the drift row's payload
    carries the lineage links.
    """
    audit_path = audit_path or DEFAULT_AUDIT_PATH

    drift = compute_drift(a, b)
    payload = drift_to_payload(
        drift,
        window_a_run_id=window_a_run_id,
        window_b_run_id=window_b_run_id,
    )
    audit_event = {
        "event_type":             "drift_analysis",
        "drift_kind":             drift.drift_kind,
        "signal_kind":            drift.signal_kind,
        "vol_delta_pct":          drift.vol_delta_pct,
        "regime_transition":      drift.regime_transition,
        "transition_confidence":  drift.transition_confidence,
        "magnitude_band":         drift.magnitude_band,
        "window_a_run_id":        window_a_run_id,
        "window_b_run_id":        window_b_run_id,
        "schema_version":         drift.schema_version,
    }
    return emit_evidence(
        audit_log_path=audit_path,
        evidence_kind="drift_analysis",
        audit_event=audit_event,
        payload=payload,
    )
