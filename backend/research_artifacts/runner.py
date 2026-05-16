"""Threshold-recommendation runner: emit + lookup + supersedes enforcement.

The runner is the only writer. It:

  1. Locates the cited ``calibration_report`` in the audit chain.
  2. REFUSES to project from a calibration refusal — a refusal
     payload has ``recommendation=null`` and is not promotable.
     This is the load-bearing anti-laundering guard: a typed
     wrapper cannot turn a non-recommendation into a recommendation.
  3. Validates the supersedes chain (linear, acyclic, no branching).
  4. Projects calibration_report.recommendation + calibration_scope
     into a new threshold_recommendation row.
  5. Emits exactly one audit event via emit_evidence.

## Day-one emit constraint

Only ``adoption_status="proposed"`` rows can be emitted in Step 13.
The remaining statuses (under_review / adopted / rejected /
superseded) are reserved for Step 15's promotion machinery; emitting
them from Step 13 would silently cross the no-auto-promotion
boundary. The runner refuses out-of-band statuses with a typed
``ValueError`` (no audit row appended).

## Production isolation

The runner NEVER mutates:
  - production threshold constants (HIGH_CORRELATION_THRESHOLD etc.)
  - METHODOLOGY_VERSIONS
  - any existing calibration_report or other evidence row

The only side effect of a successful emit is one threshold_recommendation
audit row + one by-reference evidence file.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.calibration.config import CALIBRATION_TARGETS
from backend.evidence.replay import find_record_by_run_id
from backend.evidence.store import emit_evidence
from backend.research_artifacts.config import (
    ADOPTION_STATUSES,
    STEP_13_PERMITTED_EMIT_STATUSES,
    THRESHOLD_RECOMMENDATION_SCHEMA_VERSION,
)


ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_AUDIT_PATH = ROOT / "data" / "audit" / "audit.jsonl"


# ── Helpers ──────────────────────────────────────────────────────────


def _load_evidence_payload(
    audit_path: Path, evidence_ref: Dict[str, Any],
) -> Dict[str, Any]:
    """Load the by-reference evidence file for an audit row.

    Path resolution follows the convention established in Step 4:
    ``evidence_ref.path`` is relative to ``data/``, which is
    ``audit_path.parent.parent``."""
    data_dir = audit_path.parent.parent
    rel = evidence_ref["path"]
    p = Path(rel)
    if not p.is_absolute():
        p = data_dir / rel
    if not p.exists():
        raise FileNotFoundError(
            f"evidence file missing for cited calibration: {p}"
        )
    with p.open("r", encoding="utf-8") as h:
        envelope = json.load(h)
    return envelope.get("payload", {})


def _find_threshold_recommendations_for_target(
    audit_path: Path, target_canonical_id: str,
) -> List[Dict[str, Any]]:
    """Return every threshold_recommendation row for the given
    target, in chain order. Used by emit to validate supersedes
    chain linearity + acyclicity."""
    if not audit_path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with audit_path.open("r", encoding="utf-8") as h:
        for line in h:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = record.get("event", {})
            if event.get("evidence_kind") != "threshold_recommendation":
                continue
            if event.get("target_canonical_id") != target_canonical_id:
                continue
            rows.append(event)
    return rows


def _enforce_acyclic_supersedes_chain(
    audit_path: Path, target_canonical_id: str, start_run_id: str,
) -> None:
    """Walk the supersedes chain backward from ``start_run_id`` for
    the same target; raise on any repeat visit. Defense-in-depth
    against chain edits or reused run_ids that could create a cycle
    a future lineage walker would loop on.

    Mirrors the discipline in ``backend/regimes/runner.py``."""
    visited: set = set()
    cursor: Optional[str] = start_run_id
    target_rows = {
        r["run_id"]: r
        for r in _find_threshold_recommendations_for_target(
            audit_path, target_canonical_id,
        )
    }
    while cursor is not None:
        if cursor in visited:
            raise ValueError(
                f"supersedes chain contains a cycle at run_id="
                f"{cursor!r}; refusing to emit. visited: {sorted(visited)}"
            )
        visited.add(cursor)
        row = target_rows.get(cursor)
        if row is None:
            # cursor escaped the per-target set; chain terminates
            # cleanly (cited a row outside this target — should not
            # happen given upstream validation but harmless to stop).
            return
        cursor = row.get("supersedes_run_id")


# ── Public emit ──────────────────────────────────────────────────────


def emit_threshold_recommendation(
    calibration_report_run_id: str,
    *,
    audit_path: Optional[Path] = None,
    supersedes_run_id: Optional[str] = None,
    rationale: Optional[str] = None,
    adoption_status: str = "proposed",
) -> Dict[str, Any]:
    """Project a ``calibration_report`` into a typed
    ``threshold_recommendation`` row.

    Args:
      calibration_report_run_id: run_id of the calibration_report
        the recommendation derives from. MUST cite a calibration
        with a non-null recommendation; refusal-projection is
        rejected at emit time.
      audit_path: chain path. Defaults to project's live chain.
      supersedes_run_id: run_id of the prior threshold_recommendation
        this one supersedes. Required if any prior row exists for
        the same target; refused otherwise (no silent shadowing).
      rationale: free-text justification; recorded in
        non_semantic_metadata, EXCLUDED from semantic identity.
      adoption_status: must be ``"proposed"`` in Step 13. Forward
        statuses (under_review/adopted/rejected/superseded) are
        Step 15's domain and refused here.

    Returns the audit record from emit_evidence.

    Raises:
      KeyError    — calibration_report_run_id not in chain.
      ValueError  — cited row is not a calibration_report;
                    cited calibration is a refusal (recommendation
                    is null); target is unsupported; adoption_status
                    is not "proposed"; supersedes_run_id is invalid
                    (missing/wrong target/already superseded/cycle);
                    no supersedes given but prior row exists.
    """
    audit_path = audit_path or DEFAULT_AUDIT_PATH

    # ── Validate adoption_status early (cheap closed-enum check) ────
    if adoption_status not in STEP_13_PERMITTED_EMIT_STATUSES:
        raise ValueError(
            f"adoption_status={adoption_status!r} is not permitted by "
            f"Step 13's emit policy. Only "
            f"{sorted(STEP_13_PERMITTED_EMIT_STATUSES)} may be emitted "
            f"here; promotion-lifecycle transitions are Step 15's "
            f"domain and emit through a separate machinery."
        )
    if adoption_status not in ADOPTION_STATUSES:
        raise ValueError(
            f"adoption_status must be one of {sorted(ADOPTION_STATUSES)}, "
            f"got {adoption_status!r}"
        )

    # ── Locate cited calibration_report ──────────────────────────────
    record = find_record_by_run_id(audit_path, calibration_report_run_id)
    if record is None:
        raise KeyError(
            f"calibration_report_run_id={calibration_report_run_id!r} "
            f"not found in audit chain at {audit_path}"
        )
    cal_event = record.get("event", {})
    if cal_event.get("evidence_kind") != "calibration_report":
        raise ValueError(
            f"cited run_id {calibration_report_run_id!r} has "
            f"evidence_kind={cal_event.get('evidence_kind')!r}, "
            f"expected 'calibration_report'"
        )

    # ── Refusal-projection guard (load-bearing) ──────────────────────
    # A calibration with recommendation=null is a typed refusal. The
    # threshold_recommendation layer cannot turn a refusal into a
    # recommendation — that would be the canonical evidence-laundering
    # vector this whole architecture is designed to refuse.
    if cal_event.get("recommendation") is None:
        raise ValueError(
            f"cannot project threshold_recommendation from a calibration "
            f"refusal. cited calibration {calibration_report_run_id!r} "
            f"has refusal_reason={cal_event.get('refusal_reason')!r}; "
            f"emit refused to prevent a typed wrapper from becoming "
            f"an evidence-laundering loophole."
        )

    # ── Validate target ──────────────────────────────────────────────
    target = cal_event.get("target")
    if target not in CALIBRATION_TARGETS:
        raise ValueError(
            f"cited calibration_report targets {target!r}, which is not "
            f"in CALIBRATION_TARGETS. Registered: "
            f"{sorted(CALIBRATION_TARGETS)}"
        )

    # ── Validate supersedes chain ────────────────────────────────────
    prior = _find_threshold_recommendations_for_target(audit_path, target)
    if prior and supersedes_run_id is None:
        prior_ids = [r["run_id"] for r in prior]
        raise ValueError(
            f"threshold_recommendation already exists for target "
            f"{target!r} (run_id(s): {prior_ids}). Pass supersedes_run_id"
            f"=<one of those> to record an update; emit refuses to "
            f"silently shadow a prior recommendation."
        )
    if supersedes_run_id is not None:
        if not prior:
            raise ValueError(
                f"supersedes_run_id={supersedes_run_id!r} given but no "
                f"prior threshold_recommendation exists for target {target!r}"
            )
        prior_ids = {r["run_id"] for r in prior}
        if supersedes_run_id not in prior_ids:
            raise ValueError(
                f"supersedes_run_id={supersedes_run_id!r} does not match "
                f"any prior threshold_recommendation for target {target!r}. "
                f"Existing run_ids: {sorted(prior_ids)}"
            )
        # Linearity: cited row must not already be superseded.
        for row in prior:
            if row.get("supersedes_run_id") == supersedes_run_id:
                raise ValueError(
                    f"run_id {supersedes_run_id!r} is already superseded "
                    f"by {row['run_id']!r}; supersedes chains must remain "
                    f"linear (no branching)"
                )
        # Acyclicity (defense in depth).
        _enforce_acyclic_supersedes_chain(
            audit_path, target, supersedes_run_id,
        )

    # ── Load full calibration payload for scope projection ───────────
    cal_evidence_ref = cal_event.get("evidence_ref")
    if not cal_evidence_ref:
        raise ValueError(
            f"cited calibration_report {calibration_report_run_id!r} has "
            f"no evidence_ref; cannot project scope into recommendation"
        )
    cal_payload = _load_evidence_payload(audit_path, cal_evidence_ref)

    recommended_value = cal_payload.get("recommendation")
    if recommended_value is None:
        # Belt-and-suspenders: audit row showed non-null recommendation
        # but the evidence file says null. That's an integrity mismatch.
        raise ValueError(
            f"audit/evidence mismatch: calibration_report audit row "
            f"shows recommendation={cal_event.get('recommendation')!r} "
            f"but evidence payload shows recommendation=null"
        )

    # ── Project into threshold_recommendation payload ────────────────
    cal_scope = cal_payload.get("calibration_scope", {}) or {}
    recommendation_scope = {
        "valid_within_regimes":  cal_scope.get("valid_within_regimes", []),
        "assumed_stationarity":  cal_scope.get("assumed_stationarity",
                                               "(not recorded on cited calibration)"),
        "known_limitations":     cal_scope.get("known_limitations", []),
    }

    payload = {
        "schema_version":         THRESHOLD_RECOMMENDATION_SCHEMA_VERSION,
        "target_canonical_id":    target,
        "recommended_value":      recommended_value,
        "recommendation_scope":   recommendation_scope,
        "adoption_status":        adoption_status,
        "derived_from_calibration_report_run_id": calibration_report_run_id,
        "supersedes_run_id":      supersedes_run_id,
        "methodology_kind":       "data_driven_variant",
        "non_semantic_metadata":  {"rationale": rationale} if rationale else {},
    }

    audit_event = {
        "event_type":             "threshold_recommendation",
        "target_canonical_id":    target,
        "recommended_value":      recommended_value,
        "adoption_status":        adoption_status,
        "derived_from_calibration_report_run_id": calibration_report_run_id,
        "supersedes_run_id":      supersedes_run_id,
        "schema_version":         THRESHOLD_RECOMMENDATION_SCHEMA_VERSION,
    }

    # parent_run_id wires to either the prior recommendation (lineage
    # within the supersedes chain) or the cited calibration_report
    # (lineage anchor when this is the first recommendation for the
    # target). Both are valid genealogies.
    parent_run_id = supersedes_run_id or calibration_report_run_id

    return emit_evidence(
        audit_log_path=audit_path,
        evidence_kind="threshold_recommendation",
        audit_event=audit_event,
        payload=payload,
        parent_run_id=parent_run_id,
    )


# ── Query helper ─────────────────────────────────────────────────────


def find_threshold_recommendations(
    audit_path: Optional[Path] = None,
    *,
    target_canonical_id: Optional[str] = None,
    adoption_status: Optional[str] = None,
    include_superseded: bool = True,
) -> List[dict]:
    """Scan the chain for threshold_recommendation rows with optional
    filters. Returns inner event dicts (not full audit records).

    When ``include_superseded=False``, rows that have been superseded
    by a later row in the same target are excluded; only the current
    claim per target survives. Matches the regime_summary query
    convention from Step 10."""
    audit_path = audit_path or DEFAULT_AUDIT_PATH
    if not audit_path.exists():
        return []
    rows: List[dict] = []
    with audit_path.open("r", encoding="utf-8") as h:
        for line in h:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = record.get("event", {})
            if event.get("evidence_kind") != "threshold_recommendation":
                continue
            if (target_canonical_id is not None
                    and event.get("target_canonical_id") != target_canonical_id):
                continue
            if (adoption_status is not None
                    and event.get("adoption_status") != adoption_status):
                continue
            rows.append(event)
    if include_superseded:
        return rows
    superseded: set = {r["supersedes_run_id"] for r in rows
                       if r.get("supersedes_run_id")}
    return [r for r in rows if r["run_id"] not in superseded]
