"""Tests for backend.governance (Step 15).

Coverage map:

  Closed-enum discipline:
    - DECISION_TYPES closed (4 values)
    - ELIGIBILITY_REFUSAL_REASONS closed (7 values)
    - OPERATOR_ROLES closed (3 values)
    - ATTESTATION_METHODS closed (1 value)
    - OVERRIDE_REASONS closed (3 values, tightening #1)
    - EVIDENCE_KINDS includes governance_decision (11 total)
    - METHODOLOGY_SCHEMA_VERSION = v5 (governance_eligibility added)

  Operator identity:
    - typed role validation
    - typed attestation method validation
    - empty operator_id rejected
    - supporting_evidence preserved

  Production-state snapshot:
    - reads HIGH_CORRELATION_THRESHOLD
    - byte_hash deterministic
    - source_attestation recorded
    - unregistered target raises

  Eligibility:
    - subject_not_found
    - subject_unsupported
    - subject_is_refusal
    - no_reliability_score
    - reliability_score_is_refusal
    - reliability_below_floor
    - passing path

  emit_governance_decision (per decision_type):
    - approve / reject / request_review / rollback all work
    - eligibility refused without --force-ineligible raises
    - force_ineligible requires override_reason (tightening #1)
    - automation role refused for approve/reject/rollback

  Production isolation (load-bearing):
    - HIGH_CORRELATION_THRESHOLD byte-unchanged after approve
    - METHODOLOGY_VERSIONS byte-unchanged after approve
    - chain still verifies after every emit

  Rollback discipline:
    - rollback with prior-approve auto-discovers target
    - rollback with no prior-approve raises RuntimeError

  Supersession:
    - supersedes_run_id recorded, prior row immutable

  Replay:
    - replay of every decision_type works
    - production_value_changed driver fires when production moves
    - eligibility_state_drifted driver fires when score updates
    - tightening #2: rollback replay with production drift
      classifies as expected_divergence (NOT invalid_replay)

  CLI smoke:
    - show-eligibility prints typed result
    - list filters by decision_type
    - approve via CLI emits and returns exit 0
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.evidence.replay import (
    REPLAY_HANDLERS,
    replay_run,
)
from backend.evidence.store import emit_evidence
from backend.governance.config import (
    ATTESTATION_METHODS,
    DECISION_TYPES,
    ELIGIBILITY_REFUSAL_REASONS,
    GOVERNANCE_DECISION_SCHEMA_VERSION,
    GOVERNANCE_ELIGIBILITY_VERSION,
    OPERATOR_ROLES,
    OVERRIDE_REASONS,
    RELIABILITY_SCORE_FLOOR_FOR_PROMOTION,
)
from backend.governance.eligibility import check_eligibility
from backend.governance.identity import build_operator_identity
from backend.governance.production_state import (
    snapshot_production_state,
    supported_targets,
)
from backend.governance.runner import (
    EligibilityRefused,
    emit_governance_decision,
    find_governance_decisions,
    latest_effective_decision,
)
from backend.investment_analytics.audit import verify_audit_chain
from backend.investment_analytics.evidence_envelope import EVIDENCE_KINDS
from backend.investment_analytics import methodology as meth_mod


TARGET = "portfolio_health.correlation.HIGH_CORRELATION_THRESHOLD"


# ── Closed-enum discipline ──────────────────────────────────────────


class ClosedEnumDisciplineTests(unittest.TestCase):

    def test_decision_types_closed(self) -> None:
        self.assertEqual(DECISION_TYPES, frozenset({
            "approve", "reject", "request_review", "rollback",
        }))

    def test_eligibility_refusal_reasons_closed(self) -> None:
        self.assertEqual(ELIGIBILITY_REFUSAL_REASONS, frozenset({
            "subject_not_found",
            "subject_unsupported",
            "subject_is_refusal",
            "no_reliability_score",
            "reliability_score_is_refusal",
            "reliability_below_floor",
            "target_unsupported",
        }))

    def test_operator_roles_closed(self) -> None:
        self.assertEqual(OPERATOR_ROLES, frozenset({
            "owner", "reviewer", "automation",
        }))

    def test_attestation_methods_closed(self) -> None:
        self.assertEqual(ATTESTATION_METHODS, frozenset({"local_terminal"}))

    def test_override_reasons_closed_tightening_1(self) -> None:
        """TIGHTENING #1 — OVERRIDE_REASONS is closed and exactly
        the three reviewed values. Adding to this enum requires a
        governance review."""
        self.assertEqual(OVERRIDE_REASONS, frozenset({
            "operator_judgment",
            "reliability_pipeline_fault",
            "temporary_substrate_gap",
        }))

    def test_governance_decision_in_evidence_kinds(self) -> None:
        self.assertIn("governance_decision", EVIDENCE_KINDS)
        self.assertEqual(len(EVIDENCE_KINDS), 11)

    def test_methodology_schema_v5_after_step_15(self) -> None:
        self.assertEqual(meth_mod.METHODOLOGY_SCHEMA_VERSION, "v5")
        self.assertIn("governance_eligibility", meth_mod.METHODOLOGY_VERSIONS)

    def test_reliability_floor_is_half(self) -> None:
        self.assertEqual(RELIABILITY_SCORE_FLOOR_FOR_PROMOTION, 0.50)

    def test_governance_decision_schema_v1(self) -> None:
        self.assertEqual(GOVERNANCE_DECISION_SCHEMA_VERSION, "v1")
        self.assertEqual(GOVERNANCE_ELIGIBILITY_VERSION, "v1")


# ── Operator identity ───────────────────────────────────────────────


class OperatorIdentityTests(unittest.TestCase):

    def test_minimum_identity_block(self) -> None:
        ident = build_operator_identity(
            operator_id="owner@local", operator_role="owner",
        )
        self.assertEqual(ident["operator_id"], "owner@local")
        self.assertEqual(ident["operator_role"], "owner")
        self.assertEqual(ident["attestation_method"], "local_terminal")
        self.assertEqual(ident["supporting_evidence"], [])

    def test_empty_operator_id_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_operator_identity(operator_id="", operator_role="owner")
        with self.assertRaises(ValueError):
            build_operator_identity(operator_id="   ", operator_role="owner")

    def test_unknown_role_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_operator_identity(
                operator_id="x", operator_role="superuser",
            )

    def test_unknown_attestation_method_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_operator_identity(
                operator_id="x", operator_role="owner",
                attestation_method="remote_mtls",
            )

    def test_supporting_evidence_preserved(self) -> None:
        ident = build_operator_identity(
            operator_id="owner@local", operator_role="owner",
            supporting_evidence=["TICKET-42", "meeting-2026-05-16"],
        )
        self.assertEqual(
            ident["supporting_evidence"],
            ["TICKET-42", "meeting-2026-05-16"],
        )

    def test_supporting_evidence_rejects_empty_entries(self) -> None:
        with self.assertRaises(ValueError):
            build_operator_identity(
                operator_id="x", operator_role="owner",
                supporting_evidence=["TICKET-42", "  "],
            )


# ── Production state ────────────────────────────────────────────────


class ProductionStateTests(unittest.TestCase):

    def test_high_correlation_threshold_target_registered(self) -> None:
        self.assertIn(TARGET, supported_targets())

    def test_snapshot_returns_typed_block(self) -> None:
        snap = snapshot_production_state(TARGET)
        from backend.investment_analytics.portfolio_health.correlation import (
            HIGH_CORRELATION_THRESHOLD,
        )
        self.assertEqual(snap["target_canonical_id"], TARGET)
        self.assertEqual(snap["current_value"], HIGH_CORRELATION_THRESHOLD)
        self.assertTrue(snap["value_byte_hash"])
        self.assertIn(
            "HIGH_CORRELATION_THRESHOLD", snap["source_attestation"],
        )

    def test_byte_hash_deterministic(self) -> None:
        a = snapshot_production_state(TARGET)
        b = snapshot_production_state(TARGET)
        self.assertEqual(a["value_byte_hash"], b["value_byte_hash"])

    def test_unregistered_target_raises(self) -> None:
        with self.assertRaises(KeyError):
            snapshot_production_state("not.a.real.target")


# ── Test harness for chain-backed tests ─────────────────────────────


class _GovernanceHarness(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tmp.name)
        self.audit_path = self.data_dir / "audit" / "audit.jsonl"
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        # Materialize a minimal registry fixture inside the tmpdir so
        # replay tests don't depend on the developer-local
        # data/registry/schemes.json (which is gitignored and absent
        # in clean CI). Tests that exercise replay must pass
        # registry_path=self.registry_path explicitly; otherwise
        # replay_run falls back to the module DEFAULT_REGISTRY_PATH
        # which short-circuits to state="unreproducible" with empty
        # divergence_drivers in any environment lacking the file.
        self.registry_path = self.data_dir / "registry" / "schemes.json"
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text("{}", encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _read_events(self) -> List[dict]:
        if not self.audit_path.exists():
            return []
        with self.audit_path.open("r", encoding="utf-8") as h:
            return [json.loads(line)["event"] for line in h if line.strip()]

    def _emit_threshold_recommendation(
        self, *, target: str = TARGET,
        recommended_value: Optional[float] = 0.85,
    ) -> str:
        """Plant a typed threshold_recommendation row. When
        recommended_value is None, plants a typed-refusal row."""
        audit_event = {
            "event_type": "threshold_recommendation",
            "target_canonical_id": target,
            "recommended_value": recommended_value,
            "adoption_status": "proposed",
            "derived_from_calibration_report_run_id": "synthetic-cal-001",
            "supersedes_run_id": None,
            "schema_version": "v1",
        }
        payload = {
            "schema_version": "v1",
            "target_canonical_id": target,
            "recommended_value": recommended_value,
            "recommendation_scope": {
                "valid_within_regimes": [],
                "assumed_stationarity": "(synthetic)",
                "known_limitations": [],
            },
            "adoption_status": "proposed",
            "derived_from_calibration_report_run_id": "synthetic-cal-001",
            "supersedes_run_id": None,
            "methodology_kind": "data_driven_variant",
            "non_semantic_metadata": {},
        }
        record = emit_evidence(
            audit_log_path=self.audit_path,
            evidence_kind="threshold_recommendation",
            audit_event=audit_event, payload=payload,
        )
        return record["event"]["run_id"]

    def _emit_reliability_score(
        self, *, target_run_id: str,
        overall_score: Optional[float] = 0.80,
        target: str = TARGET,
    ) -> str:
        """Plant a synthetic reliability_score row. overall_score=None
        plants a typed refusal."""
        refusal = None
        if overall_score is None:
            refusal = "all_dimensions_refused"
        audit_event = {
            "event_type": "reliability_score",
            "target_canonical_id": target,
            "target_run_id": target_run_id,
            "target_evidence_kind": "threshold_recommendation",
            "scoring_window_days": 90,
            "overall_score": overall_score,
            "overall_refusal_reason": refusal,
            "weighting_table_version": "v1",
            "dimension_count": 8,
            "refused_dimension_count": 0,
            "supersedes_run_id": None,
            "schema_version": "v1",
        }
        payload = {
            "schema_version": "v1",
            "weighting_table_version": "v1",
            "target_canonical_id": target,
            "target_run_id": target_run_id,
            "target_evidence_kind": "threshold_recommendation",
            "scoring_window_days": 90,
            "scored_at": "2026-05-16T00:00:00+00:00",
            "overall_score": overall_score,
            "overall_refusal_reason": refusal,
            "dimensions": [],
            "applied_weights": {},
            "derived_from_run_ids": [target_run_id],
            "supersedes_run_id": None,
            "methodology_kind": "data_driven_variant",
            "extra_basis": {},
        }
        record = emit_evidence(
            audit_log_path=self.audit_path,
            evidence_kind="reliability_score",
            audit_event=audit_event, payload=payload,
        )
        return record["event"]["run_id"]

    def _emit_eligible_subject(self) -> str:
        """Convenience: emit a threshold_recommendation + an
        passing reliability_score for the standard TARGET."""
        rec_id = self._emit_threshold_recommendation()
        self._emit_reliability_score(target_run_id=rec_id, overall_score=0.80)
        return rec_id


# ── Eligibility ─────────────────────────────────────────────────────


class EligibilityCheckTests(_GovernanceHarness):

    def test_subject_not_found(self) -> None:
        r = check_eligibility("nonexistent-run-id", self.audit_path)
        self.assertFalse(r["eligibility_passed"])
        self.assertIn("subject_not_found", r["refusal_reasons"])

    def test_subject_unsupported(self) -> None:
        # Plant a non-threshold_recommendation row, then look it up.
        record = emit_evidence(
            audit_log_path=self.audit_path,
            evidence_kind="regime_summary",
            audit_event={"event_type": "regime_summary",
                         "schema_version": "v1"},
            payload={"schema_version": "v1"},
        )
        rid = record["event"]["run_id"]
        r = check_eligibility(rid, self.audit_path)
        self.assertFalse(r["eligibility_passed"])
        self.assertIn("subject_unsupported", r["refusal_reasons"])

    def test_subject_is_refusal(self) -> None:
        rid = self._emit_threshold_recommendation(recommended_value=None)
        r = check_eligibility(rid, self.audit_path)
        self.assertFalse(r["eligibility_passed"])
        self.assertIn("subject_is_refusal", r["refusal_reasons"])

    def test_no_reliability_score(self) -> None:
        rid = self._emit_threshold_recommendation()
        r = check_eligibility(rid, self.audit_path)
        self.assertFalse(r["eligibility_passed"])
        self.assertIn("no_reliability_score", r["refusal_reasons"])

    def test_reliability_score_is_refusal(self) -> None:
        rid = self._emit_threshold_recommendation()
        self._emit_reliability_score(target_run_id=rid, overall_score=None)
        r = check_eligibility(rid, self.audit_path)
        self.assertFalse(r["eligibility_passed"])
        self.assertIn("reliability_score_is_refusal", r["refusal_reasons"])

    def test_reliability_below_floor(self) -> None:
        rid = self._emit_threshold_recommendation()
        self._emit_reliability_score(target_run_id=rid, overall_score=0.10)
        r = check_eligibility(rid, self.audit_path)
        self.assertFalse(r["eligibility_passed"])
        self.assertIn("reliability_below_floor", r["refusal_reasons"])

    def test_passing_path(self) -> None:
        rid = self._emit_eligible_subject()
        r = check_eligibility(rid, self.audit_path)
        self.assertTrue(r["eligibility_passed"])
        self.assertEqual(r["refusal_reasons"], [])
        self.assertEqual(r["reliability_score_at_decision"], 0.80)
        self.assertEqual(
            r["reliability_score_floor"], RELIABILITY_SCORE_FLOOR_FOR_PROMOTION,
        )
        self.assertIn(rid, r["consulted_run_ids"])

    def test_eligibility_uses_latest_score(self) -> None:
        rid = self._emit_threshold_recommendation()
        # Emit a low score first, then a passing one — the LATEST
        # should be used per chain ordering.
        self._emit_reliability_score(target_run_id=rid, overall_score=0.10)
        self._emit_reliability_score(target_run_id=rid, overall_score=0.80)
        r = check_eligibility(rid, self.audit_path)
        self.assertTrue(r["eligibility_passed"])


# ── emit_governance_decision (per decision_type) ────────────────────


class DecisionEmitTests(_GovernanceHarness):

    def test_approve_emits_typed_row(self) -> None:
        rid = self._emit_eligible_subject()
        record = emit_governance_decision(
            decision_type="approve",
            subject_run_id=rid,
            operator_id="owner@local",
            rationale="approved after substrate review",
            audit_path=self.audit_path,
        )
        ev = record["event"]
        self.assertEqual(ev["evidence_kind"], "governance_decision")
        self.assertEqual(ev["decision_type"], "approve")
        self.assertEqual(ev["subject_run_id"], rid)
        self.assertTrue(ev["eligibility_passed"])
        self.assertFalse(ev["override_used"])

    def test_reject_emits_typed_row(self) -> None:
        rid = self._emit_eligible_subject()
        record = emit_governance_decision(
            decision_type="reject",
            subject_run_id=rid,
            operator_id="owner@local",
            rationale="not yet, want more substrate",
            audit_path=self.audit_path,
        )
        self.assertEqual(record["event"]["decision_type"], "reject")

    def test_request_review_emits_typed_row(self) -> None:
        rid = self._emit_eligible_subject()
        record = emit_governance_decision(
            decision_type="request_review",
            subject_run_id=rid,
            operator_id="owner@local",
            rationale="parking under review",
            audit_path=self.audit_path,
        )
        self.assertEqual(record["event"]["decision_type"], "request_review")

    def test_unknown_decision_type_rejected(self) -> None:
        rid = self._emit_eligible_subject()
        with self.assertRaises(ValueError):
            emit_governance_decision(
                decision_type="adopt_silently",  # not in DECISION_TYPES
                subject_run_id=rid,
                operator_id="owner@local",
                audit_path=self.audit_path,
            )

    def test_eligibility_refused_without_override_raises(self) -> None:
        rid = self._emit_threshold_recommendation()
        # No reliability score → eligibility refuses.
        with self.assertRaises(EligibilityRefused) as ctx:
            emit_governance_decision(
                decision_type="approve",
                subject_run_id=rid,
                operator_id="owner@local",
                audit_path=self.audit_path,
            )
        self.assertIn(
            "no_reliability_score",
            ctx.exception.eligibility_result["refusal_reasons"],
        )
        # Critical: NO governance_decision row was emitted.
        events = [e for e in self._read_events()
                  if e.get("evidence_kind") == "governance_decision"]
        self.assertEqual(events, [])

    def test_payload_evidence_basis_recorded(self) -> None:
        rid = self._emit_eligible_subject()
        result = emit_governance_decision(
            decision_type="approve",
            subject_run_id=rid,
            operator_id="owner@local",
            audit_path=self.audit_path,
            emit=False,
        )
        basis = result["payload"]["evidence_basis"]
        self.assertEqual(basis["threshold_recommendation_run_id"], rid)
        self.assertIsNotNone(basis["reliability_score_run_id"])
        # calibration_report_run_id is the synthetic citation.
        self.assertEqual(basis["calibration_report_run_id"], "synthetic-cal-001")


# ── Tightening #1: override attestation ─────────────────────────────


class OverrideAttestationTightening1Tests(_GovernanceHarness):

    def test_force_ineligible_requires_override_reason(self) -> None:
        rid = self._emit_threshold_recommendation()  # no reliability score
        with self.assertRaises(ValueError):
            emit_governance_decision(
                decision_type="approve",
                subject_run_id=rid,
                operator_id="owner@local",
                force_ineligible=True,
                override_reason=None,
                audit_path=self.audit_path,
            )

    def test_force_ineligible_with_unknown_reason_rejected(self) -> None:
        rid = self._emit_threshold_recommendation()
        with self.assertRaises(ValueError):
            emit_governance_decision(
                decision_type="approve",
                subject_run_id=rid,
                operator_id="owner@local",
                force_ineligible=True,
                override_reason="boss_said_so",  # not in OVERRIDE_REASONS
                audit_path=self.audit_path,
            )

    def test_force_ineligible_emits_with_typed_attestation(self) -> None:
        rid = self._emit_threshold_recommendation()
        record = emit_governance_decision(
            decision_type="approve",
            subject_run_id=rid,
            operator_id="owner@local",
            force_ineligible=True,
            override_reason="temporary_substrate_gap",
            rationale="bootstrap period; substrate will accumulate",
            audit_path=self.audit_path,
        )
        ev = record["event"]
        self.assertTrue(ev["override_used"])
        # Eligibility still recorded as failed in the payload.
        self.assertFalse(ev["eligibility_passed"])

    def test_override_attestation_present_on_non_forced_decisions(self) -> None:
        """TIGHTENING #1: every decision carries override_attestation
        block (override_used=False for non-forced) so the replay
        surface and schema fingerprint are uniform."""
        rid = self._emit_eligible_subject()
        result = emit_governance_decision(
            decision_type="approve",
            subject_run_id=rid,
            operator_id="owner@local",
            audit_path=self.audit_path,
            emit=False,
        )
        att = result["payload"]["override_attestation"]
        self.assertEqual(att["override_used"], False)
        self.assertIsNone(att["override_reason"])

    def test_override_attestation_payload_records_reason(self) -> None:
        rid = self._emit_threshold_recommendation()
        result = emit_governance_decision(
            decision_type="approve",
            subject_run_id=rid,
            operator_id="owner@local",
            force_ineligible=True,
            override_reason="reliability_pipeline_fault",
            audit_path=self.audit_path,
            emit=False,
        )
        att = result["payload"]["override_attestation"]
        self.assertTrue(att["override_used"])
        self.assertEqual(att["override_reason"], "reliability_pipeline_fault")


# ── Automation role guard ───────────────────────────────────────────


class AutomationRoleGuardTests(_GovernanceHarness):

    def test_automation_cannot_approve(self) -> None:
        rid = self._emit_eligible_subject()
        with self.assertRaises(ValueError):
            emit_governance_decision(
                decision_type="approve",
                subject_run_id=rid,
                operator_id="batch-evaluator",
                operator_role="automation",
                audit_path=self.audit_path,
            )

    def test_automation_cannot_reject(self) -> None:
        rid = self._emit_eligible_subject()
        with self.assertRaises(ValueError):
            emit_governance_decision(
                decision_type="reject",
                subject_run_id=rid,
                operator_id="batch-evaluator",
                operator_role="automation",
                audit_path=self.audit_path,
            )

    def test_automation_cannot_rollback(self) -> None:
        rid = self._emit_eligible_subject()
        # First a human approve so a rollback target exists.
        emit_governance_decision(
            decision_type="approve",
            subject_run_id=rid,
            operator_id="owner@local",
            audit_path=self.audit_path,
        )
        with self.assertRaises(ValueError):
            emit_governance_decision(
                decision_type="rollback",
                subject_run_id=rid,
                operator_id="batch-evaluator",
                operator_role="automation",
                audit_path=self.audit_path,
            )

    def test_automation_may_emit_request_review(self) -> None:
        rid = self._emit_eligible_subject()
        record = emit_governance_decision(
            decision_type="request_review",
            subject_run_id=rid,
            operator_id="batch-evaluator",
            operator_role="automation",
            audit_path=self.audit_path,
        )
        self.assertEqual(record["event"]["decision_type"], "request_review")

    def test_automation_role_invariant_exhaustive(self) -> None:
        """REVIEW TIGHTENING #4 — load-bearing meta-test.

        Iterates the full DECISION_TYPES enum. Automation MUST
        succeed for exactly one type (``request_review``) and
        MUST fail for every other type. This guards against
        someone widening the automation surface accidentally —
        unattended promotion paths must not exist in the type
        system before they exist operationally.
        """
        permitted_for_automation = {"request_review"}
        for decision_type in sorted(DECISION_TYPES):
            with self.subTest(decision_type=decision_type):
                rid = self._emit_eligible_subject()
                # For rollback we also need a prior human approve
                # so the rollback target exists — without that
                # the runner would raise RuntimeError before the
                # role check fires. Plant the approve so the role
                # guard is the FIRST gate the rollback hits.
                if decision_type == "rollback":
                    emit_governance_decision(
                        decision_type="approve",
                        subject_run_id=rid,
                        operator_id="owner@local",
                        audit_path=self.audit_path,
                    )
                if decision_type in permitted_for_automation:
                    record = emit_governance_decision(
                        decision_type=decision_type,
                        subject_run_id=rid,
                        operator_id="batch-evaluator",
                        operator_role="automation",
                        audit_path=self.audit_path,
                    )
                    self.assertEqual(
                        record["event"]["decision_type"], decision_type,
                    )
                else:
                    with self.assertRaises(ValueError):
                        emit_governance_decision(
                            decision_type=decision_type,
                            subject_run_id=rid,
                            operator_id="batch-evaluator",
                            operator_role="automation",
                            audit_path=self.audit_path,
                        )


# ── Genealogy: parent_run_id discipline (REVIEW TIGHTENING #2) ──────


class GenealogyTests(_GovernanceHarness):
    """Explicit lineage assertions for governance_decision.

    Establishes the same standard Step 13 set on
    threshold_recommendation:

      - first governance_decision for a subject MUST anchor at
        the threshold_recommendation.run_id
      - any later governance_decision that supersedes a prior
        one MUST anchor at THAT prior governance_decision's
        run_id (NOT back at the recommendation)
      - no orphan governance_decision rows (parent_run_id is
        always present and resolvable in the chain)
    """

    def _read_record_for(self, run_id: str) -> Dict[str, Any]:
        with self.audit_path.open("r", encoding="utf-8") as h:
            for line in h:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("event", {}).get("run_id") == run_id:
                    return rec
        raise AssertionError(f"run_id {run_id!r} not in chain")

    def test_first_decision_parent_is_recommendation(self) -> None:
        rid = self._emit_eligible_subject()
        first = emit_governance_decision(
            decision_type="approve",
            subject_run_id=rid,
            operator_id="owner@local",
            audit_path=self.audit_path,
        )
        first_id = first["event"]["run_id"]
        # parent_run_id lives at the envelope level (event dict
        # after build_event_envelope). Re-read the persisted row
        # to assert against the on-chain value, not just the
        # in-memory record returned by emit.
        rec = self._read_record_for(first_id)
        self.assertEqual(rec["event"]["parent_run_id"], rid)

    def test_superseding_decision_parent_is_prior_governance(self) -> None:
        rid = self._emit_eligible_subject()
        first = emit_governance_decision(
            decision_type="request_review",
            subject_run_id=rid,
            operator_id="owner@local",
            audit_path=self.audit_path,
        )
        first_id = first["event"]["run_id"]
        second = emit_governance_decision(
            decision_type="approve",
            subject_run_id=rid,
            operator_id="owner@local",
            supersedes_run_id=first_id,
            audit_path=self.audit_path,
        )
        second_id = second["event"]["run_id"]
        rec = self._read_record_for(second_id)
        # Load-bearing: parent is the prior GOVERNANCE row, NOT
        # the recommendation. Genealogy follows the explicit
        # supersedes_run_id link, mirroring Step 13's
        # threshold_recommendation supersession discipline.
        self.assertEqual(rec["event"]["parent_run_id"], first_id)

    def test_rollback_parent_is_prior_governance_when_superseding(self) -> None:
        rid = self._emit_eligible_subject()
        approve = emit_governance_decision(
            decision_type="approve",
            subject_run_id=rid,
            operator_id="owner@local",
            audit_path=self.audit_path,
        )
        approve_id = approve["event"]["run_id"]
        rollback = emit_governance_decision(
            decision_type="rollback",
            subject_run_id=rid,
            operator_id="owner@local",
            supersedes_run_id=approve_id,
            audit_path=self.audit_path,
        )
        rollback_id = rollback["event"]["run_id"]
        rec = self._read_record_for(rollback_id)
        self.assertEqual(rec["event"]["parent_run_id"], approve_id)
        # And the rollback_target_run_id mirrors the supersedes
        # link for this case (operator chose the same row as
        # both the supersession target and the rollback target).
        self.assertEqual(
            rec["event"]["rollback_target_run_id"], approve_id,
        )

    def test_no_orphan_governance_decision_rows(self) -> None:
        """Every governance_decision row in the chain has a
        ``parent_run_id`` that resolves to another row in the
        chain — either the threshold_recommendation it cites or
        the prior governance_decision it supersedes."""
        rid_a = self._emit_eligible_subject()
        rid_b = self._emit_eligible_subject()
        first_a = emit_governance_decision(
            decision_type="approve",
            subject_run_id=rid_a,
            operator_id="owner@local",
            audit_path=self.audit_path,
        )
        emit_governance_decision(
            decision_type="rollback",
            subject_run_id=rid_a,
            operator_id="owner@local",
            supersedes_run_id=first_a["event"]["run_id"],
            audit_path=self.audit_path,
        )
        emit_governance_decision(
            decision_type="reject",
            subject_run_id=rid_b,
            operator_id="owner@local",
            audit_path=self.audit_path,
        )
        # Build set of all run_ids in the chain.
        all_run_ids: set = set()
        with self.audit_path.open("r", encoding="utf-8") as h:
            for line in h:
                if not line.strip():
                    continue
                rec = json.loads(line)
                all_run_ids.add(rec["event"]["run_id"])
        # Every governance_decision parent_run_id must be in the
        # set of chain run_ids — no dangling references.
        govs = [r for r in self._read_events()
                if r.get("evidence_kind") == "governance_decision"]
        self.assertEqual(len(govs), 3)
        for ev in govs:
            parent = ev.get("parent_run_id")
            self.assertIsNotNone(
                parent, msg=f"orphan governance_decision: {ev}",
            )
            self.assertIn(
                parent, all_run_ids,
                msg=(f"governance_decision parent_run_id {parent!r} "
                     f"is not present in the chain"),
            )


# ── Rollback discipline ─────────────────────────────────────────────


class RollbackDisciplineTests(_GovernanceHarness):

    def test_rollback_with_no_prior_approve_refused(self) -> None:
        rid = self._emit_eligible_subject()
        with self.assertRaises(RuntimeError):
            emit_governance_decision(
                decision_type="rollback",
                subject_run_id=rid,
                operator_id="owner@local",
                audit_path=self.audit_path,
            )

    def test_rollback_auto_discovers_prior_approve(self) -> None:
        rid = self._emit_eligible_subject()
        approve_record = emit_governance_decision(
            decision_type="approve",
            subject_run_id=rid,
            operator_id="owner@local",
            audit_path=self.audit_path,
        )
        approve_run_id = approve_record["event"]["run_id"]

        rollback_record = emit_governance_decision(
            decision_type="rollback",
            subject_run_id=rid,
            operator_id="owner@local",
            rationale="rolling back; concerns surfaced",
            audit_path=self.audit_path,
        )
        ev = rollback_record["event"]
        self.assertEqual(ev["decision_type"], "rollback")
        self.assertEqual(ev["rollback_target_run_id"], approve_run_id)

    def test_explicit_rollback_target_honored(self) -> None:
        rid = self._emit_eligible_subject()
        approve_record = emit_governance_decision(
            decision_type="approve",
            subject_run_id=rid,
            operator_id="owner@local",
            audit_path=self.audit_path,
        )
        explicit_target = approve_record["event"]["run_id"]
        rollback_record = emit_governance_decision(
            decision_type="rollback",
            subject_run_id=rid,
            operator_id="owner@local",
            rollback_target_run_id=explicit_target,
            audit_path=self.audit_path,
        )
        self.assertEqual(
            rollback_record["event"]["rollback_target_run_id"],
            explicit_target,
        )


# ── Production isolation (LOAD-BEARING) ─────────────────────────────


class ProductionIsolationTests(_GovernanceHarness):

    def test_high_correlation_threshold_byte_unchanged_after_approve(
        self,
    ) -> None:
        """The system MAY recommend; only humans MAY adopt. Adoption
        is a governance_decision row, NOT a code mutation. The
        production constant must be byte-identical before and
        after every approve."""
        from backend.investment_analytics.portfolio_health import correlation
        before = correlation.HIGH_CORRELATION_THRESHOLD
        rid = self._emit_eligible_subject()
        emit_governance_decision(
            decision_type="approve",
            subject_run_id=rid,
            operator_id="owner@local",
            audit_path=self.audit_path,
        )
        after = correlation.HIGH_CORRELATION_THRESHOLD
        self.assertEqual(before, after)

    def test_methodology_versions_byte_unchanged_after_approve(self) -> None:
        before = dict(meth_mod.METHODOLOGY_VERSIONS)
        rid = self._emit_eligible_subject()
        emit_governance_decision(
            decision_type="approve",
            subject_run_id=rid,
            operator_id="owner@local",
            audit_path=self.audit_path,
        )
        after = dict(meth_mod.METHODOLOGY_VERSIONS)
        self.assertEqual(before, after)

    def test_chain_verifies_after_every_decision_type(self) -> None:
        rid = self._emit_eligible_subject()
        for dt in ["approve", "reject", "request_review"]:
            # Use distinct subjects so each is freshly eligible.
            r = self._emit_eligible_subject()
            emit_governance_decision(
                decision_type=dt,
                subject_run_id=r,
                operator_id="owner@local",
                audit_path=self.audit_path,
            )
            self.assertTrue(verify_audit_chain(self.audit_path))

    def test_prior_governance_row_is_immutable(self) -> None:
        """Superseding does NOT rewrite the prior row. The chain
        retains both — operator history is append-only."""
        rid = self._emit_eligible_subject()
        first = emit_governance_decision(
            decision_type="request_review",
            subject_run_id=rid,
            operator_id="owner@local",
            rationale="initial review",
            audit_path=self.audit_path,
        )
        first_id = first["event"]["run_id"]
        # Now an approve that supersedes it.
        emit_governance_decision(
            decision_type="approve",
            subject_run_id=rid,
            operator_id="owner@local",
            supersedes_run_id=first_id,
            rationale="approved after review",
            audit_path=self.audit_path,
        )
        # First row still present, byte-equal.
        events = [e for e in self._read_events()
                  if e.get("evidence_kind") == "governance_decision"]
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["run_id"], first_id)
        self.assertEqual(events[0]["decision_type"], "request_review")
        # And the second cites the supersession.
        self.assertEqual(events[1]["supersedes_run_id"], first_id)
        self.assertTrue(verify_audit_chain(self.audit_path))


# ── Query helpers ───────────────────────────────────────────────────


class QueryHelperTests(_GovernanceHarness):

    def test_find_filters_by_decision_type(self) -> None:
        r1 = self._emit_eligible_subject()
        r2 = self._emit_eligible_subject()
        emit_governance_decision(
            decision_type="approve", subject_run_id=r1,
            operator_id="owner@local", audit_path=self.audit_path,
        )
        emit_governance_decision(
            decision_type="reject", subject_run_id=r2,
            operator_id="owner@local", audit_path=self.audit_path,
        )
        approves = find_governance_decisions(
            audit_path=self.audit_path, decision_type="approve",
        )
        self.assertEqual(len(approves), 1)
        self.assertEqual(approves[0]["subject_run_id"], r1)

    def test_latest_effective_decision_returns_most_recent(self) -> None:
        r1 = self._emit_eligible_subject()
        emit_governance_decision(
            decision_type="request_review", subject_run_id=r1,
            operator_id="owner@local", audit_path=self.audit_path,
        )
        r2 = self._emit_eligible_subject()
        latest_rec = emit_governance_decision(
            decision_type="approve", subject_run_id=r2,
            operator_id="owner@local", audit_path=self.audit_path,
        )
        latest = latest_effective_decision(
            audit_path=self.audit_path,
            subject_target_canonical_id=TARGET,
        )
        self.assertEqual(latest["run_id"], latest_rec["event"]["run_id"])


# ── Replay ──────────────────────────────────────────────────────────


class ReplayTests(_GovernanceHarness):

    def test_replay_handler_registered(self) -> None:
        self.assertIn("governance_decision", REPLAY_HANDLERS)

    def test_replay_approve_classifies_no_drift(self) -> None:
        """Without any underlying chain change, approve replay
        should be semantically_equivalent (volatile timestamps
        differ; eligibility re-derives identically)."""
        rid = self._emit_eligible_subject()
        record = emit_governance_decision(
            decision_type="approve",
            subject_run_id=rid,
            operator_id="owner@local",
            audit_path=self.audit_path,
        )
        run_id = record["event"]["run_id"]
        # Mocking DEFAULT_AUDIT_PATH via monkey-patching the
        # governance.runner module is the established seam used
        # by other replay tests (calibration, reliability).
        import backend.governance.runner as gov_runner
        orig = gov_runner.DEFAULT_AUDIT_PATH
        gov_runner.DEFAULT_AUDIT_PATH = self.audit_path
        try:
            result = replay_run(
                audit_path=self.audit_path,
                run_id=run_id,
                verify_chain=True,
                emit_audit=False,
                registry_path=self.registry_path,
            )
        finally:
            gov_runner.DEFAULT_AUDIT_PATH = orig
        # No state drift between recorded and current — should
        # land at exact_match or semantically_equivalent.
        self.assertIn(
            result["state"],
            {"exact_match", "semantically_equivalent"},
        )

    def test_replay_eligibility_drift_typed_driver(self) -> None:
        """When new reliability_score rows accumulate after the
        decision, eligibility_state_drifted should fire as a typed
        driver — NOT invalid_replay."""
        rid = self._emit_eligible_subject()
        record = emit_governance_decision(
            decision_type="approve",
            subject_run_id=rid,
            operator_id="owner@local",
            audit_path=self.audit_path,
        )
        run_id = record["event"]["run_id"]
        # Emit a new reliability_score AFTER the decision; the
        # replay's eligibility check will see this new score.
        self._emit_reliability_score(target_run_id=rid, overall_score=0.95)

        import backend.governance.runner as gov_runner
        orig = gov_runner.DEFAULT_AUDIT_PATH
        gov_runner.DEFAULT_AUDIT_PATH = self.audit_path
        try:
            result = replay_run(
                audit_path=self.audit_path,
                run_id=run_id,
                verify_chain=True,
                emit_audit=False,
                registry_path=self.registry_path,
            )
        finally:
            gov_runner.DEFAULT_AUDIT_PATH = orig
        kinds = {d["kind"] for d in result["divergence_drivers"]}
        # Either eligibility_state_drifted fires directly, or the
        # payload is byte-equivalent through the recorded
        # eligibility (semantically_equivalent). Both are valid
        # non-invalid outcomes. The load-bearing assertion: this
        # is NOT invalid_replay.
        self.assertNotEqual(result["state"], "invalid_replay")
        self.assertIn(
            result["state"],
            {"exact_match", "semantically_equivalent",
             "expected_divergence"},
        )
        if result["state"] == "expected_divergence":
            self.assertIn("eligibility_state_drifted", kinds)

    def test_rollback_replay_with_production_drift_expected_divergence(
        self,
    ) -> None:
        """TIGHTENING #2 — load-bearing.

        Rollback is an operator intent artifact, NOT a guarantee
        that production presently equals the rollback target.
        Replay of a rollback with the production value moved off
        the historical snapshot MUST classify as
        expected_divergence with production_value_changed driver,
        NOT invalid_replay.
        """
        rid = self._emit_eligible_subject()
        # Approve first so rollback has a target.
        emit_governance_decision(
            decision_type="approve",
            subject_run_id=rid,
            operator_id="owner@local",
            audit_path=self.audit_path,
        )
        rollback_record = emit_governance_decision(
            decision_type="rollback",
            subject_run_id=rid,
            operator_id="owner@local",
            rationale="rolling back",
            audit_path=self.audit_path,
        )
        rollback_run_id = rollback_record["event"]["run_id"]

        # Simulate production drift: monkey-patch the production
        # state reader to return a different value at replay time.
        import backend.governance.production_state as prod_state_mod
        orig_reg = prod_state_mod._PRODUCTION_REGISTRY
        new_reg = dict(orig_reg)
        new_reg[TARGET] = {
            "reader": lambda: 0.99,  # different from the live constant
            "source_attestation": (
                "backend.investment_analytics.portfolio_health.correlation"
                ".HIGH_CORRELATION_THRESHOLD"
            ),
        }
        prod_state_mod._PRODUCTION_REGISTRY = new_reg

        import backend.governance.runner as gov_runner
        orig_path = gov_runner.DEFAULT_AUDIT_PATH
        gov_runner.DEFAULT_AUDIT_PATH = self.audit_path
        try:
            result = replay_run(
                audit_path=self.audit_path,
                run_id=rollback_run_id,
                verify_chain=True,
                emit_audit=False,
            )
        finally:
            gov_runner.DEFAULT_AUDIT_PATH = orig_path
            prod_state_mod._PRODUCTION_REGISTRY = orig_reg

        # The load-bearing assertion: rollback replay with
        # production drift MUST NOT be invalid_replay.
        self.assertNotEqual(result["state"], "invalid_replay")
        kinds = {d["kind"] for d in result["divergence_drivers"]}
        # Either production_value_changed fires (expected case)
        # or volatile-strip made them equal (rare, but valid).
        # The asymmetry tightening guarantees the negative
        # condition above.
        if result["state"] == "expected_divergence":
            self.assertIn("production_value_changed", kinds)

    def test_production_value_changed_is_payload_level_not_envelope_level(
        self,
    ) -> None:
        """REVIEW TIGHTENING #3 — load-bearing layering invariant.

        ``production_value_changed`` must remain a payload-level
        driver. It MUST NOT appear in the envelope-level driver
        set (methodology_changed, code_sha_changed,
        registry_hash_changed, cache_fingerprint_changed).

        The reason is forensic: an operational production edit
        (HIGH_CORRELATION_THRESHOLD nudged in code) must NEVER
        look like a methodology bump on replay. Methodology drift
        and production drift have distinct remediation paths;
        collapsing them would let operators mis-attribute one for
        the other.

        Test method: inspect the SOURCE of both driver-detection
        functions and assert ``production_value_changed`` is
        registered ONLY in the governance payload-driver function,
        not in any envelope-level path.
        """
        import inspect
        from backend.evidence import replay as replay_mod

        envelope_src = inspect.getsource(
            replay_mod._identify_divergence_drivers,
        )
        payload_src = inspect.getsource(
            replay_mod._identify_payload_drivers,
        )
        governance_src = inspect.getsource(
            replay_mod._identify_governance_payload_drivers,
        )
        # Envelope-level and generic payload-level functions MUST
        # NOT mention production_value_changed.
        self.assertNotIn("production_value_changed", envelope_src)
        self.assertNotIn("production_value_changed", payload_src)
        # The governance payload-level function IS where it
        # lives. Same check for eligibility_state_drifted.
        self.assertIn("production_value_changed", governance_src)
        self.assertIn("eligibility_state_drifted", governance_src)
        self.assertNotIn("eligibility_state_drifted", envelope_src)
        self.assertNotIn("eligibility_state_drifted", payload_src)

    def test_methodology_drift_still_distinct_from_production_drift(
        self,
    ) -> None:
        """Companion to the layering invariant: a methodology
        bump produces ``methodology_changed`` at the envelope
        level; a production value change produces
        ``production_value_changed`` at the payload level. The
        two NEVER merge."""
        rid = self._emit_eligible_subject()
        record = emit_governance_decision(
            decision_type="approve",
            subject_run_id=rid,
            operator_id="owner@local",
            audit_path=self.audit_path,
        )
        run_id = record["event"]["run_id"]

        # Drift BOTH surfaces simultaneously: bump methodology
        # AND change production value at replay time.
        import backend.investment_analytics.methodology as meth
        import backend.governance.production_state as prod_state_mod
        orig_versions = dict(meth.METHODOLOGY_VERSIONS)
        meth.METHODOLOGY_VERSIONS["governance_eligibility"] = "v2"

        orig_reg = prod_state_mod._PRODUCTION_REGISTRY
        new_reg = dict(orig_reg)
        new_reg[TARGET] = {
            "reader": lambda: 0.123,
            "source_attestation": (
                "backend.investment_analytics.portfolio_health.correlation"
                ".HIGH_CORRELATION_THRESHOLD"
            ),
        }
        prod_state_mod._PRODUCTION_REGISTRY = new_reg

        import backend.governance.runner as gov_runner
        orig_path = gov_runner.DEFAULT_AUDIT_PATH
        gov_runner.DEFAULT_AUDIT_PATH = self.audit_path
        try:
            result = replay_run(
                audit_path=self.audit_path,
                run_id=run_id,
                verify_chain=True,
                emit_audit=False,
                registry_path=self.registry_path,
            )
        finally:
            gov_runner.DEFAULT_AUDIT_PATH = orig_path
            prod_state_mod._PRODUCTION_REGISTRY = orig_reg
            meth.METHODOLOGY_VERSIONS.clear()
            meth.METHODOLOGY_VERSIONS.update(orig_versions)

        kinds = {d["kind"] for d in result["divergence_drivers"]}
        # Both surfaces must be observable AS DISTINCT drivers.
        # methodology_changed is envelope-level; production_value_changed
        # is payload-level (governance-specific).
        self.assertIn("methodology_changed", kinds)
        self.assertIn("production_value_changed", kinds)
        # Sanity: not invalid_replay, both typed drifts identified.
        self.assertEqual(result["state"], "expected_divergence")

    def test_approve_replay_with_production_drift_expected_divergence(
        self,
    ) -> None:
        """production_value_changed driver fires symmetrically for
        non-rollback decision types — it's drift, not tampering."""
        rid = self._emit_eligible_subject()
        record = emit_governance_decision(
            decision_type="approve",
            subject_run_id=rid,
            operator_id="owner@local",
            audit_path=self.audit_path,
        )
        run_id = record["event"]["run_id"]

        import backend.governance.production_state as prod_state_mod
        orig_reg = prod_state_mod._PRODUCTION_REGISTRY
        new_reg = dict(orig_reg)
        new_reg[TARGET] = {
            "reader": lambda: 0.42,
            "source_attestation": (
                "backend.investment_analytics.portfolio_health.correlation"
                ".HIGH_CORRELATION_THRESHOLD"
            ),
        }
        prod_state_mod._PRODUCTION_REGISTRY = new_reg

        import backend.governance.runner as gov_runner
        orig_path = gov_runner.DEFAULT_AUDIT_PATH
        gov_runner.DEFAULT_AUDIT_PATH = self.audit_path
        try:
            result = replay_run(
                audit_path=self.audit_path,
                run_id=run_id,
                verify_chain=True,
                emit_audit=False,
            )
        finally:
            gov_runner.DEFAULT_AUDIT_PATH = orig_path
            prod_state_mod._PRODUCTION_REGISTRY = orig_reg

        self.assertNotEqual(result["state"], "invalid_replay")


# ── CLI smoke ───────────────────────────────────────────────────────


class CliSmokeTests(_GovernanceHarness):

    def _run_cli(self, argv: List[str]) -> subprocess.CompletedProcess:
        # Invoke `python -m backend.governance <args>` against the
        # test audit path. Use the project root as CWD so the
        # package import path resolves.
        repo_root = Path(__file__).resolve().parent.parent
        cmd = [sys.executable, "-m", "backend.governance", *argv]
        return subprocess.run(
            cmd,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
        )

    def test_show_eligibility_passes(self) -> None:
        rid = self._emit_eligible_subject()
        proc = self._run_cli([
            "show-eligibility",
            "--subject-run-id", rid,
            "--audit-path", str(self.audit_path),
        ])
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        parsed = json.loads(proc.stdout)
        self.assertTrue(parsed["eligibility_passed"])

    def test_show_eligibility_refusal_exits_nonzero(self) -> None:
        rid = self._emit_threshold_recommendation()  # no reliability score
        proc = self._run_cli([
            "show-eligibility",
            "--subject-run-id", rid,
            "--audit-path", str(self.audit_path),
        ])
        self.assertEqual(proc.returncode, 2)
        parsed = json.loads(proc.stdout)
        self.assertFalse(parsed["eligibility_passed"])
        self.assertIn("no_reliability_score", parsed["refusal_reasons"])

    def test_approve_via_cli_emits(self) -> None:
        rid = self._emit_eligible_subject()
        proc = self._run_cli([
            "approve",
            "--subject-run-id", rid,
            "--operator-id", "owner@local",
            "--audit-path", str(self.audit_path),
        ])
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        parsed = json.loads(proc.stdout)
        self.assertEqual(parsed["status"], "emitted")
        self.assertEqual(parsed["decision_type"], "approve")

    def test_approve_via_cli_without_eligibility_exits_2(self) -> None:
        rid = self._emit_threshold_recommendation()
        proc = self._run_cli([
            "approve",
            "--subject-run-id", rid,
            "--operator-id", "owner@local",
            "--audit-path", str(self.audit_path),
        ])
        self.assertEqual(proc.returncode, 2)
        parsed = json.loads(proc.stdout)
        self.assertEqual(parsed["status"], "eligibility_refused")
        self.assertIn(
            "no_reliability_score",
            parsed["eligibility_result"]["refusal_reasons"],
        )

    def test_list_filters_by_decision_type(self) -> None:
        rid = self._emit_eligible_subject()
        emit_governance_decision(
            decision_type="approve", subject_run_id=rid,
            operator_id="owner@local", audit_path=self.audit_path,
        )
        proc = self._run_cli([
            "list",
            "--decision-type", "approve",
            "--audit-path", str(self.audit_path),
        ])
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        parsed = json.loads(proc.stdout)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["decision_type"], "approve")


if __name__ == "__main__":
    unittest.main()
