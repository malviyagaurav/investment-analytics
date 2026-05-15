"""Tests for the provenance envelope on audit events.

Step 3 of the evidence-layer roadmap. Covers:
  - EVIDENCE_KINDS is a closed enum; None is explicitly valid.
  - METHODOLOGY_VERSIONS has all 8 components + schema version.
  - current_methodology returns a snapshot copy (mutating the
    return value does NOT affect the source).
  - Envelope is additive (caller's existing fields preserved).
  - run_id auto-generates as UUID4 by default; loose validation
    accepts other string formats.
  - parent_run_id validation; None is the no-parent case.
  - Provenance capture: code_sha / python_version / registry_hash.
  - append_audit_record auto-wraps every event with the envelope.
  - Envelope PII sanitization (provenance fields with PII keys are
    redacted along with the event body).
  - Methodology embedded in audit records is a SNAPSHOT, not a
    reference — mutating METHODOLOGY_VERSIONS post-write does NOT
    change already-written records.
  - Existing single-arg api/main.py call patterns still work
    unchanged after the envelope landing.
"""
from __future__ import annotations

import json
import unittest
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from backend.investment_analytics import methodology as methodology_module
from backend.investment_analytics.audit import (
    append_audit_record,
    verify_audit_chain,
    verify_audit_chain_diag,
)
from backend.investment_analytics.evidence_envelope import (
    ENVELOPE_SCHEMA_FINGERPRINT,
    ENVELOPE_SCHEMA_VERSION,
    EVIDENCE_KINDS,
    build_event_envelope,
    validate_evidence_kind,
)
from backend.investment_analytics.methodology import (
    METHODOLOGY_SCHEMA_VERSION,
    METHODOLOGY_VERSIONS,
    current_methodology,
)
from backend.investment_analytics.provenance import (
    _git_head_sha,
    capture_provenance_inputs,
)


class TestEvidenceKindsEnum(unittest.TestCase):

    def test_closed_enum_has_exactly_seven_values(self) -> None:
        self.assertEqual(len(EVIDENCE_KINDS), 7)

    def test_expected_values_present(self) -> None:
        expected = {
            "ranking_snapshot",
            "portfolio_health_snapshot",
            "watchlist_run",
            "experiment_run",
            "replay_result",
            "drift_analysis",
            "regime_summary",
        }
        self.assertEqual(EVIDENCE_KINDS, expected)

    def test_none_is_intentionally_valid(self) -> None:
        validate_evidence_kind(None)  # must not raise

    def test_unknown_kind_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_evidence_kind("not_a_kind")

    def test_non_string_kind_rejected(self) -> None:
        with self.assertRaises(TypeError):
            validate_evidence_kind(42)


class TestMethodology(unittest.TestCase):

    def test_nine_components_present(self) -> None:
        # Step 10 added "regime_classifier" — METHODOLOGY_SCHEMA_VERSION
        # bumped from v1 to v2 to reflect the structural change.
        expected = {
            "equity_metric", "debt_metric", "gold_metric",
            "confidence", "coverage_integrity", "alternative_gate",
            "correlation_detection", "decision_engine",
            "regime_classifier",
        }
        self.assertEqual(set(METHODOLOGY_VERSIONS.keys()), expected)

    def test_all_components_at_v1(self) -> None:
        for component, version in METHODOLOGY_VERSIONS.items():
            self.assertEqual(version, "v1", f"{component} != v1")

    def test_snapshot_includes_methodology_schema_version(self) -> None:
        snap = current_methodology()
        self.assertIn("methodology_schema_version", snap)
        self.assertEqual(snap["methodology_schema_version"], METHODOLOGY_SCHEMA_VERSION)

    def test_snapshot_is_a_copy_not_a_reference(self) -> None:
        snap1 = current_methodology()
        snap1["equity_metric"] = "TAMPERED"
        snap2 = current_methodology()
        self.assertEqual(snap2["equity_metric"], "v1")  # source unchanged
        self.assertNotEqual(snap1["equity_metric"], snap2["equity_metric"])


class TestEnvelopeSchemaFingerprint(unittest.TestCase):

    def test_envelope_schema_version_is_v1(self) -> None:
        self.assertEqual(ENVELOPE_SCHEMA_VERSION, "v1")

    def test_fingerprint_is_64char_hex(self) -> None:
        self.assertEqual(len(ENVELOPE_SCHEMA_FINGERPRINT), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in ENVELOPE_SCHEMA_FINGERPRINT))


class TestBuildEventEnvelope(unittest.TestCase):

    def test_envelope_is_additive(self) -> None:
        original = {"event_type": "rank_category", "subject_token": "tok", "extra": 1}
        env = build_event_envelope(original)
        # All original keys preserved at the same paths.
        for k, v in original.items():
            self.assertEqual(env[k], v)
        # New envelope fields present.
        for k in (
            "evidence_kind", "run_id", "parent_run_id", "generated_at",
            "methodology", "envelope_schema_version",
            "envelope_schema_fingerprint", "inputs",
        ):
            self.assertIn(k, env)

    def test_envelope_does_not_mutate_input(self) -> None:
        original = {"event_type": "x"}
        copy_for_check = dict(original)
        build_event_envelope(original)
        self.assertEqual(original, copy_for_check)

    def test_default_evidence_kind_is_none(self) -> None:
        env = build_event_envelope({"event_type": "x"})
        self.assertIsNone(env["evidence_kind"])

    def test_evidence_kind_validated(self) -> None:
        with self.assertRaises(ValueError):
            build_event_envelope({"event_type": "x"}, evidence_kind="garbage")

    def test_known_evidence_kind_accepted(self) -> None:
        env = build_event_envelope({"event_type": "x"}, evidence_kind="ranking_snapshot")
        self.assertEqual(env["evidence_kind"], "ranking_snapshot")

    def test_run_id_default_is_valid_uuid4(self) -> None:
        env = build_event_envelope({"event_type": "x"})
        # Loose check — parse as UUID; version=4 by default in this project.
        parsed = uuid.UUID(env["run_id"])
        self.assertEqual(parsed.version, 4)

    def test_explicit_non_uuid_run_id_accepted(self) -> None:
        # Per architecture: UUID4 is the default but not universally required.
        env = build_event_envelope({"event_type": "x"}, run_id="experiment-batch-42")
        self.assertEqual(env["run_id"], "experiment-batch-42")

    def test_empty_run_id_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_event_envelope({"event_type": "x"}, run_id="")

    def test_whitespace_run_id_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_event_envelope({"event_type": "x"}, run_id="has space")

    def test_overlong_run_id_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_event_envelope({"event_type": "x"}, run_id="x" * 129)

    def test_parent_run_id_none_is_default(self) -> None:
        env = build_event_envelope({"event_type": "x"})
        self.assertIsNone(env["parent_run_id"])

    def test_parent_run_id_validated_when_provided(self) -> None:
        with self.assertRaises(ValueError):
            build_event_envelope({"event_type": "x"}, parent_run_id="")

    def test_parent_run_id_passes_through_when_valid(self) -> None:
        env = build_event_envelope({"event_type": "x"}, parent_run_id="origin-run-abc")
        self.assertEqual(env["parent_run_id"], "origin-run-abc")

    def test_methodology_snapshot_present(self) -> None:
        env = build_event_envelope({"event_type": "x"})
        self.assertIn("methodology", env)
        self.assertIn("methodology_schema_version", env["methodology"])
        for component in METHODOLOGY_VERSIONS:
            self.assertIn(component, env["methodology"])

    def test_methodology_is_snapshot_not_reference(self) -> None:
        # Mutating the source AFTER envelope is built must NOT change
        # the snapshot embedded in the already-returned envelope.
        env = build_event_envelope({"event_type": "x"})
        original_value = env["methodology"]["equity_metric"]
        methodology_module.METHODOLOGY_VERSIONS["equity_metric"] = "vTEST"
        try:
            self.assertEqual(env["methodology"]["equity_metric"], original_value)
        finally:
            methodology_module.METHODOLOGY_VERSIONS["equity_metric"] = "v1"

    def test_fingerprint_matches_module_constant(self) -> None:
        env = build_event_envelope({"event_type": "x"})
        self.assertEqual(env["envelope_schema_fingerprint"], ENVELOPE_SCHEMA_FINGERPRINT)


class TestProvenanceCapture(unittest.TestCase):

    def test_capture_returns_expected_keys(self) -> None:
        inputs = capture_provenance_inputs()
        for k in (
            "code_sha", "python_version", "analyzer_version",
            "registry_hash", "registry_path", "cache_fingerprint",
        ):
            self.assertIn(k, inputs)

    def test_python_version_is_dotted_string(self) -> None:
        inputs = capture_provenance_inputs()
        parts = inputs["python_version"].split(".")
        self.assertEqual(len(parts), 3)
        for p in parts:
            self.assertTrue(p.isdigit())

    def test_code_sha_is_string(self) -> None:
        inputs = capture_provenance_inputs()
        self.assertIsInstance(inputs["code_sha"], str)
        self.assertTrue(len(inputs["code_sha"]) >= 7)  # at least short-SHA length, OR "unknown"

    def test_registry_hash_null_without_path(self) -> None:
        inputs = capture_provenance_inputs(registry_path=None)
        self.assertIsNone(inputs["registry_hash"])
        self.assertIsNone(inputs["registry_path"])

    def test_registry_hash_computed_when_path_provided(self) -> None:
        with TemporaryDirectory() as td:
            registry = Path(td) / "registry.json"
            registry.write_text('{"schemes": [{"code": 1}]}', encoding="utf-8")
            inputs = capture_provenance_inputs(registry_path=registry)
            self.assertEqual(len(inputs["registry_hash"]), 64)
            self.assertEqual(inputs["registry_path"], str(registry))

    def test_cache_fingerprint_null_at_this_layer(self) -> None:
        # cache_fingerprint is deferred to step 4. Step-3 callers MUST
        # see null so it's clear this isn't yet a guarantee.
        inputs = capture_provenance_inputs()
        self.assertIsNone(inputs["cache_fingerprint"])

    def test_git_head_sha_is_cached(self) -> None:
        # lru_cache(maxsize=1): two calls return the same string.
        sha1 = _git_head_sha()
        sha2 = _git_head_sha()
        self.assertEqual(sha1, sha2)


class TestAppendAuditRecordAutoWraps(unittest.TestCase):

    def test_existing_caller_signature_works(self) -> None:
        # Backward compat: callers that pass only path + event still
        # work; envelope is auto-added with evidence_kind=None.
        with TemporaryDirectory() as td:
            path = Path(td) / "audit.jsonl"
            record = append_audit_record(path, {"event_type": "legacy"})
            ev = record["event"]
            self.assertEqual(ev["event_type"], "legacy")  # original field
            self.assertIsNone(ev["evidence_kind"])         # default
            self.assertIn("methodology", ev)
            self.assertIn("run_id", ev)
            self.assertIn("inputs", ev)

    def test_evidence_kind_kwarg_propagates_to_record(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "audit.jsonl"
            record = append_audit_record(
                path,
                {"event_type": "rank_category"},
                evidence_kind="ranking_snapshot",
            )
            self.assertEqual(record["event"]["evidence_kind"], "ranking_snapshot")

    def test_envelope_does_not_break_chain(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "audit.jsonl"
            for i in range(5):
                append_audit_record(path, {"event_type": "x", "i": i})
            self.assertTrue(verify_audit_chain(path))
            diag = verify_audit_chain_diag(path)
            self.assertTrue(diag["valid"])
            self.assertEqual(diag["lines_scanned"], 5)

    def test_run_id_unique_across_calls(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "audit.jsonl"
            ids = set()
            for _ in range(50):
                r = append_audit_record(path, {"event_type": "x"})
                ids.add(r["event"]["run_id"])
            self.assertEqual(len(ids), 50)

    def test_explicit_run_id_honoured(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "audit.jsonl"
            r = append_audit_record(
                path, {"event_type": "x"}, run_id="custom-run-id"
            )
            self.assertEqual(r["event"]["run_id"], "custom-run-id")

    def test_parent_run_id_lineage(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "audit.jsonl"
            r1 = append_audit_record(path, {"event_type": "x"})
            r2 = append_audit_record(
                path, {"event_type": "x_rerun"},
                parent_run_id=r1["event"]["run_id"],
            )
            self.assertEqual(r2["event"]["parent_run_id"], r1["event"]["run_id"])


class TestEnvelopePiiSanitization(unittest.TestCase):

    def test_pii_in_inputs_redacted(self) -> None:
        # If a caller (incorrectly) puts PII in inputs, the PII keys
        # must still be redacted post-envelope-build.
        with TemporaryDirectory() as td:
            path = Path(td) / "audit.jsonl"
            inputs_with_pii = {
                "code_sha": "abc123",
                "email": "test@example.com",
                "user_id": "u-42",
                "python_version": "3.9.6",
                "analyzer_version": "mf_v2",
                "registry_hash": None,
                "registry_path": None,
                "cache_fingerprint": None,
            }
            record = append_audit_record(
                path,
                {"event_type": "x"},
                inputs=inputs_with_pii,
            )
            self.assertEqual(record["event"]["inputs"]["email"], "[redacted]")
            self.assertEqual(record["event"]["inputs"]["user_id"], "[redacted]")
            # Non-PII fields unchanged.
            self.assertEqual(record["event"]["inputs"]["code_sha"], "abc123")

    def test_pii_in_event_body_still_redacted(self) -> None:
        # The pre-existing PII guarantee on event body must hold even
        # after envelope wrap.
        with TemporaryDirectory() as td:
            path = Path(td) / "audit.jsonl"
            r = append_audit_record(
                path,
                {"event_type": "x", "email": "leak@example.com", "name": "Alice"},
            )
            self.assertEqual(r["event"]["email"], "[redacted]")
            self.assertEqual(r["event"]["name"], "[redacted]")


class TestMethodologySnapshotInChain(unittest.TestCase):

    def test_record_methodology_frozen_after_write(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "audit.jsonl"

            # Write a record at v1.
            r = append_audit_record(path, {"event_type": "before_bump"})
            self.assertEqual(r["event"]["methodology"]["equity_metric"], "v1")

            # Simulate a bump in the live process.
            methodology_module.METHODOLOGY_VERSIONS["equity_metric"] = "v2_simulated"
            try:
                # Read the written record from disk; methodology must
                # still be v1 because the snapshot was taken at write time.
                with path.open(encoding="utf-8") as handle:
                    on_disk = json.loads(handle.readline())
                self.assertEqual(
                    on_disk["event"]["methodology"]["equity_metric"], "v1",
                    "Written record's methodology must be a SNAPSHOT, "
                    "not a reference to METHODOLOGY_VERSIONS",
                )

                # A new record written after the bump carries v2_simulated.
                r2 = append_audit_record(path, {"event_type": "after_bump"})
                self.assertEqual(
                    r2["event"]["methodology"]["equity_metric"], "v2_simulated"
                )

                # Chain still verifies (each record's hash is over its own content).
                self.assertTrue(verify_audit_chain(path))
            finally:
                methodology_module.METHODOLOGY_VERSIONS["equity_metric"] = "v1"


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
