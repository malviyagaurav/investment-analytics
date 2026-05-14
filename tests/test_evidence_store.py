"""Tests for the by-reference evidence store + emit_evidence helper.

Step 4 of the evidence-layer roadmap. Covers:
  - write_evidence persists the sanitized envelope to
    data/evidence/<kind>/<run_id>.json and returns the correct
    evidence_ref shape ({path, sha256, size_bytes}).
  - The sha256 in evidence_ref matches the raw file bytes — not a
    re-canonicalized reparse — so replay can verify bit-for-bit.
  - size_bytes equals the persisted file size.
  - PII keys present in the payload (e.g. user_id, email, pan) are
    redacted in the evidence file itself, AND the redaction happens
    BEFORE hashing — so what's persisted == what's hashed.
  - Duplicate run_id raises FileExistsError (immutability — same-day
    reruns generate fresh run_ids per the architecture).
  - Missing run_id raises KeyError (caller programming error,
    surfaced loudly).
  - emit_evidence two-phase ordering: evidence file written FIRST,
    audit ref appended SECOND, both referencing the SAME run_id and
    the SAME captured inputs dict.
  - emit_evidence preserves the user's caller event shape: the
    audit record includes the lightweight audit_event fields PLUS
    the evidence_ref — not the heavy payload.
  - The audit record's run_id matches the evidence file's filename
    (so a replay tool can locate the evidence from the audit row).
"""
from __future__ import annotations

import hashlib
import json
import unittest
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from backend.evidence.store import emit_evidence, write_evidence
from backend.investment_analytics.audit import verify_audit_chain
from backend.investment_analytics.evidence_envelope import build_event_envelope


class TestWriteEvidenceBasics(unittest.TestCase):

    def _make_envelope(self, run_id=None, payload=None):
        return build_event_envelope(
            {"event_type": "rank_category", "payload": payload or {"ranked": [], "schema_version": "v1"}},
            evidence_kind="ranking_snapshot",
            run_id=run_id,
        )

    def test_write_creates_file_at_kind_subdir(self) -> None:
        with TemporaryDirectory() as tmp:
            audit_dir = Path(tmp) / "audit"
            audit_dir.mkdir()
            env = self._make_envelope()
            ref = write_evidence(audit_dir, "ranking_snapshot", env)
            target = Path(tmp) / "evidence" / "ranking_snapshot" / f"{env['run_id']}.json"
            self.assertTrue(target.exists())
            self.assertEqual(ref["path"], f"evidence/ranking_snapshot/{env['run_id']}.json")

    def test_returns_evidence_ref_shape(self) -> None:
        with TemporaryDirectory() as tmp:
            audit_dir = Path(tmp) / "audit"
            audit_dir.mkdir()
            env = self._make_envelope()
            ref = write_evidence(audit_dir, "ranking_snapshot", env)
            self.assertEqual(set(ref.keys()), {"path", "sha256", "size_bytes"})
            self.assertEqual(len(ref["sha256"]), 64)
            self.assertGreater(ref["size_bytes"], 0)
            self.assertIsInstance(ref["path"], str)

    def test_sha256_matches_raw_file_bytes(self) -> None:
        # Critical invariant: the sha256 in evidence_ref must match
        # what's actually on disk byte-for-byte — NOT a reparse.
        with TemporaryDirectory() as tmp:
            audit_dir = Path(tmp) / "audit"
            audit_dir.mkdir()
            env = self._make_envelope()
            ref = write_evidence(audit_dir, "ranking_snapshot", env)
            target = Path(tmp) / "evidence" / "ranking_snapshot" / f"{env['run_id']}.json"
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
            self.assertEqual(ref["sha256"], actual)

    def test_size_bytes_matches_file_size(self) -> None:
        with TemporaryDirectory() as tmp:
            audit_dir = Path(tmp) / "audit"
            audit_dir.mkdir()
            env = self._make_envelope(payload={"big": "x" * 2000})
            ref = write_evidence(audit_dir, "ranking_snapshot", env)
            target = Path(tmp) / "evidence" / "ranking_snapshot" / f"{env['run_id']}.json"
            self.assertEqual(ref["size_bytes"], target.stat().st_size)

    def test_duplicate_run_id_raises(self) -> None:
        # Immutability: a fresh run_id is the only way to write a new
        # evidence file for the same kind.
        with TemporaryDirectory() as tmp:
            audit_dir = Path(tmp) / "audit"
            audit_dir.mkdir()
            run_id = str(uuid.uuid4())
            env1 = self._make_envelope(run_id=run_id, payload={"x": 1})
            env2 = self._make_envelope(run_id=run_id, payload={"x": 2})
            write_evidence(audit_dir, "ranking_snapshot", env1)
            with self.assertRaises(FileExistsError):
                write_evidence(audit_dir, "ranking_snapshot", env2)

    def test_missing_run_id_raises_key_error(self) -> None:
        with TemporaryDirectory() as tmp:
            audit_dir = Path(tmp) / "audit"
            audit_dir.mkdir()
            with self.assertRaises(KeyError):
                write_evidence(audit_dir, "ranking_snapshot", {"no": "run_id"})

    def test_creates_kind_subdir_lazily(self) -> None:
        with TemporaryDirectory() as tmp:
            audit_dir = Path(tmp) / "audit"
            audit_dir.mkdir()
            self.assertFalse((Path(tmp) / "evidence").exists())
            env = self._make_envelope()
            write_evidence(audit_dir, "ranking_snapshot", env)
            self.assertTrue((Path(tmp) / "evidence" / "ranking_snapshot").is_dir())


class TestWriteEvidenceSanitization(unittest.TestCase):

    def test_pii_in_payload_is_redacted(self) -> None:
        # PII keys recognized by sanitize_audit_event MUST be redacted
        # in the evidence file. Persisted artifact == hashed artifact,
        # so the redaction also propagates into the evidence_ref.sha256.
        with TemporaryDirectory() as tmp:
            audit_dir = Path(tmp) / "audit"
            audit_dir.mkdir()
            env = build_event_envelope(
                {
                    "event_type": "rank_category",
                    "subject_token": "tok",
                    "payload": {
                        "investor": {
                            "email": "user@example.com",
                            "pan": "ABCDE1234F",
                            "full_name": "Real Name",
                            "score": 99,  # non-PII, must survive
                        },
                    },
                },
                evidence_kind="ranking_snapshot",
            )
            ref = write_evidence(audit_dir, "ranking_snapshot", env)
            target = Path(tmp) / ref["path"]
            persisted = json.loads(target.read_text(encoding="utf-8"))
            investor = persisted["payload"]["investor"]
            self.assertEqual(investor["email"], "[redacted]")
            self.assertEqual(investor["pan"], "[redacted]")
            self.assertEqual(investor["full_name"], "[redacted]")
            # Non-PII fields untouched.
            self.assertEqual(investor["score"], 99)

    def test_sanitize_before_hash_invariant(self) -> None:
        # The hash must cover the SANITIZED bytes, not the raw
        # envelope — otherwise a replay verifying the file's bytes
        # would mismatch the audit ref.
        with TemporaryDirectory() as tmp:
            audit_dir = Path(tmp) / "audit"
            audit_dir.mkdir()
            env = build_event_envelope(
                {
                    "event_type": "x",
                    "payload": {"email": "a@b.com"},
                },
                evidence_kind="ranking_snapshot",
            )
            ref = write_evidence(audit_dir, "ranking_snapshot", env)
            target = Path(tmp) / ref["path"]
            disk_bytes = target.read_bytes()
            # The bytes already on disk contain [redacted] — hashing
            # those gives ref.sha256. Hashing the raw env would NOT.
            self.assertIn(b"[redacted]", disk_bytes)
            self.assertNotIn(b"a@b.com", disk_bytes)
            self.assertEqual(hashlib.sha256(disk_bytes).hexdigest(), ref["sha256"])


class TestEmitEvidenceOrchestration(unittest.TestCase):

    def test_writes_file_first_then_audit_ref(self) -> None:
        # End-to-end: emit_evidence writes the file AND appends the
        # audit row, sharing run_id, evidence_kind, and inputs.
        with TemporaryDirectory() as tmp:
            audit_path = Path(tmp) / "audit" / "audit.jsonl"
            record = emit_evidence(
                audit_log_path=audit_path,
                evidence_kind="ranking_snapshot",
                audit_event={
                    "event_type": "rank_category",
                    "category": "equity_large_cap",
                    "ranked_count": 5,
                },
                payload={"ranked": [{"scheme_code": 1, "rank": 1}], "schema_version": "v1"},
            )
            # Audit row exists and chain verifies.
            self.assertTrue(audit_path.exists())
            self.assertTrue(verify_audit_chain(audit_path))
            # Evidence file exists at the path referenced in the audit row.
            event = record["event"]
            self.assertIn("evidence_ref", event)
            evidence_path = Path(tmp) / event["evidence_ref"]["path"]
            self.assertTrue(evidence_path.exists())

    def test_run_id_matches_filename(self) -> None:
        with TemporaryDirectory() as tmp:
            audit_path = Path(tmp) / "audit" / "audit.jsonl"
            record = emit_evidence(
                audit_log_path=audit_path,
                evidence_kind="ranking_snapshot",
                audit_event={"event_type": "rank_category"},
                payload={"x": 1},
            )
            run_id = record["event"]["run_id"]
            evidence_path = record["event"]["evidence_ref"]["path"]
            # The filename basename must be <run_id>.json
            self.assertTrue(evidence_path.endswith(f"{run_id}.json"))

    def test_evidence_file_sha256_matches_audit_ref(self) -> None:
        with TemporaryDirectory() as tmp:
            audit_path = Path(tmp) / "audit" / "audit.jsonl"
            record = emit_evidence(
                audit_log_path=audit_path,
                evidence_kind="portfolio_health_snapshot",
                audit_event={"event_type": "portfolio_health_check", "holdings_count": 3},
                payload={"health": "ok", "schema_version": "v1"},
            )
            ref = record["event"]["evidence_ref"]
            file_path = Path(tmp) / ref["path"]
            actual_sha = hashlib.sha256(file_path.read_bytes()).hexdigest()
            self.assertEqual(actual_sha, ref["sha256"])
            self.assertEqual(file_path.stat().st_size, ref["size_bytes"])

    def test_audit_record_has_no_heavy_payload(self) -> None:
        # The whole point of by-reference is that the audit log stays
        # lightweight. The heavy payload MUST NOT appear inline.
        with TemporaryDirectory() as tmp:
            audit_path = Path(tmp) / "audit" / "audit.jsonl"
            big_payload = {"ranked": [{"scheme_code": i, "rank": i} for i in range(500)]}
            record = emit_evidence(
                audit_log_path=audit_path,
                evidence_kind="ranking_snapshot",
                audit_event={"event_type": "rank_category", "ranked_count": 500},
                payload=big_payload,
            )
            # Audit event has evidence_ref but NOT the payload.
            self.assertIn("evidence_ref", record["event"])
            self.assertNotIn("payload", record["event"])
            self.assertNotIn("ranked", record["event"])

    def test_audit_envelope_matches_evidence_envelope_provenance(self) -> None:
        # run_id, evidence_kind, parent_run_id, and the inputs dict
        # must match between the two records — replay correlates them
        # via run_id and verifies inputs match.
        with TemporaryDirectory() as tmp:
            audit_path = Path(tmp) / "audit" / "audit.jsonl"
            parent = str(uuid.uuid4())
            record = emit_evidence(
                audit_log_path=audit_path,
                evidence_kind="ranking_snapshot",
                audit_event={"event_type": "rank_category"},
                payload={"x": 1},
                parent_run_id=parent,
            )
            audit_event = record["event"]
            evidence_path = Path(tmp) / audit_event["evidence_ref"]["path"]
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            self.assertEqual(audit_event["run_id"], evidence["run_id"])
            self.assertEqual(audit_event["parent_run_id"], evidence["parent_run_id"])
            self.assertEqual(audit_event["parent_run_id"], parent)
            self.assertEqual(audit_event["evidence_kind"], evidence["evidence_kind"])
            self.assertEqual(audit_event["inputs"], evidence["inputs"])

    def test_pii_redacted_in_both_evidence_and_audit(self) -> None:
        with TemporaryDirectory() as tmp:
            audit_path = Path(tmp) / "audit" / "audit.jsonl"
            record = emit_evidence(
                audit_log_path=audit_path,
                evidence_kind="ranking_snapshot",
                audit_event={
                    "event_type": "rank_category",
                    "email": "leak@example.com",  # PII at the top level
                },
                payload={"investor_pan": "ABCDE1234F", "ranked": []},
            )
            # Audit row redacts the PII top-level key.
            self.assertEqual(record["event"]["email"], "[redacted]")
            # Evidence file also redacts.
            evidence_path = Path(tmp) / record["event"]["evidence_ref"]["path"]
            persisted = json.loads(evidence_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["email"], "[redacted]")


class TestEmitEvidenceOrphanSemantics(unittest.TestCase):

    def test_audit_append_failure_leaves_orphan_file(self) -> None:
        # User's explicit ordering rule: if the audit append fails,
        # an orphan evidence file on disk is ACCEPTABLE; the reverse
        # (audit ref without file) is NEVER acceptable.
        with TemporaryDirectory() as tmp:
            audit_path = Path(tmp) / "audit" / "audit.jsonl"
            # Patch append_audit_record to raise — simulating disk
            # full / lock contention / chain-corruption.
            with patch(
                "backend.investment_analytics.audit.append_audit_record",
                side_effect=RuntimeError("simulated chain failure"),
            ):
                with self.assertRaises(RuntimeError):
                    emit_evidence(
                        audit_log_path=audit_path,
                        evidence_kind="ranking_snapshot",
                        audit_event={"event_type": "rank_category"},
                        payload={"x": 1},
                    )
            # No audit log was created.
            self.assertFalse(audit_path.exists())
            # BUT an orphan evidence file exists somewhere under
            # data/evidence/ranking_snapshot/.
            kind_dir = Path(tmp) / "evidence" / "ranking_snapshot"
            self.assertTrue(kind_dir.is_dir())
            orphans = list(kind_dir.glob("*.json"))
            self.assertEqual(len(orphans), 1)


if __name__ == "__main__":
    unittest.main()
