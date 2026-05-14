"""Tests for chain_epoch field, epochs.json migration, multi-epoch
verifier, and orphan classification.

Step 2 of the evidence-layer roadmap. Covers:
  - chain_epoch field is written on every new record (defaults to 1
    before/after migration if only epoch 1 is open).
  - Old records (no chain_epoch field) still verify cleanly.
  - migrate_audit_to_epochs creates epochs.json and is idempotent.
  - Migration refuses to register a broken chain as epoch 1.
  - Orphan classification: clean independent / partially corrupt /
    chained predecessor / genesis vs imported_legacy root types.
  - verify_audit_chain_multi returns typed overall_status
    ("valid" / "partial_failure" / "invalid" / "empty" /
    "unverifiable").
  - schema_fingerprint is deterministic and present.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from backend.investment_analytics.audit import (
    _active_epoch,
    _canonical_json,
    _sha256,
    append_audit_record,
    verify_audit_chain,
    verify_audit_chain_diag,
    verify_audit_chain_multi,
)
from backend.investment_analytics.audit_migrate import (
    EPOCHS_SCHEMA_DEFINITION,
    EPOCHS_SCHEMA_FINGERPRINT,
    classify_orphan,
    migrate_audit_to_epochs,
)


def _seed_clean_chain(audit_dir: Path, n_records: int = 5) -> Path:
    """Create a fresh audit.jsonl with n valid records under the
    current schema. Returns the audit log path."""
    audit_dir.mkdir(parents=True, exist_ok=True)
    path = audit_dir / "audit.jsonl"
    for i in range(n_records):
        append_audit_record(path, {"event_type": "test", "i": i})
    return path


class TestChainEpochField(unittest.TestCase):

    def test_new_records_carry_chain_epoch(self) -> None:
        with TemporaryDirectory() as td:
            audit_dir = Path(td)
            path = audit_dir / "audit.jsonl"
            record = append_audit_record(path, {"event_type": "x"})
            self.assertIn("chain_epoch", record)
            self.assertEqual(record["chain_epoch"], 1)

    def test_chain_epoch_defaults_to_1_without_epochs_json(self) -> None:
        with TemporaryDirectory() as td:
            audit_dir = Path(td)
            self.assertEqual(_active_epoch(audit_dir), 1)

    def test_chain_epoch_reads_open_epoch_from_index(self) -> None:
        with TemporaryDirectory() as td:
            audit_dir = Path(td)
            index = {
                "schema_version": "v1",
                "schema_fingerprint": "x" * 64,
                "epochs": [
                    {"epoch": 1, "file": "audit.jsonl.epoch-1", "status": "closed",
                     "chain_root_type": "genesis"},
                    {"epoch": 2, "file": "audit.jsonl", "status": "open",
                     "chain_root_type": "rotated_handoff"},
                ],
                "orphan_chains": [],
                "registered_at": "2026-05-15T00:00:00+00:00",
            }
            (audit_dir / "epochs.json").write_text(json.dumps(index))
            self.assertEqual(_active_epoch(audit_dir), 2)

    def test_chain_epoch_falls_back_to_1_when_index_unreadable(self) -> None:
        with TemporaryDirectory() as td:
            audit_dir = Path(td)
            (audit_dir / "epochs.json").write_text("{ not valid json")
            self.assertEqual(_active_epoch(audit_dir), 1)


class TestMixedSchemaChainStillVerifies(unittest.TestCase):
    """Existing records may lack chain_epoch (pre-migration); new
    records carry the field. Chain links must still verify because
    the hash is over each record's actual content."""

    def test_pre_existing_records_without_chain_epoch_chain_to_new_ones(self) -> None:
        with TemporaryDirectory() as td:
            audit_dir = Path(td)
            path = audit_dir / "audit.jsonl"

            # Manually craft a record WITHOUT chain_epoch (simulating
            # pre-migration data).
            old_record = {
                "timestamp": "2026-04-30T08:00:00+00:00",
                "schema_version": "v1",
                "analyzer_version": "mf_v2",
                "prev_hash": "0" * 64,
                "payload_hash": _sha256(_canonical_json({"e": "old"})),
                "event": {"e": "old"},
            }
            old_record["current_hash"] = _sha256(_canonical_json(old_record))
            path.write_text(_canonical_json(old_record) + "\n", encoding="utf-8")

            # Append a new record (this WILL have chain_epoch).
            new_record = append_audit_record(path, {"event_type": "new"})
            self.assertEqual(new_record["prev_hash"], old_record["current_hash"])
            self.assertIn("chain_epoch", new_record)

            # Full chain still verifies.
            diag = verify_audit_chain_diag(path)
            self.assertTrue(diag["valid"], f"Mixed-schema chain failed: {diag}")
            self.assertEqual(diag["lines_scanned"], 2)


class TestSchemaFingerprint(unittest.TestCase):

    def test_fingerprint_is_64_char_hex(self) -> None:
        self.assertEqual(len(EPOCHS_SCHEMA_FINGERPRINT), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in EPOCHS_SCHEMA_FINGERPRINT))

    def test_fingerprint_is_deterministic(self) -> None:
        # Same schema definition → same fingerprint, every time.
        again = _sha256(_canonical_json(EPOCHS_SCHEMA_DEFINITION))
        self.assertEqual(again, EPOCHS_SCHEMA_FINGERPRINT)


class TestMigrationIdempotency(unittest.TestCase):

    def test_migrate_clean_chain_creates_epochs_json(self) -> None:
        with TemporaryDirectory() as td:
            audit_dir = Path(td)
            _seed_clean_chain(audit_dir, n_records=3)

            index = migrate_audit_to_epochs(audit_dir)

            self.assertEqual(index["__migration_status"], "migrated")
            self.assertTrue((audit_dir / "epochs.json").exists())
            self.assertEqual(len(index["epochs"]), 1)

            epoch1 = index["epochs"][0]
            self.assertEqual(epoch1["epoch"], 1)
            self.assertEqual(epoch1["file"], "audit.jsonl")
            self.assertEqual(epoch1["status"], "open")
            self.assertIsNone(epoch1["closed_at"])
            self.assertEqual(epoch1["chain_root_type"], "genesis")
            self.assertEqual(len(epoch1["first_record_hash"]), 64)

            self.assertEqual(index["schema_fingerprint"], EPOCHS_SCHEMA_FINGERPRINT)

    def test_migrate_is_idempotent(self) -> None:
        with TemporaryDirectory() as td:
            audit_dir = Path(td)
            _seed_clean_chain(audit_dir, n_records=3)

            index1 = migrate_audit_to_epochs(audit_dir)
            index2 = migrate_audit_to_epochs(audit_dir)

            self.assertEqual(index1["__migration_status"], "migrated")
            self.assertEqual(index2["__migration_status"], "already_migrated")

            # File content unchanged between runs.
            written = json.loads((audit_dir / "epochs.json").read_text(encoding="utf-8"))
            self.assertEqual(written["schema_fingerprint"], EPOCHS_SCHEMA_FINGERPRINT)

    def test_migrate_dry_run_does_not_write(self) -> None:
        with TemporaryDirectory() as td:
            audit_dir = Path(td)
            _seed_clean_chain(audit_dir, n_records=2)

            index = migrate_audit_to_epochs(audit_dir, dry_run=True)

            self.assertEqual(index["__migration_status"], "dry_run")
            self.assertFalse((audit_dir / "epochs.json").exists())

    def test_migrate_refuses_broken_chain(self) -> None:
        with TemporaryDirectory() as td:
            audit_dir = Path(td)
            path = _seed_clean_chain(audit_dir, n_records=3)

            # Corrupt the middle record.
            lines = path.read_text(encoding="utf-8").splitlines()
            tampered = json.loads(lines[1])
            tampered["payload_hash"] = "0" * 64  # break the body hash
            lines[1] = _canonical_json(tampered)
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            with self.assertRaises(RuntimeError):
                migrate_audit_to_epochs(audit_dir)
            self.assertFalse((audit_dir / "epochs.json").exists())


class TestOrphanClassification(unittest.TestCase):

    def test_classify_valid_independent_legacy(self) -> None:
        with TemporaryDirectory() as td:
            audit_dir = Path(td)
            # Build a stand-alone valid chain in a separate file.
            orphan_path = audit_dir / "audit.jsonl.legacy.bak"
            for i in range(3):
                append_audit_record(orphan_path, {"event_type": "old", "i": i})
            # Live chain starts at genesis (independent of orphan).
            result = classify_orphan(orphan_path, live_first_prev_hash="0" * 64)
            self.assertEqual(result["classification"], "valid_independent_legacy")
            self.assertEqual(result["chain_root_type"], "genesis")
            self.assertEqual(result["total_lines"], 3)
            self.assertEqual(result["valid_through_line"], 3)
            self.assertIsNone(result["first_failure_reason"])

    def test_classify_partially_corrupt_independent_legacy(self) -> None:
        with TemporaryDirectory() as td:
            audit_dir = Path(td)
            orphan_path = audit_dir / "audit.jsonl.corrupt.bak"
            for i in range(4):
                append_audit_record(orphan_path, {"event_type": "old", "i": i})

            # Corrupt line 3 (1-indexed): break prev_hash linkage.
            lines = orphan_path.read_text(encoding="utf-8").splitlines()
            tampered = json.loads(lines[2])
            tampered["prev_hash"] = "0" * 64
            lines[2] = _canonical_json(tampered)
            orphan_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            result = classify_orphan(orphan_path, live_first_prev_hash="0" * 64)
            self.assertEqual(
                result["classification"],
                "partially_corrupt_independent_legacy",
            )
            self.assertEqual(result["valid_through_line"], 2)
            self.assertIn("at line 3", result["first_failure_reason"])

    def test_classify_chained_predecessor_detected(self) -> None:
        with TemporaryDirectory() as td:
            audit_dir = Path(td)
            orphan_path = audit_dir / "audit.jsonl.prior.bak"
            last_record = None
            for i in range(3):
                last_record = append_audit_record(orphan_path, {"event_type": "old", "i": i})

            # Live chain's first prev_hash matches orphan's tail.
            live_head_prev = last_record["current_hash"]
            result = classify_orphan(orphan_path, live_first_prev_hash=live_head_prev)
            self.assertEqual(result["classification"], "chained_predecessor")


class TestVerifierMultiEpoch(unittest.TestCase):

    def test_empty_dir_returns_empty(self) -> None:
        with TemporaryDirectory() as td:
            result = verify_audit_chain_multi(Path(td))
            self.assertEqual(result["overall_status"], "empty")
            self.assertEqual(result["epochs_total"], 0)
            self.assertEqual(result["epochs_valid"], 0)

    def test_falls_back_to_single_file_without_epochs_json(self) -> None:
        with TemporaryDirectory() as td:
            audit_dir = Path(td)
            _seed_clean_chain(audit_dir, n_records=3)

            result = verify_audit_chain_multi(audit_dir)
            self.assertEqual(result["overall_status"], "valid")
            self.assertEqual(result["epochs_total"], 1)
            self.assertEqual(result["epochs_valid"], 1)
            self.assertEqual(result["epochs_failed"], 0)
            self.assertEqual(len(result["per_epoch"]), 1)
            self.assertEqual(result["per_epoch"][0]["status"], "valid")

    def test_walks_epochs_json_when_present(self) -> None:
        with TemporaryDirectory() as td:
            audit_dir = Path(td)
            _seed_clean_chain(audit_dir, n_records=3)
            migrate_audit_to_epochs(audit_dir)

            result = verify_audit_chain_multi(audit_dir)
            self.assertEqual(result["overall_status"], "valid")
            self.assertEqual(result["epochs_total"], 1)
            self.assertEqual(result["per_epoch"][0]["lines_scanned"], 3)

    def test_invalid_epochs_json_returns_unverifiable(self) -> None:
        with TemporaryDirectory() as td:
            audit_dir = Path(td)
            _seed_clean_chain(audit_dir, n_records=2)
            (audit_dir / "epochs.json").write_text("{ broken")

            result = verify_audit_chain_multi(audit_dir)
            self.assertEqual(result["overall_status"], "unverifiable")
            self.assertIn("reason", result)

    def test_partial_failure_with_one_valid_one_invalid_epoch(self) -> None:
        with TemporaryDirectory() as td:
            audit_dir = Path(td)
            # Two stand-alone chains in two files.
            path_a = audit_dir / "epoch_1.jsonl"
            path_b = audit_dir / "audit.jsonl"
            for i in range(2):
                append_audit_record(path_a, {"i": i})
            for i in range(2):
                append_audit_record(path_b, {"i": i})

            # Corrupt path_b to fail verification.
            lines = path_b.read_text(encoding="utf-8").splitlines()
            tampered = json.loads(lines[1])
            tampered["payload_hash"] = "f" * 64
            lines[1] = _canonical_json(tampered)
            path_b.write_text("\n".join(lines) + "\n", encoding="utf-8")

            # Manually craft an epochs.json registering both files.
            index = {
                "schema_version": "v1",
                "schema_fingerprint": EPOCHS_SCHEMA_FINGERPRINT,
                "epochs": [
                    {"epoch": 1, "file": "epoch_1.jsonl", "status": "closed",
                     "chain_root_type": "genesis"},
                    {"epoch": 2, "file": "audit.jsonl", "status": "open",
                     "chain_root_type": "rotated_handoff"},
                ],
                "orphan_chains": [],
                "registered_at": "2026-05-15T00:00:00+00:00",
            }
            (audit_dir / "epochs.json").write_text(json.dumps(index))

            result = verify_audit_chain_multi(audit_dir)
            self.assertEqual(result["overall_status"], "partial_failure")
            self.assertEqual(result["epochs_total"], 2)
            self.assertEqual(result["epochs_valid"], 1)
            self.assertEqual(result["epochs_failed"], 1)


class TestExistingApiUnchanged(unittest.TestCase):
    """The existing single-file callers (api/main.py /health,
    legacy tests) must keep working exactly as before."""

    def test_verify_audit_chain_returns_bool(self) -> None:
        with TemporaryDirectory() as td:
            path = _seed_clean_chain(Path(td), n_records=2)
            self.assertTrue(verify_audit_chain(path))

    def test_verify_audit_chain_diag_single_file_shape(self) -> None:
        with TemporaryDirectory() as td:
            path = _seed_clean_chain(Path(td), n_records=2)
            diag = verify_audit_chain_diag(path)
            # Existing shape — no new fields added to the single-file diag.
            self.assertIn("valid", diag)
            self.assertIn("lines_scanned", diag)
            self.assertIn("first_bad_line", diag)
            self.assertIn("reason", diag)
            self.assertTrue(diag["valid"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
