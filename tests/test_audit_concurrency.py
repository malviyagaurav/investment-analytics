"""Regression tests for backend.investment_analytics.audit concurrency.

Reproduces the race condition that historically corrupted the chain at
data/audit/audit.jsonl line 7572 (two records sharing 2026-04-30T08:33:51Z
with mismatched prev_hash). Without serialization, N concurrent
append_audit_record() callers will interleave reads and corrupt the chain.
"""
from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

from backend.investment_analytics.audit import (
    append_audit_record,
    verify_audit_chain,
    verify_audit_chain_diag,
)


class AuditConcurrencyTests(unittest.TestCase):
    """Many threads appending in parallel must produce a valid chain."""

    def test_chain_remains_valid_under_concurrent_appends(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "audit.jsonl"

            n_writers = 16
            writes_per_writer = 25
            total_expected = n_writers * writes_per_writer

            barrier = threading.Barrier(n_writers)

            def writer(worker_id: int) -> None:
                # Synchronize start so all threads contend on the lock.
                barrier.wait()
                for i in range(writes_per_writer):
                    append_audit_record(
                        path,
                        {
                            "event_type": "concurrency_test",
                            "worker": worker_id,
                            "seq": i,
                        },
                    )

            threads = [threading.Thread(target=writer, args=(w,)) for w in range(n_writers)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # Chain must be valid.
            self.assertTrue(verify_audit_chain(path),
                            "Concurrent appends produced a broken chain")

            # All records must be present.
            with path.open("r", encoding="utf-8") as fh:
                lines = [ln for ln in fh.read().splitlines() if ln.strip()]
            self.assertEqual(len(lines), total_expected,
                             f"Expected {total_expected} records, got {len(lines)}")

    def test_diagnostic_returns_clean_for_valid_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "audit.jsonl"
            for i in range(5):
                append_audit_record(path, {"event_type": "smoke", "seq": i})
            diag = verify_audit_chain_diag(path)
            self.assertTrue(diag["valid"])
            self.assertEqual(diag["lines_scanned"], 5)
            self.assertIsNone(diag["first_bad_line"])
            self.assertIsNone(diag["reason"])

    def test_diagnostic_pinpoints_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "audit.jsonl"
            for i in range(3):
                append_audit_record(path, {"event_type": "smoke", "seq": i})
            # Inject a corrupt record (wrong prev_hash) on a 4th line.
            with path.open("a", encoding="utf-8") as fh:
                fh.write(
                    '{"timestamp":"2026-04-30T00:00:00+00:00","schema_version":"v1",'
                    '"analyzer_version":"mf_v2","prev_hash":"deadbeef",'
                    '"payload_hash":"x","event":{"event_type":"forged"},'
                    '"current_hash":"y"}\n'
                )
            diag = verify_audit_chain_diag(path)
            self.assertFalse(diag["valid"])
            self.assertEqual(diag["first_bad_line"], 4)
            self.assertEqual(diag["reason"], "prev_hash_mismatch")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
