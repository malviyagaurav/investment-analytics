"""Cross-process integrity tests for ``audit.append_audit_record``.

Complements ``tests/test_audit_concurrency.py`` (within-process
threading) by exercising the fcntl.flock layer with independent OS
processes. Required because step 5 of the evidence-layer roadmap
puts cron-driven watchlist appends concurrent with FastAPI appends;
without process-safe locking, that combination silently corrupts
the hash chain.

Three cases:
  1. Happy path — 4 processes × 25 records, all chain links valid.
  2. Same as (1), but one randomly-chosen worker raises an injected
     exception inside the lock (after flock acquire, before write).
     Verifies that the ``finally`` release path runs, no partial
     line is written, and subsequent appends still chain validly.
  3. Single-process targeted failure — patches ``_last_record_hash``
     to raise mid-lock and asserts the file ends up with exactly
     the expected (clean) line count plus an unbroken chain.

The happy-path test repeats 3× to surface OS-scheduling flake.
"""
from __future__ import annotations

import multiprocessing
import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from backend.investment_analytics.audit import (
    append_audit_record,
    verify_audit_chain,
    verify_audit_chain_diag,
)


# Workers must be top-level so multiprocessing.Pool (spawn mode on
# macOS) can pickle and re-import them in child processes.


def _worker_append(args):
    """Append ``n_records`` audit records to ``path``. Returns the
    list of current_hash values so the parent can assert uniqueness."""
    path_str, worker_id, n_records = args
    path = Path(path_str)
    hashes = []
    for i in range(n_records):
        record = append_audit_record(
            path,
            {
                "event_type": "test_event",
                "worker_id": worker_id,
                "record_index": i,
            },
        )
        hashes.append(record["current_hash"])
    return {"worker_id": worker_id, "hashes": hashes, "failures": 0}


def _worker_append_maybe_failing(args):
    """Same as ``_worker_append`` but, if ``fail_at_record >= 0``,
    monkey-patches the audit module in this child process so the
    ``fail_at_record``-th call to ``append_audit_record`` raises
    AFTER acquiring fcntl.flock but BEFORE writing the line.

    The patch targets ``_last_record_hash`` because that function is
    called inside the critical section (after flock acquire) but
    before any write touches the file. A raise here exercises the
    finally-release path and proves no partial line is left behind.
    """
    path_str, worker_id, n_records, fail_at_record = args
    path = Path(path_str)
    hashes = []
    failures = 0

    from backend.investment_analytics import audit as _audit

    if fail_at_record >= 0:
        original = _audit._last_record_hash
        call_state = {"count": 0}

        def maybe_failing(p):
            call_state["count"] += 1
            if call_state["count"] == fail_at_record + 1:
                raise RuntimeError(
                    f"injected mid-lock failure in worker {worker_id}"
                )
            return original(p)

        _audit._last_record_hash = maybe_failing

    for i in range(n_records):
        try:
            record = append_audit_record(
                path,
                {
                    "event_type": "test_event",
                    "worker_id": worker_id,
                    "record_index": i,
                },
            )
            hashes.append(record["current_hash"])
        except RuntimeError:
            failures += 1

    return {"worker_id": worker_id, "hashes": hashes, "failures": failures}


class TestMultiProcessAppendIntegrity(unittest.TestCase):
    """Happy-path cross-process append integrity."""

    N_WORKERS = 4
    PER_WORKER = 25
    REPEATS = 3  # surface OS-scheduling flake

    def _run_once(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "audit.jsonl"
            args = [
                (str(path), wid, self.PER_WORKER)
                for wid in range(self.N_WORKERS)
            ]
            with multiprocessing.Pool(self.N_WORKERS) as pool:
                results = pool.map(_worker_append, args)

            expected = self.N_WORKERS * self.PER_WORKER
            diag = verify_audit_chain_diag(path)
            self.assertTrue(diag["valid"], f"Chain invalid: {diag}")
            self.assertEqual(diag["lines_scanned"], expected,
                             f"Expected {expected} records, got {diag['lines_scanned']}")

            all_hashes = [h for r in results for h in r["hashes"]]
            self.assertEqual(len(all_hashes), expected)
            self.assertEqual(
                len(set(all_hashes)), expected,
                "Duplicate current_hash — two workers wrote the same content "
                "under the same prev_hash (lock failed)",
            )

    def test_repeated_happy_path(self) -> None:
        for run in range(self.REPEATS):
            with self.subTest(run=run):
                self._run_once()


class TestMultiProcessFailurePathIntegrity(unittest.TestCase):
    """One worker injects an exception inside the lock; chain must
    remain valid and no partial line may be written."""

    N_WORKERS = 4
    PER_WORKER = 25

    def test_random_mid_lock_failure_preserves_chain(self) -> None:
        # Seed for deterministic CI behaviour. The chosen worker /
        # record varies across seeds but the assertion is the same:
        # exactly one failure, n-1 records persisted, chain valid.
        random.seed(2026)
        failing_worker = random.randint(0, self.N_WORKERS - 1)
        failing_record = random.randint(0, self.PER_WORKER - 1)

        with TemporaryDirectory() as td:
            path = Path(td) / "audit.jsonl"
            args = [
                (
                    str(path),
                    wid,
                    self.PER_WORKER,
                    failing_record if wid == failing_worker else -1,
                )
                for wid in range(self.N_WORKERS)
            ]
            with multiprocessing.Pool(self.N_WORKERS) as pool:
                results = pool.map(_worker_append_maybe_failing, args)

            total_failures = sum(r["failures"] for r in results)
            self.assertEqual(
                total_failures, 1,
                f"Expected exactly 1 injected failure, got {total_failures}",
            )

            expected_lines = self.N_WORKERS * self.PER_WORKER - 1
            diag = verify_audit_chain_diag(path)
            self.assertTrue(
                diag["valid"],
                f"Chain invalid after injected failure: {diag}",
            )
            self.assertEqual(
                diag["lines_scanned"], expected_lines,
                f"Expected {expected_lines} surviving records, got {diag['lines_scanned']} "
                "(may indicate a partial line was written under the failure)",
            )

            all_hashes = [h for r in results for h in r["hashes"]]
            self.assertEqual(len(all_hashes), expected_lines)
            self.assertEqual(
                len(set(all_hashes)), expected_lines,
                "Duplicate current_hash after failure-path test",
            )


class TestSingleProcessFailurePathIntegrity(unittest.TestCase):
    """Targeted single-process test for the finally-release invariant.

    Multi-process tests verify end-to-end correctness; this test
    isolates the failure semantics so a regression in finally-release
    surfaces a clean, single-process diagnostic rather than a flaky
    pool result.
    """

    def test_exception_inside_lock_releases_and_no_partial_line(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "audit.jsonl"
            # Baseline record — chain starts clean.
            r1 = append_audit_record(path, {"event_type": "ok_1"})
            self.assertTrue(verify_audit_chain(path))

            # Inject a raise inside the lock, after flock acquire,
            # before any write happens. _last_record_hash runs in
            # exactly that window.
            with patch(
                "backend.investment_analytics.audit._last_record_hash",
                side_effect=RuntimeError("injected mid-lock failure"),
            ):
                with self.assertRaises(RuntimeError):
                    append_audit_record(path, {"event_type": "should_fail"})

            # Chain remains valid (no partial line, no broken hash link).
            self.assertTrue(
                verify_audit_chain(path),
                "Chain corrupted by failed mid-lock append",
            )

            # Lock was released — next append must succeed and chain
            # onto r1's current_hash directly (the failed append left
            # no trace).
            r2 = append_audit_record(path, {"event_type": "ok_2"})
            self.assertEqual(r2["prev_hash"], r1["current_hash"])
            self.assertTrue(verify_audit_chain(path))

            # Exactly 2 records on disk — the failed attempt left no line.
            with path.open(encoding="utf-8") as handle:
                lines = [line for line in handle if line.strip()]
            self.assertEqual(
                len(lines), 2,
                "Failed append left a partial or stray line on disk",
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
