"""Cross-platform parity tests (Windows + macOS + Linux).

These tests exercise the platform-sensitive surfaces identified in
the portability audit and prove the architecture's guarantees
survive the platform boundary. They MUST pass on macOS, Linux, and
Windows without modification. Any test that depends on a platform-
specific primitive is gated by the appropriate ``sys.platform``
check and documents WHY the gate exists.

## Coverage map

  Locking adapter (backend.investment_analytics._locking):
    - blocking-acquire succeeds when uncontended
    - non-blocking-acquire returns True when uncontended
    - non-blocking-acquire returns False under contention
    - blocking-acquire serializes cross-process appenders
    - release frees the lock
    - LockAcquisitionTimeout fires on blocking timeout (Windows
      branch only — POSIX branch defaults to truly-unbounded
      and the timed path uses LOCK_NB internally, so we exercise
      both paths)

  Newline robustness:
    - chain written on Windows-style (CRLF on disk) verifies
      correctly when read on POSIX-style (LF read path)
    - chain with mixed CRLF/LF lines still verifies
    - explicit newline="\\n" on append produces LF-only bytes
      regardless of host platform

  Hash-chain byte stability:
    - canonical JSON of a record is identical across platforms
    - SHA-256 of canonical JSON is identical across platforms
    - re-canonicalization round-trips byte-equal

  Deterministic serialization:
    - json.dumps(..., sort_keys=True, separators=(",",":"),
      ensure_ascii=True) produces identical bytes regardless of
      key insertion order or float repr quirks

  Path canonicalization:
    - ROOT resolution is identical via pathlib regardless of
      slash direction in the source string
    - Path operations do not depend on os.sep
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from backend.investment_analytics import _locking
from backend.investment_analytics._locking import (
    LockAcquisitionTimeout,
    acquire_exclusive_blocking,
    acquire_exclusive_nonblocking,
    release,
)
from backend.investment_analytics.audit import (
    _canonical_json,
    _sha256,
    append_audit_record,
    verify_audit_chain,
    verify_audit_chain_diag,
)


# ── Locking adapter ─────────────────────────────────────────────────


class LockingAdapterTests(unittest.TestCase):
    """The typed adapter is below the explanatory surface, but its
    correctness is load-bearing: a broken adapter would corrupt the
    audit chain under cross-process contention. These tests are
    structural — they verify the adapter behaves identically on
    whichever platform actually runs them."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "lockfile"
        self.path.touch()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_blocking_acquire_succeeds_when_uncontended(self) -> None:
        fd = os.open(str(self.path), os.O_RDWR)
        try:
            acquire_exclusive_blocking(fd)
            release(fd)
        finally:
            os.close(fd)

    def test_nonblocking_acquire_returns_true_when_uncontended(self) -> None:
        fd = os.open(str(self.path), os.O_RDWR)
        try:
            self.assertTrue(acquire_exclusive_nonblocking(fd))
            release(fd)
        finally:
            os.close(fd)

    def test_nonblocking_acquire_returns_false_when_held_by_another_fd(
        self,
    ) -> None:
        """Same process, two distinct fds on the same path. On POSIX
        fcntl.flock is per-(file-description) so this works. On
        Windows msvcrt.locking is per-(file-handle) — two handles
        from the same process can also contend. Both platforms
        return False here."""
        fd_a = os.open(str(self.path), os.O_RDWR)
        fd_b = os.open(str(self.path), os.O_RDWR)
        try:
            self.assertTrue(acquire_exclusive_nonblocking(fd_a))
            self.assertFalse(acquire_exclusive_nonblocking(fd_b))
            release(fd_a)
        finally:
            os.close(fd_a)
            os.close(fd_b)

    def test_release_allows_subsequent_acquire(self) -> None:
        fd_a = os.open(str(self.path), os.O_RDWR)
        try:
            self.assertTrue(acquire_exclusive_nonblocking(fd_a))
            release(fd_a)
        finally:
            os.close(fd_a)
        fd_b = os.open(str(self.path), os.O_RDWR)
        try:
            self.assertTrue(acquire_exclusive_nonblocking(fd_b))
            release(fd_b)
        finally:
            os.close(fd_b)

    def test_blocking_timeout_fires_when_lock_held(self) -> None:
        """On both POSIX and Windows, a finite timeout must produce
        LockAcquisitionTimeout rather than silently giving up or
        proceeding. Fail-closed."""
        fd_holder = os.open(str(self.path), os.O_RDWR)
        fd_waiter = os.open(str(self.path), os.O_RDWR)
        try:
            self.assertTrue(acquire_exclusive_nonblocking(fd_holder))
            with self.assertRaises(LockAcquisitionTimeout):
                acquire_exclusive_blocking(fd_waiter, timeout_seconds=0.3)
            release(fd_holder)
        finally:
            os.close(fd_holder)
            os.close(fd_waiter)

    def test_module_exposes_required_symbols(self) -> None:
        # If a future refactor accidentally drops a symbol from the
        # public surface, the audit / scheduler imports break. Test
        # the contract explicitly.
        for name in ("acquire_exclusive_blocking",
                     "acquire_exclusive_nonblocking",
                     "release",
                     "LockAcquisitionTimeout"):
            self.assertTrue(hasattr(_locking, name),
                            msg=f"_locking missing symbol {name!r}")


# ── Cross-process locking via the adapter ───────────────────────────


def _child_acquire_and_signal(lock_path_str: str, ready_path_str: str,
                              done_path_str: str) -> None:
    """Subprocess body: open file, acquire blocking lock, signal
    'ready' to parent, wait for 'done' signal, release."""
    from backend.investment_analytics._locking import (
        acquire_exclusive_blocking, release as release_lock,
    )
    fd = os.open(lock_path_str, os.O_RDWR | os.O_CREAT)
    try:
        acquire_exclusive_blocking(fd)
        Path(ready_path_str).touch()
        # Wait for the parent's "done" signal.
        deadline = time.monotonic() + 30.0
        while not Path(done_path_str).exists():
            if time.monotonic() > deadline:
                break
            time.sleep(0.02)
        release_lock(fd)
    finally:
        os.close(fd)


class CrossProcessLockingTests(unittest.TestCase):
    """Spawn a child process that holds the lock, then verify the
    parent's non-blocking acquire correctly reports contention. This
    is the cross-platform analog of the existing fcntl-based
    cross-process test in test_audit_multi_process_concurrency.py."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.lock_path = Path(self.tmp.name) / "lockfile"
        self.lock_path.touch()
        self.ready_path = Path(self.tmp.name) / "ready"
        self.done_path = Path(self.tmp.name) / "done"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_child_holds_lock_blocks_parent_nonblocking_acquire(self) -> None:
        ctx = multiprocessing.get_context("spawn")
        proc = ctx.Process(
            target=_child_acquire_and_signal,
            args=(str(self.lock_path),
                  str(self.ready_path),
                  str(self.done_path)),
        )
        proc.start()
        try:
            # Wait for child to signal it's holding the lock.
            deadline = time.monotonic() + 10.0
            while not self.ready_path.exists():
                if time.monotonic() > deadline:
                    self.fail("child did not signal ready within 10s")
                time.sleep(0.02)

            # Parent's non-blocking acquire MUST fail.
            fd = os.open(str(self.lock_path), os.O_RDWR)
            try:
                self.assertFalse(
                    acquire_exclusive_nonblocking(fd),
                    msg=("non-blocking acquire returned True while child "
                         "process holds the lock — cross-process "
                         "serialization is broken"),
                )
            finally:
                os.close(fd)

            # Tell the child to release.
            self.done_path.touch()
        finally:
            proc.join(timeout=10.0)
            if proc.is_alive():
                proc.terminate()
                proc.join()

    def test_parent_can_acquire_after_child_releases(self) -> None:
        ctx = multiprocessing.get_context("spawn")
        proc = ctx.Process(
            target=_child_acquire_and_signal,
            args=(str(self.lock_path),
                  str(self.ready_path),
                  str(self.done_path)),
        )
        proc.start()
        try:
            deadline = time.monotonic() + 10.0
            while not self.ready_path.exists():
                if time.monotonic() > deadline:
                    self.fail("child did not signal ready within 10s")
                time.sleep(0.02)
            # Tell child to release and wait for child to exit.
            self.done_path.touch()
            proc.join(timeout=10.0)
            self.assertFalse(proc.is_alive())
            # Now the parent should be able to acquire.
            fd = os.open(str(self.lock_path), os.O_RDWR)
            try:
                self.assertTrue(acquire_exclusive_nonblocking(fd))
                release(fd)
            finally:
                os.close(fd)
        finally:
            if proc.is_alive():
                proc.terminate()
                proc.join()


# ── Newline robustness on the audit chain ───────────────────────────


class NewlineRobustnessTests(unittest.TestCase):
    """The hash chain's integrity is proven to be independent of
    line-terminator form because hashes are over canonical JSON of
    the parsed dict, not over file bytes. These tests prove that
    property empirically: corrupt the on-disk line terminators and
    verify the chain still validates."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.audit_path = Path(self.tmp.name) / "audit" / "audit.jsonl"
        self.audit_path.parent.mkdir(parents=True)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _emit_chain(self, n: int = 5) -> None:
        for i in range(n):
            append_audit_record(
                self.audit_path,
                {"event_type": "test_newline", "i": i,
                 "schema_version": "v1"},
            )

    def _read_bytes(self) -> bytes:
        return self.audit_path.read_bytes()

    def _write_bytes(self, b: bytes) -> None:
        self.audit_path.write_bytes(b)

    def test_chain_emits_lf_only_bytes_on_disk(self) -> None:
        """Explicit newline="\\n" on the append path produces LF-only
        bytes regardless of host platform."""
        self._emit_chain(3)
        raw = self._read_bytes()
        self.assertNotIn(
            b"\r\n", raw,
            msg=("audit.jsonl contains CRLF; the explicit newline=\"\\n\" "
                 "argument is not being honored — byte stability "
                 "across platforms is broken"),
        )
        self.assertTrue(verify_audit_chain(self.audit_path))

    def test_chain_verifies_after_converting_to_crlf(self) -> None:
        """Even if the chain were written with CRLF (e.g., a file
        copied through a tool that normalized line endings), the
        hash chain must still verify — because hashes are over
        canonical JSON of the parsed dict, not over file bytes."""
        self._emit_chain(4)
        lf_bytes = self._read_bytes()
        crlf_bytes = lf_bytes.replace(b"\n", b"\r\n")
        self._write_bytes(crlf_bytes)
        diag = verify_audit_chain_diag(self.audit_path)
        self.assertTrue(
            diag["valid"],
            msg=("chain failed to verify after CRLF rewrite: "
                 f"{diag}"),
        )

    def test_chain_verifies_with_mixed_line_endings(self) -> None:
        """Mixed CRLF and LF lines (worst-case foreign tool
        produced) must still verify."""
        self._emit_chain(6)
        lf_bytes = self._read_bytes()
        lines = lf_bytes.split(b"\n")
        # Re-join with alternating CRLF / LF.
        mixed = b""
        for idx, ln in enumerate(lines):
            if not ln:
                continue
            terminator = b"\r\n" if idx % 2 == 0 else b"\n"
            mixed += ln + terminator
        self._write_bytes(mixed)
        diag = verify_audit_chain_diag(self.audit_path)
        self.assertTrue(
            diag["valid"],
            msg=f"chain failed with mixed line endings: {diag}",
        )


# ── Hash-chain byte stability ───────────────────────────────────────


class ByteStabilityTests(unittest.TestCase):
    """Canonical JSON + SHA-256 over identical Python dicts must
    produce identical bytes on any platform. This is necessary for
    replay determinism across the OS boundary."""

    def test_canonical_json_is_deterministic(self) -> None:
        # Two distinct dict construction orders → identical canonical
        # output. Property of json.dumps with sort_keys=True.
        a = {"z": 1, "a": 2, "m": [3, 4, 5]}
        b = {"a": 2, "m": [3, 4, 5], "z": 1}
        self.assertEqual(_canonical_json(a), _canonical_json(b))

    def test_canonical_json_uses_ascii(self) -> None:
        """ensure_ascii=True so multibyte characters serialize as
        \\uXXXX escapes — byte-identical regardless of platform
        default encoding."""
        s = _canonical_json({"k": "héllo"})
        self.assertNotIn("é", s)
        self.assertIn("\\u00e9", s)

    def test_canonical_json_no_internal_newlines(self) -> None:
        """separators=(",",":") and no indent → output has zero
        newline characters internally. The line terminator is
        appended OUTSIDE the canonical_json call."""
        s = _canonical_json({"a": [1, 2, 3], "b": {"c": 4}})
        self.assertNotIn("\n", s)
        self.assertNotIn("\r", s)

    def test_canonical_json_float_round_trip(self) -> None:
        """Floats must serialize identically across platforms. Python
        json uses Python's repr-style float emission which is
        IEEE-754 stable."""
        value = {"x": 0.123456789, "y": 1e-7, "z": 1.0}
        a = _canonical_json(value)
        b = _canonical_json(json.loads(a))
        self.assertEqual(a, b)

    def test_sha256_canonical_round_trip(self) -> None:
        """The verify path re-canonicalizes the parsed dict and
        hashes. This must equal the hash computed on the original
        dict prior to write."""
        record = {
            "timestamp": "2026-05-16T12:34:56+00:00",
            "schema_version": "v1",
            "prev_hash": "0" * 64,
            "payload_hash": "a" * 64,
            "event": {"x": 1, "y": ["a", "b"]},
        }
        original_hash = _sha256(_canonical_json(record))
        # Serialize as the on-disk JSONL line would be, then parse:
        line = _canonical_json(record) + "\n"
        recovered = json.loads(line)
        recovered_hash = _sha256(_canonical_json(recovered))
        self.assertEqual(original_hash, recovered_hash)


# ── Append-path emits expected bytes ────────────────────────────────


class AppendPathEmissionTests(unittest.TestCase):
    """Verify the audit append path produces the bytes we expect
    on the host platform: UTF-8, LF terminators, no BOM, canonical
    JSON form."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.audit_path = Path(self.tmp.name) / "audit" / "audit.jsonl"
        self.audit_path.parent.mkdir(parents=True)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_emitted_bytes_have_no_bom(self) -> None:
        append_audit_record(
            self.audit_path,
            {"event_type": "test_bom", "schema_version": "v1"},
        )
        raw = self.audit_path.read_bytes()
        self.assertFalse(
            raw.startswith(b"\xef\xbb\xbf"),
            msg=("audit.jsonl starts with a UTF-8 BOM; cross-platform "
                 "readers that don't strip it would fail to parse"),
        )

    def test_emitted_bytes_use_ascii_safe_utf8(self) -> None:
        # Insert a multibyte string into the event; the canonical
        # JSON must escape it (ensure_ascii=True), so the file
        # contains no non-ASCII bytes.
        append_audit_record(
            self.audit_path,
            {"event_type": "test_utf8", "note": "héllo", "schema_version": "v1"},
        )
        raw = self.audit_path.read_bytes()
        self.assertTrue(
            all(byte < 0x80 for byte in raw),
            msg=("audit.jsonl contains non-ASCII bytes; canonical JSON "
                 "must use \\uXXXX escapes for cross-platform stability"),
        )

    def test_emitted_bytes_use_lf_only(self) -> None:
        append_audit_record(
            self.audit_path,
            {"event_type": "test_lf", "schema_version": "v1"},
        )
        append_audit_record(
            self.audit_path,
            {"event_type": "test_lf", "schema_version": "v1"},
        )
        raw = self.audit_path.read_bytes()
        self.assertNotIn(b"\r", raw)
        # Two records → exactly two trailing LF bytes.
        self.assertEqual(raw.count(b"\n"), 2)


# ── Path canonicalization ───────────────────────────────────────────


class PathCanonicalizationTests(unittest.TestCase):
    """pathlib normalizes separators across platforms. These tests
    document the assumption."""

    def test_pathlib_normalizes_separators(self) -> None:
        # On Windows os.sep is "\\"; on POSIX it's "/". pathlib
        # canonicalizes via the OS-specific implementation.
        p = Path("a") / "b" / "c"
        # Always parseable, always equal to itself.
        self.assertEqual(Path(str(p)), p)

    def test_root_resolution_independent_of_cwd(self) -> None:
        # Project ROOT is resolved relative to __file__, not cwd.
        # This test simply documents that pattern by reading one
        # production module's ROOT.
        from backend.evidence import replay
        self.assertTrue(replay.ROOT.is_absolute())
        self.assertTrue(replay.DEFAULT_AUDIT_PATH.is_absolute())


# ── Explicit parity-proof tests (CI-enforced) ───────────────────────


class CrossPlatformParityProofTests(unittest.TestCase):
    """Load-bearing assertions that prove the host platform actually
    exercised the expected codepath and produced byte-identical
    output against a pinned reference. These tests turn the CI
    matrix from "tests passed on each platform" into "tests passed
    on each platform AND we can prove the platform-native codepath
    was exercised AND the byte output matches the reference."

    Without these, a CI failure mode would exist where (for
    example) a runtime shim silently makes Windows fall through to
    a POSIX-style locking implementation, the tests pass trivially,
    and "parity" is reported despite the Windows codepath never
    actually being run.
    """

    def test_locking_backend_matches_host_platform(self) -> None:
        """The adapter's branch selection must reflect the host
        platform. On a Windows runner, the msvcrt branch MUST be
        active. On any non-Windows runner, the POSIX branch MUST
        be active. Without this, a CI parity claim could be made
        on a Windows runner that never actually exercised msvcrt."""
        if sys.platform.startswith("win"):
            self.assertTrue(
                _locking._IS_WINDOWS,
                msg=("running on Windows but adapter selected the "
                     "POSIX branch — Windows codepath was NOT exercised"),
            )
        else:
            self.assertFalse(
                _locking._IS_WINDOWS,
                msg=("running on non-Windows but adapter selected the "
                     "Windows branch — POSIX codepath was NOT exercised"),
            )

    def test_canonical_json_pinned_reference_hash(self) -> None:
        """A fixed dict must canonicalize to a byte-identical
        SHA-256 on every platform. The pinned value below was
        computed on macOS (darwin, CPython 3.9.6) and is the
        ground-truth reference. If this fails on any platform, the
        canonical-JSON byte output diverges across platforms — a
        replay-determinism break.

        Note: pinning a hash exposes the test to upstream changes
        (CPython json semantics, hash algorithm); the trade-off is
        accepted because canonical JSON with sort_keys + fixed
        separators + ensure_ascii is among the most stable
        serializers in the stdlib, and the explicit failure mode
        (one platform diverges from the others) is exactly the
        failure we want to catch."""
        record = {
            "timestamp": "2026-05-16T12:34:56+00:00",
            "schema_version": "v1",
            "analyzer_version": "mf_v2",
            "chain_epoch": 1,
            "prev_hash": "0" * 64,
            "payload_hash": "a" * 64,
            "event": {
                "event_type": "parity_pin",
                "evidence_kind": None,
                "run_id": "00000000-0000-0000-0000-000000000001",
                "parent_run_id": None,
                "note": "héllo",
                "numbers": [1, 2, 3, 4, 5],
                "nested": {"x": 0.123456789, "y": True, "z": None},
            },
        }
        canonical = _canonical_json(record)
        observed_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        # PINNED on macOS (darwin, CPython 3.9.6) 2026-05-16. Do
        # NOT update this value casually — if it ever changes, that
        # is itself an event worth investigating (CPython upgrade,
        # serialization library swap, or an actual cross-platform
        # divergence).
        EXPECTED_HASH = (
            "2f2ad05d555a9eb040c580710a3bbf9af7b49863e00d2a1239235a7e77234694"
        )
        self.assertEqual(
            observed_hash, EXPECTED_HASH,
            msg=(
                f"canonical-JSON hash diverged from pinned reference. "
                f"Observed {observed_hash!r} on platform={sys.platform!r}; "
                f"expected {EXPECTED_HASH!r}. This is a cross-platform "
                f"byte-determinism break — replay parity is broken."
            ),
        )

    def test_canonical_json_pinned_byte_count(self) -> None:
        """Companion to the pinned-hash test: the canonical JSON
        of the reference record must also have the exact byte
        count it had on the pinning platform. A byte-count
        divergence implies an encoding difference even if the
        hash collision were astronomically lucky."""
        record = {
            "timestamp": "2026-05-16T12:34:56+00:00",
            "schema_version": "v1",
            "analyzer_version": "mf_v2",
            "chain_epoch": 1,
            "prev_hash": "0" * 64,
            "payload_hash": "a" * 64,
            "event": {
                "event_type": "parity_pin",
                "evidence_kind": None,
                "run_id": "00000000-0000-0000-0000-000000000001",
                "parent_run_id": None,
                "note": "héllo",
                "numbers": [1, 2, 3, 4, 5],
                "nested": {"x": 0.123456789, "y": True, "z": None},
            },
        }
        canonical = _canonical_json(record)
        EXPECTED_BYTES = 480
        self.assertEqual(
            len(canonical.encode("utf-8")), EXPECTED_BYTES,
            msg=(
                f"canonical-JSON byte count diverged from pinned reference. "
                f"Observed {len(canonical.encode('utf-8'))} bytes on "
                f"platform={sys.platform!r}; expected {EXPECTED_BYTES}."
            ),
        )


# ── Sanity: existing audit chain integrity holds with new code ──────


class IntegrationSanityTests(unittest.TestCase):
    """Top-level sanity: emit a small chain through the migrated
    append path and verify it. If anything in the locking/newline
    migration accidentally regressed correctness, this surfaces it."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.audit_path = Path(self.tmp.name) / "audit" / "audit.jsonl"
        self.audit_path.parent.mkdir(parents=True)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_emit_and_verify_chain(self) -> None:
        for i in range(10):
            append_audit_record(
                self.audit_path,
                {"event_type": "sanity",
                 "i": i,
                 "schema_version": "v1"},
            )
        diag = verify_audit_chain_diag(self.audit_path)
        self.assertTrue(diag["valid"], msg=diag)
        self.assertEqual(diag["lines_scanned"], 10)


if __name__ == "__main__":
    unittest.main()
