"""Tests for the per-call NAV cache read tracking module.

Step 4 of the evidence-layer roadmap. Covers:
  - start_tracking / record_cache_read / cache_fingerprint flow
    produces a deterministic 64-char sha256 over the consumed entries.
  - When no session is active, record_cache_read is a no-op and
    cache_fingerprint() returns None — preserving CLI / ad-hoc usage.
  - An empty session (start_tracking, no reads, cache_fingerprint)
    returns a stable non-None sentinel — the sha256 of an empty list.
  - Thread isolation: two threads in concurrent tracking sessions do
    not bleed reads into each other's fingerprints.
  - Duplicate record_cache_read calls for the same scheme_code are
    idempotent — same mtime+size overwrites cleanly.
  - Fingerprint is order-independent: same set of reads in any order
    yields the same hash.
  - stop_tracking clears session state — a subsequent
    cache_fingerprint() returns None until start_tracking is called
    again.
  - cache.get_cached_nav reports to the tracker on a hit, contributing
    to the fingerprint end-to-end.
"""
from __future__ import annotations

import hashlib
import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from backend.data_discovery import cache_tracker


class TestCacheTrackerBasics(unittest.TestCase):

    def setUp(self) -> None:
        # Always clean state per test — the module uses thread-local
        # storage, but a leak from a prior test could mask a bug.
        cache_tracker.stop_tracking()

    def tearDown(self) -> None:
        cache_tracker.stop_tracking()

    def test_no_session_returns_none(self) -> None:
        # No start_tracking — fingerprint is unambiguously None, not
        # an empty-set sha. Callers distinguish "no session" from
        # "session with no reads".
        self.assertIsNone(cache_tracker.cache_fingerprint())
        self.assertFalse(cache_tracker.is_active())

    def test_record_outside_session_is_noop(self) -> None:
        # record_cache_read MUST silently no-op when no session is
        # active so cache.get_cached_nav can call it unconditionally.
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "12345.json"
            path.write_text("{}", encoding="utf-8")
            cache_tracker.record_cache_read(path, 12345)
            # Still no session, still no fingerprint.
            self.assertIsNone(cache_tracker.cache_fingerprint())

    def test_empty_session_returns_stable_non_null_fingerprint(self) -> None:
        # The sha256 of an empty list is a REAL value — explicitly
        # not None — so callers can prove they were inside an active
        # session that consumed nothing.
        cache_tracker.start_tracking()
        fp = cache_tracker.cache_fingerprint()
        self.assertIsNotNone(fp)
        self.assertEqual(len(fp), 64)
        # Deterministic: sha256 over canonical empty-list JSON.
        expected = hashlib.sha256(b"[]").hexdigest()
        self.assertEqual(fp, expected)

    def test_basic_flow(self) -> None:
        # start → record N reads → fingerprint reflects them as a
        # sorted JSON list keyed by scheme_code.
        with TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            paths = []
            for code in (200, 100, 300):
                p = tmp_dir / f"{code}.json"
                p.write_text(json.dumps({"scheme_code": code}), encoding="utf-8")
                paths.append((p, code))
            cache_tracker.start_tracking()
            for p, code in paths:
                cache_tracker.record_cache_read(p, code)
            fp = cache_tracker.cache_fingerprint()
            self.assertEqual(len(fp), 64)
            # Re-derive by hand to confirm canonical form.
            sorted_entries = sorted(
                ({"scheme_code": code, "mtime_ns": p.stat().st_mtime_ns, "size_bytes": p.stat().st_size}
                 for p, code in paths),
                key=lambda d: d["scheme_code"],
            )
            canonical = json.dumps(sorted_entries, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            self.assertEqual(fp, expected)

    def test_duplicate_scheme_code_idempotent(self) -> None:
        # Calling record_cache_read twice for the same scheme_code
        # MUST NOT double-count or change the fingerprint compared to
        # one call. Mtime+size for the same file is deterministic.
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "555.json"
            p.write_text("{}", encoding="utf-8")
            cache_tracker.start_tracking()
            cache_tracker.record_cache_read(p, 555)
            once = cache_tracker.cache_fingerprint()
            cache_tracker.record_cache_read(p, 555)
            cache_tracker.record_cache_read(p, 555)
            twice = cache_tracker.cache_fingerprint()
            self.assertEqual(once, twice)

    def test_order_independent(self) -> None:
        # Reading [A, B, C] and [C, B, A] produce IDENTICAL fingerprints
        # — order-independence is a core requirement for replay.
        with TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            files = []
            for code in (1, 2, 3, 4, 5):
                p = tmp_dir / f"{code}.json"
                p.write_text(f"{{\"c\": {code}}}", encoding="utf-8")
                files.append((p, code))
            cache_tracker.start_tracking()
            for p, code in files:
                cache_tracker.record_cache_read(p, code)
            fp_ascending = cache_tracker.cache_fingerprint()
            cache_tracker.stop_tracking()
            cache_tracker.start_tracking()
            for p, code in reversed(files):
                cache_tracker.record_cache_read(p, code)
            fp_descending = cache_tracker.cache_fingerprint()
            self.assertEqual(fp_ascending, fp_descending)

    def test_stop_tracking_clears_state(self) -> None:
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "1.json"
            p.write_text("{}", encoding="utf-8")
            cache_tracker.start_tracking()
            cache_tracker.record_cache_read(p, 1)
            self.assertIsNotNone(cache_tracker.cache_fingerprint())
            cache_tracker.stop_tracking()
            self.assertIsNone(cache_tracker.cache_fingerprint())
            self.assertFalse(cache_tracker.is_active())

    def test_stop_tracking_idempotent(self) -> None:
        # Used in finally blocks — calling twice (or without prior
        # start) is safe.
        cache_tracker.stop_tracking()
        cache_tracker.stop_tracking()
        self.assertFalse(cache_tracker.is_active())

    def test_missing_path_does_not_poison_fingerprint(self) -> None:
        # If a cache file vanished between read and stat, the tracker
        # MUST skip it silently rather than inject a sentinel that
        # would change the fingerprint depending on transient FS state.
        cache_tracker.start_tracking()
        cache_tracker.record_cache_read(Path("/tmp/this-does-not-exist-99999.json"), 99999)
        fp = cache_tracker.cache_fingerprint()
        # Should match empty-session fingerprint (no entries recorded).
        self.assertEqual(fp, hashlib.sha256(b"[]").hexdigest())


class TestCacheTrackerThreadIsolation(unittest.TestCase):

    def test_threads_do_not_bleed(self) -> None:
        # Two concurrent tracking sessions in different threads MUST
        # produce independent fingerprints — threading.local prevents
        # cross-contamination of one request's cache reads into
        # another's evidence record.
        with TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            shared_a = tmp_dir / "100.json"
            shared_b = tmp_dir / "200.json"
            shared_a.write_text("{}", encoding="utf-8")
            shared_b.write_text("{}", encoding="utf-8")

            results: dict[str, str] = {}
            barrier = threading.Barrier(2)

            def thread_a() -> None:
                cache_tracker.start_tracking()
                try:
                    cache_tracker.record_cache_read(shared_a, 100)
                    barrier.wait()
                    # Other thread now records its own — should not
                    # affect ours.
                    barrier.wait()
                    results["a"] = cache_tracker.cache_fingerprint()
                finally:
                    cache_tracker.stop_tracking()

            def thread_b() -> None:
                cache_tracker.start_tracking()
                try:
                    barrier.wait()
                    cache_tracker.record_cache_read(shared_b, 200)
                    barrier.wait()
                    results["b"] = cache_tracker.cache_fingerprint()
                finally:
                    cache_tracker.stop_tracking()

            t_a = threading.Thread(target=thread_a)
            t_b = threading.Thread(target=thread_b)
            t_a.start()
            t_b.start()
            t_a.join()
            t_b.join()

            # Each thread should see ONLY its own read.
            self.assertNotEqual(results["a"], results["b"])
            # Sanity: both sessions saw exactly one entry, just
            # different ones.
            expected_a = hashlib.sha256(
                json.dumps(
                    [{"scheme_code": 100, "mtime_ns": shared_a.stat().st_mtime_ns,
                      "size_bytes": shared_a.stat().st_size}],
                    sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                ).encode("utf-8")
            ).hexdigest()
            self.assertEqual(results["a"], expected_a)


class TestCacheTrackerIntegrationWithCache(unittest.TestCase):
    """cache.get_cached_nav must report to the tracker on a hit so
    the fingerprint reflects actual data reads end-to-end."""

    def setUp(self) -> None:
        cache_tracker.stop_tracking()

    def tearDown(self) -> None:
        cache_tracker.stop_tracking()

    def test_get_cached_nav_reports_to_tracker(self) -> None:
        # Patch CACHE_DIR to a temp directory so we don't disturb the
        # real cache and avoid cross-test interference.
        with TemporaryDirectory() as tmp:
            tmp_cache = Path(tmp) / "cache"
            tmp_cache.mkdir()
            scheme_code = 12345
            from datetime import datetime, timezone
            # Pre-populate a fresh cache entry.
            entry = {
                "fetched_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "scheme_code": scheme_code,
                "response": {"data": [{"date": "01-01-2020", "nav": "100.0"}]},
            }
            cache_path = tmp_cache / f"{scheme_code}.json"
            cache_path.write_text(json.dumps(entry), encoding="utf-8")

            with patch("backend.data_discovery.cache.CACHE_DIR", tmp_cache):
                from backend.data_discovery.cache import get_cached_nav
                cache_tracker.start_tracking()
                response = get_cached_nav(scheme_code)
                self.assertIsNotNone(response)
                fp = cache_tracker.cache_fingerprint()
                # Sha over the one read.
                expected = hashlib.sha256(
                    json.dumps(
                        [{"scheme_code": scheme_code,
                          "mtime_ns": cache_path.stat().st_mtime_ns,
                          "size_bytes": cache_path.stat().st_size}],
                        sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                    ).encode("utf-8")
                ).hexdigest()
                self.assertEqual(fp, expected)


if __name__ == "__main__":
    unittest.main()
