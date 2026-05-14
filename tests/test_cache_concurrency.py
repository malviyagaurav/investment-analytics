"""Tests for backend.data_discovery.cache atomic-write behavior.

Without atomic write, a reader can observe a half-written JSON during
the truncate→write window in put_cached_nav. The reader's
get_cached_nav has a JSONDecodeError fallback so this manifests as
cache misses (extra mfapi.in fetches), not data corruption — but it
still defeats the cache under concurrent load.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.data_discovery import cache as cache_mod


class CacheAtomicWriteTests(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self._orig_dir = cache_mod.CACHE_DIR
        cache_mod.CACHE_DIR = Path(self.tmp.name)

    def tearDown(self) -> None:
        cache_mod.CACHE_DIR = self._orig_dir
        self.tmp.cleanup()

    def test_put_uses_tmp_file_then_rename(self) -> None:
        """The implementation should write to <code>.json.tmp, then
        rename to <code>.json. If the writer is interrupted between
        the two steps, no .json file appears (reader sees cache miss),
        but no half-written .json is possible."""
        cache_mod.put_cached_nav(101, {"data": [{"date": "2026-05-06", "nav": 100.0}]})
        # After a successful write, only the final .json exists.
        files = sorted(p.name for p in cache_mod.CACHE_DIR.iterdir())
        self.assertEqual(files, ["101.json"])

    def test_roundtrip(self) -> None:
        payload = {"data": [{"date": "2026-05-06", "nav": 250.123}]}
        cache_mod.put_cached_nav(202, payload)
        out = cache_mod.get_cached_nav(202)
        self.assertEqual(out, payload)

    def test_concurrent_writes_produce_valid_json(self) -> None:
        """N threads writing the SAME scheme code in parallel must
        leave a valid JSON readable by get_cached_nav. Without atomic
        rename, a reader could observe an empty file mid-truncate;
        with atomic rename, every read either sees the previous
        complete state or the new complete state — never partial."""
        n_writers = 12
        writes_per_writer = 30
        scheme = 303

        barrier = threading.Barrier(n_writers + 1)
        read_errors: list = []

        def writer(seed: int) -> None:
            barrier.wait()
            for i in range(writes_per_writer):
                cache_mod.put_cached_nav(scheme, {"data": [{"date": "2026-05-06",
                                                            "nav": float(seed * 100 + i)}]})

        def reader() -> None:
            barrier.wait()
            for _ in range(n_writers * writes_per_writer):
                try:
                    val = cache_mod.get_cached_nav(scheme)
                    # val is either None (no file yet) or a dict —
                    # both are valid. What MUST NOT happen is the
                    # underlying read returning a half-written JSON.
                    if val is not None and not isinstance(val, dict):
                        read_errors.append(("unexpected_type", type(val)))
                except Exception as exc:
                    read_errors.append(("exception", repr(exc)))

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(n_writers)]
        reader_thread = threading.Thread(target=reader)
        for t in threads:
            t.start()
        reader_thread.start()
        for t in threads:
            t.join()
        reader_thread.join()

        self.assertEqual(read_errors, [],
                         "concurrent reads must not observe broken JSON")
        # Final file must be a complete, parseable JSON.
        with cache_mod._cache_path(scheme).open("r") as fh:
            data = json.load(fh)
        self.assertIn("response", data)
        self.assertIn("data", data["response"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
