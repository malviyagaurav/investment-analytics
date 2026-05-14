"""Error-path tests for backend.data_discovery.fetch.

Before this file, fetch.py was at 48.6% coverage — happy paths only.
This file exercises the failure modes that a real production
deployment will hit at some point: mfapi.in timeouts, 404s, 5xx
errors, malformed JSON, partial NAV data, date-parse failures,
self-benchmark fallback.

All tests mock httpx so no network calls leave the test runner.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx

from backend.data_discovery import cache as cache_mod
from backend.data_discovery import fetch as fetch_mod


def _fake_response(status_code=200, json_payload=None, text=""):
    """Build a minimal httpx.Response stand-in. We can't easily build a
    real httpx.Response without a Request, so use a MagicMock that
    behaves like one for our purposes."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    if json_payload is not None:
        resp.json = MagicMock(return_value=json_payload)
    else:
        resp.json = MagicMock(side_effect=json.JSONDecodeError("expecting value", "", 0))
    resp.text = text
    if status_code >= 400:
        request = MagicMock(spec=httpx.Request)
        request.method = "GET"
        request.url = "https://api.mfapi.in/mf/0"
        resp.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                f"{status_code} error", request=request, response=resp,
            )
        )
    else:
        resp.raise_for_status = MagicMock(return_value=None)
    return resp


def _client_returning(response):
    """Make a context-manager mock for httpx.Client that yields a
    client whose .get returns the given response."""
    client = MagicMock()
    client.get = MagicMock(return_value=response)
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=client)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


def _client_raising(exc):
    """Make a context-manager mock whose .get raises exc."""
    client = MagicMock()
    client.get = MagicMock(side_effect=exc)
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=client)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


class FetchSchemeNavTests(unittest.TestCase):

    def setUp(self) -> None:
        # Isolate cache so test runs don't pollute the real data/cache/.
        self.tmp = tempfile.TemporaryDirectory()
        self._orig_cache = cache_mod.CACHE_DIR
        cache_mod.CACHE_DIR = Path(self.tmp.name)

    def tearDown(self) -> None:
        cache_mod.CACHE_DIR = self._orig_cache
        self.tmp.cleanup()

    def test_cache_hit_returns_cached_without_http_call(self) -> None:
        cache_mod.put_cached_nav(101, {"data": [{"date": "06-05-2026", "nav": "100.5"}]})
        with patch.object(fetch_mod.httpx, "Client") as client_factory:
            result = fetch_mod.fetch_scheme_nav(101)
        self.assertEqual(result, {"data": [{"date": "06-05-2026", "nav": "100.5"}]})
        client_factory.assert_not_called()

    def test_cache_miss_fetches_and_persists(self) -> None:
        body = {"data": [{"date": "06-05-2026", "nav": "100.5"}]}
        with patch.object(fetch_mod.httpx, "Client",
                          return_value=_client_returning(_fake_response(200, body))):
            result = fetch_mod.fetch_scheme_nav(202)
        self.assertEqual(result, body)
        # And the cache now has it for next time.
        self.assertEqual(cache_mod.get_cached_nav(202), body)

    def test_404_raises_http_status_error(self) -> None:
        with patch.object(fetch_mod.httpx, "Client",
                          return_value=_client_returning(_fake_response(404))):
            with self.assertRaises(httpx.HTTPStatusError):
                fetch_mod.fetch_scheme_nav(99999)

    def test_500_raises_http_status_error(self) -> None:
        with patch.object(fetch_mod.httpx, "Client",
                          return_value=_client_returning(_fake_response(503))):
            with self.assertRaises(httpx.HTTPStatusError):
                fetch_mod.fetch_scheme_nav(101)

    def test_timeout_bubbles_up(self) -> None:
        with patch.object(fetch_mod.httpx, "Client",
                          return_value=_client_raising(httpx.ReadTimeout("upstream slow"))):
            with self.assertRaises(httpx.ReadTimeout):
                fetch_mod.fetch_scheme_nav(101)

    def test_malformed_json_bubbles_up(self) -> None:
        with patch.object(fetch_mod.httpx, "Client",
                          return_value=_client_returning(_fake_response(200, None))):
            with self.assertRaises(json.JSONDecodeError):
                fetch_mod.fetch_scheme_nav(101)


class ParseDdMmYyyyTests(unittest.TestCase):

    def test_valid_format(self) -> None:
        from datetime import date
        self.assertEqual(fetch_mod._parse_dd_mm_yyyy("06-05-2026"), date(2026, 5, 6))

    def test_strips_whitespace(self) -> None:
        from datetime import date
        self.assertEqual(fetch_mod._parse_dd_mm_yyyy("  06-05-2026  "), date(2026, 5, 6))

    def test_iso_format_returns_none(self) -> None:
        # ISO is YYYY-MM-DD, the function expects DD-MM-YYYY; reject.
        self.assertIsNone(fetch_mod._parse_dd_mm_yyyy("2026-05-06"))

    def test_garbage_returns_none(self) -> None:
        self.assertIsNone(fetch_mod._parse_dd_mm_yyyy("not a date"))

    def test_none_input_returns_none(self) -> None:
        # Caller passes entry.get("date", "") so should always be str,
        # but defensive guard exists for AttributeError.
        self.assertIsNone(fetch_mod._parse_dd_mm_yyyy(None))  # type: ignore[arg-type]

    def test_empty_string_returns_none(self) -> None:
        self.assertIsNone(fetch_mod._parse_dd_mm_yyyy(""))


class ConvertNavToRecordsTests(unittest.TestCase):

    def test_normal_conversion_oldest_first(self) -> None:
        # mfapi returns newest-first; function reverses to oldest-first.
        nav = [
            {"date": "08-05-2026", "nav": "102.0"},  # newest
            {"date": "07-05-2026", "nav": "101.0"},
            {"date": "06-05-2026", "nav": "100.0"},  # oldest
        ]
        out = fetch_mod._convert_nav_to_records(nav)
        self.assertEqual(len(out), 3)
        # First entry must now be the oldest.
        self.assertEqual(out[0]["date"], "2026-05-06")
        self.assertEqual(out[0]["nav"], 100.0)
        self.assertEqual(out[-1]["date"], "2026-05-08")

    def test_skips_entries_with_invalid_date(self) -> None:
        nav = [
            {"date": "08-05-2026", "nav": "102.0"},
            {"date": "garbage", "nav": "101.0"},
            {"date": "06-05-2026", "nav": "100.0"},
        ]
        out = fetch_mod._convert_nav_to_records(nav)
        self.assertEqual(len(out), 2)
        self.assertEqual({r["date"] for r in out}, {"2026-05-06", "2026-05-08"})

    def test_skips_entries_with_unparseable_nav(self) -> None:
        nav = [
            {"date": "08-05-2026", "nav": "102.0"},
            {"date": "07-05-2026", "nav": "not_a_number"},
            {"date": "06-05-2026", "nav": None},  # TypeError on float(None)
            {"date": "05-05-2026"},  # KeyError on "nav"
        ]
        out = fetch_mod._convert_nav_to_records(nav)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["date"], "2026-05-08")

    def test_skips_zero_and_negative_navs(self) -> None:
        nav = [
            {"date": "08-05-2026", "nav": "102.0"},
            {"date": "07-05-2026", "nav": "0"},
            {"date": "06-05-2026", "nav": "-1.5"},
        ]
        out = fetch_mod._convert_nav_to_records(nav)
        self.assertEqual([r["date"] for r in out], ["2026-05-08"])

    def test_caps_at_max_nav_points(self) -> None:
        # Build > MAX_NAV_POINTS entries. Function must trim to the
        # most-recent MAX_NAV_POINTS by oldest-first slicing [-N:].
        from datetime import date, timedelta
        big = []
        d = date(2000, 1, 1)
        for i in range(fetch_mod.MAX_NAV_POINTS + 50):
            big.append({"date": d.strftime("%d-%m-%Y"), "nav": str(100 + i)})
            d += timedelta(days=1)
        # The list comes in oldest-first here; the function expects
        # newest-first per mfapi convention. Reverse to mimic mfapi.
        out = fetch_mod._convert_nav_to_records(list(reversed(big)))
        self.assertEqual(len(out), fetch_mod.MAX_NAV_POINTS)


class IngestFromMfapiTests(unittest.TestCase):
    """End-to-end ingest_from_mfapi paths — happy + benchmark fallback."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self._orig_cache = cache_mod.CACHE_DIR
        cache_mod.CACHE_DIR = Path(self.tmp.name)

    def tearDown(self) -> None:
        cache_mod.CACHE_DIR = self._orig_cache
        self.tmp.cleanup()

    def _nav_blob(self, n_points: int, start_nav: float = 100.0):
        """Build an mfapi-style payload (newest-first) with n_points entries."""
        from datetime import date, timedelta
        d = date(2018, 1, 1)
        entries = []
        for i in range(n_points):
            entries.append({"date": d.strftime("%d-%m-%Y"),
                            "nav": f"{start_nav + i * 0.05:.4f}"})
            d += timedelta(days=1)
        entries.reverse()  # mfapi convention: newest first
        return {"meta": {"scheme_name": "Test Fund", "scheme_category": "Equity Scheme - Large Cap Fund"},
                "data": entries}

    def test_self_benchmark_when_scheme_is_the_benchmark(self) -> None:
        """Holding the Nifty 50 index fund (120716) — fall back to self
        benchmark with explicit limitation flag."""
        bench_code = fetch_mod.DEFAULT_BENCHMARK[0]
        body = self._nav_blob(50)
        with patch.object(fetch_mod.httpx, "Client",
                          return_value=_client_returning(_fake_response(200, body))):
            payload, report = fetch_mod.ingest_from_mfapi(
                scheme_code=bench_code,
                fund_name="UTI Nifty 50",
                category="Equity Scheme - Large Cap Fund",
            )
        # Self-benchmark name reflects fallback.
        self.assertIn("self", payload["benchmark_name"])
        # And the limitation is surfaced.
        self.assertTrue(
            any("self-referencing" in lim.lower()
                for lim in report["ingestion_limitations"]),
            f"missing self-benchmark limitation: {report['ingestion_limitations']}",
        )

    def test_benchmark_fetch_failure_falls_back_to_self(self) -> None:
        """If mfapi.in is down for the benchmark code, the ingest
        path must NOT crash — it falls back to self-benchmark and
        records the limitation."""
        body = self._nav_blob(50)

        # First call (the fund itself) succeeds; second call (benchmark)
        # raises. Use a stateful side_effect.
        call_count = [0]

        def client_factory(*args, **kwargs):
            cm = MagicMock()

            def enter(*a):
                call_count[0] += 1
                client = MagicMock()
                if call_count[0] == 1:
                    client.get = MagicMock(return_value=_fake_response(200, body))
                else:
                    client.get = MagicMock(side_effect=httpx.ReadTimeout("bench down"))
                return client

            cm.__enter__ = enter
            cm.__exit__ = MagicMock(return_value=False)
            return cm

        with patch.object(fetch_mod.httpx, "Client", side_effect=client_factory):
            payload, report = fetch_mod.ingest_from_mfapi(
                scheme_code=12345,
                fund_name="X",
                category="Equity Scheme - Large Cap Fund",
            )
        self.assertIn("self", payload["benchmark_name"])
        self.assertTrue(
            any("self-referencing" in lim.lower()
                for lim in report["ingestion_limitations"]),
        )

    def test_empty_nav_data_raises_value_error(self) -> None:
        """mfapi returned a 200 OK but with empty data array. The
        ingest path must produce a clear error, not crash mysteriously
        further down."""
        body = {"meta": {}, "data": []}
        with patch.object(fetch_mod.httpx, "Client",
                          return_value=_client_returning(_fake_response(200, body))):
            with self.assertRaises(ValueError) as ctx:
                fetch_mod.ingest_from_mfapi(scheme_code=12345)
        self.assertIn("No NAV data", str(ctx.exception))

    def test_too_few_records_after_filter_raises(self) -> None:
        """All NAV entries had bad dates → only 1 record survives → fail."""
        body = {"meta": {}, "data": [{"date": "garbage", "nav": "100"}]}
        with patch.object(fetch_mod.httpx, "Client",
                          return_value=_client_returning(_fake_response(200, body))):
            with self.assertRaises(ValueError):
                fetch_mod.ingest_from_mfapi(scheme_code=12345)


class ResolveBenchmarkTests(unittest.TestCase):

    def test_known_category_returns_mapped_benchmark(self) -> None:
        code, name = fetch_mod._resolve_benchmark("Equity Scheme - Large Cap Fund")
        self.assertEqual(code, 120716)

    def test_unknown_category_returns_default(self) -> None:
        code, name = fetch_mod._resolve_benchmark("Some Unknown Category")
        self.assertEqual((code, name), fetch_mod.DEFAULT_BENCHMARK)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
